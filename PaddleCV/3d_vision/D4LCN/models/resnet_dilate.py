# import torch.nn as nn
from lib.rpn_util import *
from models.backbone import resnet
#from backbone import resnet
# import torch
import numpy as np
# from models.deform_conv_v2 import *
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Dropout
from paddle.fluid.dygraph.container import Sequential
from paddle.fluid.initializer import Normal
from models.deform_conv_v2 import DeformConv2DV2


def initial_type(name,
                 input_channels,
                 init="kaiming",
                 use_bias=False,
                 filter_size=0,
                 stddev=0.02):
    if init == "kaiming":
        fan_in = input_channels * filter_size * filter_size
        bound = 1 / math.sqrt(fan_in)
        param_attr = fluid.ParamAttr(
            name=name + "weight",
            initializer=fluid.initializer.Uniform(
                low=-bound, high=bound))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + 'bias',
                initializer=fluid.initializer.Uniform(
                    low=-bound, high=bound))
        else:
            bias_attr = False
    else:
        param_attr = fluid.ParamAttr(
            name=name + "weight",
            initializer=fluid.initializer.NormalInitializer(
                loc=0.0, scale=stddev))
        if use_bias == True:
            bias_attr = fluid.ParamAttr(
                name=name + "bias", initializer=fluid.initializer.Constant(0.0))
        else:
            bias_attr = False
    return param_attr, bias_attr
        

def dynamic_local_filtering(x, depth, dilated=1):
    # shift pooling
    y = fluid.layers.concat([x[:, -1:, :, :], x[:, :-1, :, :]], axis=1)
    z = fluid.layers.concat([x[:, -2:, :, :], x[:, :-2, :, :]], axis=1)
    x = (x + y + z) / 3.

    h = int(x.shape[2])
    w = int(x.shape[3])

    # pad x & depth for XXX
    paddings = [dilated] * 4
    x = fluid.layers.pad2d(x, paddings=paddings, mode='reflect')
    depth = fluid.layers.pad2d(depth, paddings=paddings, mode='reflect')

    out = depth[:, :, dilated: -dilated, dilated: -dilated] * x[:, :, dilated: -dilated, dilated: -dilated]
    for i in [-dilated, 0, dilated]:
        for j in [-dilated, 0, dilated]:
            if i != 0 or j != 0:
                out += depth[:, :, dilated + i: h + dilated + i, dilated + j: w + dilated + j] * x[:, :, dilated + i: h + dilated + i, dilated + j: w + dilated + j]
    return out / 9.


class ConvLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 padding=0,
                 stride=1,
                 groups=None,
                 act=None,
                 name=None):
        super(ConvLayer, self).__init__()
        
        param_attr, bias_attr = initial_type(
                    name=name,
                    input_channels=num_channels,
                    use_bias=True,
                    filter_size=filter_size)
        
        self.num_filters = num_filters
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            padding=padding,
            stride=stride,
            groups=groups,
            act=act,
            param_attr=param_attr,
            bias_attr=bias_attr)

    def forward(self, inputs):
        x = self._conv(inputs)        
        return x


class AdaptiveDiatedLayer(fluid.dygraph.Layer):
    def __init__(self,
                 pool_size,
                 pool_type,
                 num_channels,
                 num_filters,
                 filter_size,
                 padding=0,
                 stride=1,
                 groups=None,
                 act='relu',
                 name=None):
        super(AdaptiveDiatedLayer, self).__init__()

        self.num_channels = num_channels

        param_attr, _ = initial_type(name=name,
                                  input_channels=num_channels,
                                  filter_size=filter_size)
        self.pool_size = pool_size
        self.pool_type = pool_type
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            padding=padding,
            stride=stride,
            groups=groups,
            act=None,
            param_attr=param_attr)

        bn_name = name + "_bn"
        self._bn = BatchNorm(num_channels,
                             param_attr=ParamAttr(name=bn_name + "weight"),
                             bias_attr=ParamAttr(name=bn_name + "bias"),
                             moving_mean_name=bn_name + "mean",
                             moving_variance_name=bn_name + "variance",
                             act=act)

    def forward(self, x, depth):
        
        weight = fluid.layers.adaptive_pool2d(x, pool_size=self.pool_size, pool_type=self.pool_type)
        weight = self._conv(weight)
        weight = fluid.layers.reshape(weight, [-1, self.num_channels, 1, 3])
        weight = fluid.layers.softmax(weight, axis=-1)
        x = dynamic_local_filtering(x, depth, dilated=1) * weight[:, :, :, 0:1] \
                + dynamic_local_filtering(x, depth, dilated=2) * weight[:, :, :, 1:2] \
                + dynamic_local_filtering(x, depth, dilated=3) * weight[:, :, :, 2:3]
        x = self._bn(x)

        return x


class DeformLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 deformable_groups=None,
                 im2col_step=None,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32',
                 name=None):
        assert param_attr is not False, "param_attr should not be False here."
        super(DeformLayer, self).__init__()
        self.p_conv = ConvLayer(num_channels=num_channels,
                                num_filters=2*filter_size*filter_size,
                                filter_size=3,
                                padding=1,
                                stride=stride,
                                name=name+'offset_conv')
        self.m_conv = ConvLayer(num_channels=num_channels,
                                num_filters=filter_size*filter_size,
                                filter_size=3,
                                padding=1,
                                stride=stride,
                                name=name+'mask_conv')
        self.deform_conv = DeformConv2DV2(num_channels=num_channels,
                                          num_filters=num_filters,
                                          filter_size=filter_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          deformable_groups=deformable_groups,
                                          im2col_step=im2col_step,
                                          param_attr=param_attr,
                                          bias_attr=bias_attr,
                                          dtype=dtype)

    def forward(self, x):
        offset = self.p_conv(x)
        mask = self.m_conv(x)
        return self.deform_conv(x, offset, mask)

class RPN(fluid.dygraph.Layer):


    def __init__(self, phase, conf):
        super(RPN, self).__init__()

        self.base = resnet.ResNetDilate(conf.base_model)
        self.adaptive_diated = conf.adaptive_diated
        self.dropout_position = conf.dropout_position
        self.use_dropout = conf.use_dropout
        self.drop_channel = conf.drop_channel
        self.use_corner = conf.use_corner
        self.corner_in_3d = conf.corner_in_3d
        self.deformable = conf.deformable

        self.depthnet = resnet.ResNetDilate(50)

        if self.deformable:
            self.deform_layer = DeformLayer(512, 512, 3, padding=1, bias_attr=False, name="deform_layer")

        if self.adaptive_diated:
            self.adaptive_layer = AdaptiveDiatedLayer(3, 'max', 512, 512 * 3, 3, padding=0, name='adaptive_diate1')
            self.adaptive_layer1 = AdaptiveDiatedLayer(3, 'max', 1024, 1024 * 3, 3, padding=0, name='adaptive_diated2')

        # settings
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]
        #self.num_anchors = 32

        self.prop_feats = ConvLayer(num_channels=2048,
                                    num_filters=512,
                                    filter_size=3,
                                    padding=1,
                                    act='relu',
                                    name='rpn_prop_feats')
        if self.use_dropout:
            self.dropout = Dropout(p=conf.dropout_rate, dropout_implementation='upscale_in_train')

        if self.drop_channel:
            self.drop_channel = Dropout(p=0.3, dropout_implementation='upscale_in_train')

        self.cls = ConvLayer(num_channels=self.prop_feats.num_filters, 
                             num_filters=self.num_classes*self.num_anchors,
                             filter_size=1,
                             name='rpn_cls')

        self.bbox_x = ConvLayer(num_channels=self.prop_feats.num_filters, 
                                num_filters=self.num_anchors,
                                filter_size=1,
                                name='rpn_bbox_x')
        
        self.bbox_y = ConvLayer(num_channels=self.prop_feats.num_filters, 
                                num_filters=self.num_anchors,
                                filter_size=1,
                                name='rpn_bbox_y')

        self.bbox_w = ConvLayer(num_channels=self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                name='rpn_bbox_w')

        self.bbox_h = ConvLayer(num_channels=self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                name='rpn_bbox_h')

        self.bbox_x3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  name='rpn_bbox_x3d')

        self.bbox_y3d = ConvLayer(num_channels=self.prop_feats.num_filters, 
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  name='rpn_bbox_y3d')

        self.bbox_z3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  name='rpn_bbox_z3d')
        
        self.bbox_w3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  name='rpn_bbox_w3d')

        self.bbox_h3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  name='rpn_bbox_h3d')

        self.bbox_l3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  name='rpn_bbox_l3d')

        self.bbox_rY3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                   num_filters=self.num_anchors,
                                   filter_size=1,
                                   name='rpn_bbox_rY3d')

        if self.corner_in_3d:
            self.bbox_3d_corners = ConvLayer(512, self.num_anchors * 18, 1, name='bbox_3d_corners')
            self.bbox_vertices = ConvLayer(512, self.num_anchors * 24, 1, name='bbox_vertices')
        elif self.use_corner:
            self.bbox_vertices = ConvLayer(512, self.num_anchors * 24, 1, name='bbox_vertices')
        
        self.feat_stride = conf.feat_stride

        
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride)
        self.anchors = conf.anchors

    def forward(self, x, depth):
        batch_size = x.shape[0]
        
        x = self.base.conv(x)
        
        depth = self.depthnet.conv(depth)
        x = self.base.pool(x)
        depth = self.depthnet.pool(depth)

        x = self.base.layer_0(x)
        depth = self.depthnet.layer_0(depth)
        
        # x = dynamic_local_filtering(x, depth, dilated=1) + dynamic_local_filtering(x, depth, dilated=2) + dynamic_local_filtering(x, depth, dilated=3)

        x = self.base.layer_1(x)
        
        depth = self.depthnet.layer_1(depth)

        if self.deformable:
            depth = self.deform_layer(depth)
            x = x * depth

        if self.adaptive_diated:
            x = self.adaptive_layer(x, depth) # TODO
        else:
            x = dynamic_local_filtering(x, depth, dilated=1) + dynamic_local_filtering(x, depth, dilated=2) + dynamic_local_filtering(x, depth, dilated=3)

        if self.use_dropout and self.dropout_position == 'adaptive':
            x = self.dropout(x)

        if self.drop_channel: # TODO
            nc = fluid.layers.ones_like(x[:, :, 0, 0])
            dropped_nc = self.drop_channel(nc)
            x = fluid.layers.elementwise_mul(x, nc, axis=0)

        x = self.base.layer_2(x)
        
        depth = self.depthnet.layer_2(depth)

        if self.adaptive_diated:
            x = self.adaptive_layer1(x, depth)
        else:
            x = x * depth
            

        # x = fluid.layers.expand(x, [1, 2, 1, 1])
        # depth = fluid.layers.expand(depth, [1, 2, 1, 1])


        x = self.base.layer_3(x)
        
        depth = self.depthnet.layer_3(depth)

        
        
        x = x * depth

        if self.use_dropout and self.dropout_position == 'early':
            x = self.dropout(x)

        prop_feats = self.prop_feats(x)

        if self.use_dropout and self.dropout_position == 'late':
            prop_feats = self.dropout(prop_feats)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)
        # targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY

        batch_size, c, feat_h, feat_w = cls.shape
        feat_size = fluid.layers.shape(cls)[2:4]

        # reshape for cross entropy
        cls = fluid.layers.reshape(x=cls, shape=[batch_size, self.num_classes, feat_h * self.num_anchors, feat_w])
        # score probabilities
        prob = fluid.layers.softmax(cls, axis=1)

        # reshape for consistency
        bbox_x = flatten_tensor(fluid.layers.reshape(x=bbox_x, shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_y = flatten_tensor(fluid.layers.reshape(x=bbox_y, shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_w = flatten_tensor(fluid.layers.reshape(x=bbox_w, shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_h = flatten_tensor(fluid.layers.reshape(x=bbox_h, shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))

        bbox_x3d = flatten_tensor(fluid.layers.reshape(x=bbox_x3d, shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_y3d = flatten_tensor(fluid.layers.reshape(x=bbox_y3d, shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_z3d = flatten_tensor(fluid.layers.reshape(x=bbox_z3d, shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_w3d = flatten_tensor(fluid.layers.reshape(x=bbox_w3d, shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_h3d = flatten_tensor(fluid.layers.reshape(x=bbox_h3d, shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_l3d = flatten_tensor(fluid.layers.reshape(x=bbox_l3d, shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))
        bbox_rY3d = flatten_tensor(fluid.layers.reshape(x=bbox_rY3d, shape=[batch_size, 1, feat_h * self.num_anchors, feat_w]))

        # bundle
        bbox_2d = fluid.layers.concat(input=[bbox_x, bbox_y, bbox_w, bbox_h], axis=2)
        bbox_3d = fluid.layers.concat(input=[bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d], axis=2)
        
        if self.corner_in_3d:
            corners_3d = self.bbox_3d_corners(prop_feats)
            corners_3d = fluid.layers.reshape(corners_3d, [batch_size, 18, feat_h * self.num_anchors, feat_w])
            corners_3d = flatten_tensor(corners_3d)
            bbox_vertices = self.bbox_vertices(prop_feats)
            corners_3d = fluid.layers.reshape(corners_3d, [batch_size, 24, feat_h * self.num_anchors, feat_w])
            corners_3d = flatten_tensor(corners_3d)
        elif self.use_corner:
            bbox_vertices = self.bbox_vertices(prop_feats)
            corners_3d = fluid.layers.reshape(corners_3d, [batch_size, 24, feat_h * self.num_anchors, feat_w])
            corners_3d = flatten_tensor(corners_3d)

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.phase == "train":
            # TODO
            return cls, prob, bbox_2d, bbox_3d, feat_size

        else:
            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                #self.feat_size = [feat_h, feat_w]
                #self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride)
                self.rois = locate_anchors(self.anchors, [feat_h, feat_w], self.feat_stride)
        
        return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


def build(conf, phase='train'):

    train = phase.lower() == 'train'

    rpn_net = RPN(phase, conf)

    # pretrain 
    if 'pretrained' in conf and conf.pretrained is not None:
        print("load pretrain model from ", conf.pretrained)
        src_weights, _ = fluid.load_dygraph(conf.pretrained)
        src_weights_depthnet = src_weights.copy()
        #pdb.set_trace()
        #rpn_net.base.set_dict(src_weights, use_structured_name=False)
        #rpn_net.base.set_dict(pretrained, use_structured_name=True)
        src_keylist = list(src_weights.keys())
        
        dst_keylist_base = list(rpn_net.base.state_dict().keys())
        
        dst_keylist_depthnet = list(rpn_net.depthnet.state_dict().keys())
        for key in dst_keylist_base:
            if key not in src_keylist:
                #print(key)
                src_weights[key] = rpn_net.base.state_dict()[key]
        rpn_net.base.set_dict(src_weights, use_structured_name=True)
        
        for key in dst_keylist_depthnet:
            if key not in src_keylist:
                #print(key)
                src_weights_depthnet[key] = rpn_net.depthnet.state_dict()[key]
        rpn_net.depthnet.set_dict(src_weights_depthnet, use_structured_name=True)


    if train: rpn_net.train()
    else: rpn_net.eval()

    return rpn_net

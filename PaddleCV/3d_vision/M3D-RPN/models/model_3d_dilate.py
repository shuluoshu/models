"""
model_3d_dilate
"""

from lib.rpn_util import *
from models.backbone.densenet import densenet121
#from models.backbone.resnet import resnet101


import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.container import Sequential



class ConvLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 padding=0,
                 stride=1,
                 groups=None,
                 act=None,
                 param_attr=None,
                 bias_attr=None):
        super(ConvLayer, self).__init__()
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

def dilate_layer(layer, val):
    """dilate layer"""
    layer.dilation = val
    layer.padding = val


class RPN(fluid.dygraph.Layer):
    def __init__(self, phase, base, conf):
        super(RPN, self).__init__()
        self.base = base
        
        del self.base.transition3.pool

        # dilate
        dilate_layer(self.base.denseblock4.denselayer1.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer2.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer3.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer4.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer5.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer6.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer7.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer8.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer9.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer10.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer11.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer12.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer13.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer14.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer15.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer16.conv2, 2)

        
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]
        self.prop_feats = ConvLayer(num_channels=self.base.num_features, 
                                    num_filters=512,
                                    filter_size=3,
                                    padding=1,
                                    act='relu',
                                    param_attr=ParamAttr(name='rpn_prop_feats_weights'),
                                    bias_attr=ParamAttr(name='rpn_prop_feats_bias'))
        self.cls = ConvLayer(num_channels=self.prop_feats.num_filters, 
                             num_filters=self.num_classes*self.num_anchors,
                             filter_size=1,
                             param_attr=ParamAttr(name='rpn_cls_weights'),
                             bias_attr=ParamAttr(name='rpn_cls_bias'))

        self.bbox_x = ConvLayer(num_channels=self.prop_feats.num_filters, 
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_x_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_x_bias'))
        
        self.bbox_y = ConvLayer(num_channels=self.prop_feats.num_filters, 
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_y_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_y_bias'))

        self.bbox_w = ConvLayer(num_channels=self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_w_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_w_bias'))

        self.bbox_h = ConvLayer(num_channels=self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_h_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_h_bias'))

        self.bbox_x3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  param_attr=ParamAttr(name='rpn_bbox_x3d_weights'),
                                  bias_attr=ParamAttr(name='rpn_bbox_x3d_bias'))

        self.bbox_y3d = ConvLayer(num_channels=self.prop_feats.num_filters, # TODO
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  param_attr=ParamAttr(name='rpn_bbox_y3d_weights'),
                                  bias_attr=ParamAttr(name='rpn_bbox_y3d_bias'))

        self.bbox_z3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  param_attr=ParamAttr(name='rpn_bbox_z3d_weights'),
                                  bias_attr=ParamAttr(name='rpn_bbox_z3d_bias'))

        self.bbox_w3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  param_attr=ParamAttr(name='rpn_bbox_w3d_weights'),
                                  bias_attr=ParamAttr(name='rpn_bbox_w3d_bias'))

        self.bbox_h3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  param_attr=ParamAttr(name='rpn_bbox_h3d_weights'),
                                  bias_attr=ParamAttr(name='rpn_bbox_h3d_bias'))

        self.bbox_l3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                  num_filters=self.num_anchors,
                                  filter_size=1,
                                  param_attr=ParamAttr(name='rpn_bbox_l3d_weights'),
                                  bias_attr=ParamAttr(name='rpn_bbox_l3d_bias'))

        self.bbox_rY3d = ConvLayer(num_channels=self.prop_feats.num_filters,
                                   num_filters=self.num_anchors,
                                   filter_size=1,
                                   param_attr=ParamAttr(name='rpn_bbox_rY3d_weights'),
                                   bias_attr=ParamAttr(name='rpn_bbox_rY3d_bias'))
        
        self.feat_stride = conf.feat_stride

        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride)
        self.anchors = conf.anchors

    def forward(self, inputs):
        # backbone 
        x = self.base(inputs)
        
        prop_feats = self.prop_feats(x)

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

        batch_size, c, feat_h, feat_w = cls.shape

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

        feat_size = fluid.layers.shape(cls)[2:4]
    

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.phase == "train":
            return cls, prob, bbox_2d, bbox_3d, feat_size

        else:
            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                self.feat_size = fluid.layers.shape(cls)[2:4]
                self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride)
        
        return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois    
    


def build(conf, backbone, phase='train'):

    train = phase.lower() == 'train'

    if backbone.lower() == "densenet121":
        model_backbone = densenet121() # pretrain TODO

    # TODO
    # if backbone.lower() == "resnet101":
    #     model_backbone = resnet101(pretrained=True)

    rpn_net = RPN(phase, model_backbone.features, conf)

    if train: rpn_net.train()
    else: rpn_net.eval()
    
    return rpn_net
   


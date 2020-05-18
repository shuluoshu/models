"""
model_3d_dilate_depth_aware
"""

from lib.rpn_util import *
from models.backbone.densenet import densenet121 
#from models.backbone.resnet import ResNet101
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
            num_channels=num_channels,#输入图像的通道数
            num_filters=num_filters,#输出特征图个数
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

class LocalConv2d(fluid.dygraph.Layer):
    """LocalConv2d"""
    def __init__(self, 
                num_rows, 
                num_feats_in, 
                num_feats_out, 
                kernel=1, 
                padding=0,
                param_attr=None,
                bias_attr=None):
        super(LocalConv2d, self).__init__()
        self.num_rows = num_rows
        self.out_channels = num_feats_out
        self.kernel = kernel
        self.pad = padding
        self.group_conv = Conv2D(num_feats_in * num_rows, num_feats_out * num_rows, kernel, stride=1, groups=num_rows)

    def forward(self, x):       
        b, c, h, w = x.shape
        if self.pad:
            x = fluid.layers.pad2d(x, paddings=[self.pad, self.pad, self.pad, self.pad],
                               mode='constant', pad_value=0.0) 
        t = int(h / self.num_rows)
    
        # unfold by rows # (dimension, size, step) 2, t+padding*2, t
        tmp_list = []
        for i in range(0, self.num_rows):
            tmp_list.append(fluid.layers.slice(x, axes=[2], starts=[i*t], ends=[i*t + (t + self.pad *2)]))
        x = fluid.layers.stack(tmp_list, axis=4) 
        x = fluid.layers.transpose(x, [0,2,1,4,3])
        #b, h/row, c , row, w
        x = fluid.layers.reshape(x, [b, c * self.num_rows, t + self.pad * 2, (w+ self.pad *2)])

        # group convolution for efficient parallel processing
        y = self.group_conv(x)
        y = fluid.layers.reshape(y, [b, self.num_rows, self.out_channels, t, w])
        y = fluid.layers.transpose(y, [0,2,1,3,4])
        y = fluid.layers.reshape(y, [b, self.out_channels, h, w])
        return y


class RPN(fluid.dygraph.Layer):
    """RPN module"""
    def __init__(self, phase, base, conf):
        super(RPN, self).__init__()
        self.base = base
        
        del self.base.transition3.pool
        # settings
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]
        self.num_rows = int(min(conf.bins, calc_output_size(conf.test_scale, conf.feat_stride)))
        self.phase = phase

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

        self.prop_feats = ConvLayer(num_channels = self.base.num_features, 
                                    num_filters=512,
                                    filter_size=3,
                                    padding=1,
                                    act='relu',
                                    param_attr=ParamAttr(name='rpn_prop_feats_weights'),
                                    bias_attr=ParamAttr(name='rpn_prop_feats_bias'))

        # outputs
        self.cls = ConvLayer(num_channels = self.prop_feats.num_filters,
                             num_filters=self.num_classes * self.num_anchors,
                             filter_size=1,
                             param_attr=ParamAttr(name='rpn_cls_weights'),
                             bias_attr=ParamAttr(name='rpn_cls_bias'))
        
        # bbox 2d
        self.bbox_x = ConvLayer(num_channels = self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_x_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_x_bias'))

        self.bbox_y = ConvLayer(num_channels = self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_y_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_y_bias'))

        self.bbox_w = ConvLayer(num_channels = self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_w_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_w_bias'))

        self.bbox_h = ConvLayer(num_channels = self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_h_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_h_bias'))
        
        # bbox 3d
        self.bbox_x3d = ConvLayer(num_channels = self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_x3d_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_x3d_bias'))

        self.bbox_y3d = ConvLayer(num_channels = self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_y3d_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_y3d_bias'))

        self.bbox_z3d = ConvLayer(num_channels = self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_z3d_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_z3d_bias'))

        self.bbox_w3d = ConvLayer(num_channels = self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_w3d_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_w3d_bias'))

        self.bbox_h3d = ConvLayer(num_channels = self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_h3d_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_h3d_bias'))

        self.bbox_l3d = ConvLayer(num_channels = self.prop_feats.num_filters,
                                num_filters=self.num_anchors,
                                filter_size=1,
                                param_attr=ParamAttr(name='rpn_bbox_l3d_weights'),
                                bias_attr=ParamAttr(name='rpn_bbox_l3d_bias'))

        self.bbox_rY3d = ConvLayer(num_channels = self.prop_feats.num_filters,
                                    num_filters=self.num_anchors,
                                    filter_size=1,
                                    param_attr=ParamAttr(name='rpn_bbox_rY3d_weights'),
                                    bias_attr=ParamAttr(name='rpn_bbox_rY3d_bias'))

        self.prop_feats_loc = LocalConv2d(self.num_rows, self.base.num_features, 512, 3, padding=1,
                                     param_attr=ParamAttr(name='rpn_prop_feats_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_prop_feats_bias_loc'))
        # outputs
        self.cls_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_classes * self.num_anchors, 1, 
                                     param_attr=ParamAttr(name='rpn_cls_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_cls_loc_bias_loc'))

        # bbox 2d
        self.bbox_x_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_anchors, 1, 
                                     param_attr=ParamAttr(name='rpn_bbox_x_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_bbox_x_loc_bias_loc'))
        self.bbox_y_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_anchors, 1, 
                                     param_attr=ParamAttr(name='rpn_bbox_y_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_bbox_y_loc_bias_loc'))
        self.bbox_w_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_anchors, 1, 
                                     param_attr=ParamAttr(name='rpn_bbox_w_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_bbox_w_loc_bias_loc'))
        self.bbox_h_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_anchors, 1,
                                     param_attr=ParamAttr(name='rpn_bbox_h_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_bbox_h_loc_bias_loc'))

        # bbox 3d
        self.bbox_x3d_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_anchors, 1,
                                     param_attr=ParamAttr(name='rpn_bbox_x3d_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_bbox_x3d_loc_bias_loc'))
        self.bbox_y3d_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_anchors, 1,
                                     param_attr=ParamAttr(name='rpn_bbox_y3d_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_bbox_y3d_loc_bias_loc'))
        self.bbox_z3d_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_anchors, 1,
                                     param_attr=ParamAttr(name='rpn_bbox_z3d_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_bbox_z3d_loc_bias_loc'))
        self.bbox_w3d_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_anchors, 1,
                                     param_attr=ParamAttr(name='rpn_bbox_w3d_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_bbox_w3d_loc_bias_loc'))
        self.bbox_h3d_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_anchors, 1,
                                     param_attr=ParamAttr(name='rpn_bbox_h3d_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_bbox_h3d_loc_bias_loc'))
        self.bbox_l3d_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_anchors, 1,
                                     param_attr=ParamAttr(name='rpn_bbox_l3d_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_bbox_l3d_loc_bias_loc'))
        self.bbox_rY3d_loc = LocalConv2d(self.num_rows, self.prop_feats.num_filters, self.num_anchors, 1,
                                     param_attr=ParamAttr(name='rpn_bbox_rY3d_loc_weights_loc'),
                                     bias_attr=ParamAttr(name='rpn_bbox_rY3d_loc_bias_loc'))


        self.feat_stride = conf.feat_stride 
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride)
        self.anchors = conf.anchors 
        self.bbox_means = conf.bbox_means 
        self.bbox_stds = conf.bbox_stds

        self.cls_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5)) # TODO check
        # bbox 2d
        self.bbox_x_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5))
        self.bbox_y_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5))
        self.bbox_w_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5))
        self.bbox_h_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5))

        # bbox 3d
        self.bbox_x3d_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5))
        self.bbox_y3d_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5))
        self.bbox_z3d_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5))
        self.bbox_w3d_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5))
        self.bbox_h3d_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5))
        self.bbox_l3d_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5))
        self.bbox_rY3d_ble = self.create_parameter(shape=[1], dtype='float32', default_initializer=fluid.initializer.Constant(value=10e-5))
    

    def forward(self, inputs):
        # backbone 
        x = self.base(inputs)
        prop_feats = self.prop_feats(x)
        prop_feats_loc = self.prop_feats_loc(x)
        prop_feats_loc = fluid.layers.relu(prop_feats_loc)

        cls = self.cls(prop_feats)

        #bbox 2d
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

        cls_loc = self.cls_loc(prop_feats_loc)

        # bbox 2d
        bbox_x_loc = self.bbox_x_loc(prop_feats_loc)
        bbox_y_loc = self.bbox_y_loc(prop_feats_loc)
        bbox_w_loc = self.bbox_w_loc(prop_feats_loc)
        bbox_h_loc = self.bbox_h_loc(prop_feats_loc)

        
        # bbox 3d
        bbox_x3d_loc = self.bbox_x3d_loc(prop_feats_loc)
        bbox_y3d_loc = self.bbox_y3d_loc(prop_feats_loc)
        bbox_z3d_loc = self.bbox_z3d_loc(prop_feats_loc)
        bbox_w3d_loc = self.bbox_w3d_loc(prop_feats_loc)
        bbox_h3d_loc = self.bbox_h3d_loc(prop_feats_loc)
        bbox_l3d_loc = self.bbox_l3d_loc(prop_feats_loc)
        bbox_rY3d_loc = self.bbox_rY3d_loc(prop_feats_loc)

        cls_ble = fluid.layers.sigmoid(self.cls_ble)

        # bbox 2d
        bbox_x_ble = fluid.layers.sigmoid(self.bbox_x_ble)
        bbox_y_ble = fluid.layers.sigmoid(self.bbox_y_ble)
        bbox_w_ble = fluid.layers.sigmoid(self.bbox_w_ble)
        bbox_h_ble = fluid.layers.sigmoid(self.bbox_h_ble)

        # bbox 3d
        bbox_x3d_ble = fluid.layers.sigmoid(self.bbox_x3d_ble)
        bbox_y3d_ble = fluid.layers.sigmoid(self.bbox_y3d_ble)
        bbox_z3d_ble = fluid.layers.sigmoid(self.bbox_z3d_ble)
        bbox_w3d_ble = fluid.layers.sigmoid(self.bbox_w3d_ble)
        bbox_h3d_ble = fluid.layers.sigmoid(self.bbox_h3d_ble)
        bbox_l3d_ble = fluid.layers.sigmoid(self.bbox_l3d_ble)
        bbox_rY3d_ble = fluid.layers.sigmoid(self.bbox_rY3d_ble)
    
        # blend
        cls = (cls * cls_ble) + (cls_loc * (1 - cls_ble))
        bbox_x = (bbox_x * bbox_x_ble) + (bbox_x_loc * (1 - bbox_x_ble))
        bbox_y = (bbox_y * bbox_y_ble) + (bbox_y_loc * (1 - bbox_y_ble))
        bbox_w = (bbox_w * bbox_w_ble) + (bbox_w_loc * (1 - bbox_w_ble))
        bbox_h = (bbox_h * bbox_h_ble) + (bbox_h_loc * (1 - bbox_h_ble))
        
        bbox_x3d = (bbox_x3d * bbox_x3d_ble) + (bbox_x3d_loc * (1 - bbox_x3d_ble))
        bbox_y3d = (bbox_y3d * bbox_y3d_ble) + (bbox_y3d_loc * (1 - bbox_y3d_ble))
        bbox_z3d = (bbox_z3d * bbox_z3d_ble) + (bbox_z3d_loc * (1 - bbox_z3d_ble))
        bbox_h3d = (bbox_h3d * bbox_h3d_ble) + (bbox_h3d_loc * (1 - bbox_h3d_ble))
        bbox_w3d = (bbox_w3d * bbox_w3d_ble) + (bbox_w3d_loc * (1 - bbox_w3d_ble))
        bbox_l3d = (bbox_l3d * bbox_l3d_ble) + (bbox_l3d_loc * (1 - bbox_l3d_ble))
        bbox_rY3d = (bbox_rY3d * bbox_rY3d_ble) + (bbox_rY3d_loc * (1 - bbox_rY3d_ble))

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
        #feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.phase == "train":
            return cls, prob, bbox_2d, bbox_3d, feat_size

        else:
            feat_stride = conf.feat_stride
            anchors = conf.anchors
            feat_size = calc_output_size(np.array(conf.crop_size), feat_stride)
            rois = locate_anchors(anchors, feat_size, feat_stride) # numpy TODO
            
            if feat_size[0] != feat_h or feat_size[1] != feat_w:
                feat_size = [feat_h, feat_w]
                rois = locate_anchors(anchors, feat_size, feat_stride)
            return cls, prob, bbox_2d, bbox_3d, feat_size, rois    
       

def build(conf, backbone, phase='train'):
    # Backbone
    if backbone.lower() == "densenet121":
        backbone_res = densenet121()
    # if backbone.lower() == "resnet101":
    #     backbone_res = ResNet101().net(input) # TODO

    train = phase.lower() == 'train'
    
    num_cls = len(conf['lbls']) + 1
    num_anchors = conf['anchors'].shape[0]

    # RPN
    rpn_net = RPN(phase, backbone_res.features, conf) 

    #TODO
    '''
    if 'pretrained' in conf and conf.pretrained is not None:
        src_weights = torch.load(conf.pretrained)
        src_keylist = list(src_weights.keys())

        conv_layers = ['cls', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'bbox_x3d', 'bbox_y3d', 'bbox_w3d', 'bbox_h3d', 'bbox_l3d', 'bbox_z3d', 'bbox_rY3d']

        # prop_feats
        src_weight_key = 'module.{}.0.weight'.format('prop_feats')
        src_bias_key = 'module.{}.0.bias'.format('prop_feats')

        dst_weight_key = 'module.{}.0.group_conv.weight'.format('prop_feats_loc')
        dst_bias_key = 'module.{}.0.group_conv.bias'.format('prop_feats_loc')

        src_weights[dst_weight_key] = src_weights[src_weight_key].repeat(conf.bins, 1, 1, 1)
        src_weights[dst_bias_key] = src_weights[src_bias_key].repeat(conf.bins, )

        #del src_weights[src_weight_key]
        #del src_weights[src_bias_key]

        for layer in conv_layers:
            src_weight_key = 'module.{}.weight'.format(layer)
            src_bias_key = 'module.{}.bias'.format(layer)

            dst_weight_key = 'module.{}.group_conv.weight'.format(layer+'_loc')
            dst_bias_key = 'module.{}.group_conv.bias'.format(layer+'_loc')

            src_weights[dst_weight_key] = src_weights[src_weight_key].repeat(conf.bins, 1, 1, 1)
            src_weights[dst_bias_key] = src_weights[src_bias_key].repeat(conf.bins, )

            #del src_weights[src_weight_key]
            #del src_weights[src_bias_key]

        src_keylist = list(src_weights.keys())

        # rename layers without module..
        for key in src_keylist:

            key_new = key.replace('module.', '')
            if key_new in dst_keylist:
                src_weights[key_new] = src_weights[key]
                del src_weights[key]

        src_keylist = list(src_weights.keys())
    '''



    if train: rpn_net.train()
    else: rpn_net.eval()
    
    return rpn_net





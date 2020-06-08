"""
model_3d_dilate_depth_aware
"""

from lib.rpn_util import *
from models.backbone.densenet import DenseNet121 
from models.backbone.resnet import ResNet101
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


def LocalConv2d(x,
                num_rows,
                num_feats_out,
                kernel=1,
                padding=0,
                param_attr=None,
                bias_attr=None):
    b, c, h, w = x.shape
    if padding:
        x = fluid.layers.pad2d(x, paddings=[padding, padding, padding, padding],
                               mode='constant', pad_value=0.0) 
    t = int(h / num_rows)
    
    # unfold by rows # (dimension, size, step) 2, t+padding*2, t
    tmp_list = []
    for i in range(0,num_rows):
        tmp_list.append(fluid.layers.slice(x, axes=[2], starts=[i*t], ends=[i*t + (t+padding*2)]))
    x = fluid.layers.stack(tmp_list, axis=4) 
    x = fluid.layers.transpose(x, [0,2,1,4,3])
    #b, h/row, c , row, w
    x = fluid.layers.reshape(x, [b, c * num_rows, t + padding*2, (w+padding*2)])

    # group convolution for efficient parallel processing
    y = fluid.layers.conv2d(
        input=x,
        num_filters=num_feats_out*num_rows,
        filter_size=kernel,
        groups=num_rows,
        stride=1,
        param_attr=param_attr,
        bias_attr=bias_attr)
    
    y = fluid.layers.reshape(y, [b, num_rows, num_feats_out, t, w])
    y = fluid.layers.transpose(y, [0,2,1,3,4])
    y = fluid.layers.reshape(y, [b, num_feats_out, h, w])

    return y


def rpn_module(input, conf, phase):
    # settings
    num_classes = len(conf['lbls']) + 1
    num_anchors = conf['anchors'].shape[0]
    num_rows = int(min(conf.bins, calc_output_size(conf.test_scale, conf.feat_stride)))

    prop_feats = fluid.layers.conv2d(
        input=input,
        num_filters=512,
        filter_size=3,
        padding=1,
        act='relu',
        param_attr=ParamAttr(name='rpn_prop_feats_weights'),
        bias_attr=ParamAttr(name='rpn_prop_feats_bias'))

        
    # outputs
    cls = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_classes*num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_cls_weights'),
        bias_attr=ParamAttr(name='rpn_cls_bias'))
        
    # bbox 2d
    bbox_x = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_bbox_x_weights'),
        bias_attr=ParamAttr(name='rpn_bbox_x_bias'))

    bbox_y = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_bbox_y_weights'),
        bias_attr=ParamAttr(name='rpn_bbox_y_bias'))

    bbox_w = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_bbox_w_weights'),
        bias_attr=ParamAttr(name='rpn_bbox_w_bias'))

    bbox_h = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_bbox_h_weights'),
        bias_attr=ParamAttr(name='rpn_bbox_h_bias'))
    
    # bbox 3d
    bbox_x3d = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_bbox_x3d_weights'),
        bias_attr=ParamAttr(name='rpn_bbox_x3d_bias'))

    bbox_y3d = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_bbox_y3d_weights'),
        bias_attr=ParamAttr(name='rpn_bbox_y3d_bias'))

    bbox_z3d = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_bbox_z3d_weights'),
        bias_attr=ParamAttr(name='rpn_bbox_z3d_bias'))

    bbox_w3d = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_bbox_w3d_weights'),
        bias_attr=ParamAttr(name='rpn_bbox_w3d_bias'))

    bbox_h3d = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_bbox_h3d_weights'),
        bias_attr=ParamAttr(name='rpn_bbox_h3d_bias'))

    bbox_l3d = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_bbox_l3d_weights'),
        bias_attr=ParamAttr(name='rpn_bbox_l3d_bias'))

    bbox_rY3d = fluid.layers.conv2d(
        input=prop_feats,
        num_filters=num_anchors,
        filter_size=1,
        param_attr=ParamAttr(name='rpn_bbox_rY3d_weights'),
        bias_attr=ParamAttr(name='rpn_bbox_rY3d_bias'))

    # loc
    prop_feats_loc = LocalConv2d(input, num_rows, 512, 3, 1,
                                 param_attr=ParamAttr(name='rpn_prop_feats_weights_loc'),
                                 bias_attr=ParamAttr(name='rpn_prop_feats_bias_loc'))
    prop_feats_loc = fluid.layers.relu(prop_feats_loc)

    # outputs
    cls_loc = LocalConv2d(prop_feats_loc, num_rows, num_classes*num_anchors, 1,
                          param_attr=ParamAttr(name='rpn_cls_weights_loc'),
                          bias_attr=ParamAttr(name='rpn_cls_bias_loc'))
        
    # bbox 2d
    bbox_x_loc = LocalConv2d(prop_feats_loc, num_rows, num_anchors, 1,
                             param_attr=ParamAttr(name='rpn_bbox_x_weights_loc'),
                             bias_attr=ParamAttr(name='rpn_bbox_x_bias_loc'))
    bbox_y_loc = LocalConv2d(prop_feats_loc, num_rows, num_anchors, 1,
                             param_attr=ParamAttr(name='rpn_bbox_y_weights_loc'),
                             bias_attr=ParamAttr(name='rpn_bbox_y_bias_loc'))
    bbox_w_loc = LocalConv2d(prop_feats_loc, num_rows, num_anchors, 1,
                             param_attr=ParamAttr(name='rpn_bbox_w_weights_loc'),
                             bias_attr=ParamAttr(name='rpn_bbox_w_bias_loc'))
    bbox_h_loc = LocalConv2d(prop_feats_loc, num_rows, num_anchors, 1,
                             param_attr=ParamAttr(name='rpn_bbox_h_weights_loc'),
                             bias_attr=ParamAttr(name='rpn_bbox_h_bias_loc'))
    
    # bbox 3d
    bbox_x3d_loc = LocalConv2d(prop_feats_loc, num_rows, num_anchors, 1,
                               param_attr=ParamAttr(name='rpn_bbox_x3d_weights_loc'),
                               bias_attr=ParamAttr(name='rpn_bbox_x3d_bias_loc'))
    bbox_y3d_loc = LocalConv2d(prop_feats_loc, num_rows, num_anchors, 1,
                               param_attr=ParamAttr(name='rpn_bbox_y3d_weights_loc'),
                               bias_attr=ParamAttr(name='rpn_bbox_y3d_bias_loc'))
    bbox_z3d_loc = LocalConv2d(prop_feats_loc, num_rows, num_anchors, 1,
                               param_attr=ParamAttr(name='rpn_bbox_z3d_weights_loc'),
                               bias_attr=ParamAttr(name='rpn_bbox_z3d_bias_loc'))
    bbox_w3d_loc = LocalConv2d(prop_feats_loc, num_rows, num_anchors, 1,
                               param_attr=ParamAttr(name='rpn_bbox_w3d_weights_loc'),
                               bias_attr=ParamAttr(name='rpn_bbox_w3d_bias_loc'))
    bbox_h3d_loc = LocalConv2d(prop_feats_loc, num_rows, num_anchors, 1,
                               param_attr=ParamAttr(name='rpn_bbox_h3d_weights_loc'),
                               bias_attr=ParamAttr(name='rpn_bbox_h3d_bias_loc'))
    bbox_l3d_loc = LocalConv2d(prop_feats_loc, num_rows, num_anchors, 1,
                               param_attr=ParamAttr(name='rpn_bbox_l3d_weights_loc'),
                               bias_attr=ParamAttr(name='rpn_bbox_l3d_bias_loc'))
    bbox_rY3d_loc = LocalConv2d(prop_feats_loc, num_rows, num_anchors, 1,
                                param_attr=ParamAttr(name='rpn_bbox_rY3d_weights_loc'),
                               bias_attr=ParamAttr(name='rpn_bbox_rY3d_bias_loc'))

    cls_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='cls_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5))) # TODO check
    bbox_x_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='bbox_x_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5)))
    bbox_y_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='bbox_y_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5)))
    bbox_w_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='bbox_w_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5)))
    bbox_h_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='bbox_h_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5)))

    bbox_x3d_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='bbox_x3d_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5)))
    bbox_y3d_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='bbox_y3d_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5)))
    bbox_z3d_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='bbox_z3d_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5)))
    bbox_w3d_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='bbox_w3d_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5)))
    bbox_h3d_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='bbox_h3d_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5)))
    bbox_l3d_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='bbox_l3d_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5)))
    bbox_rY3d_ble = fluid.layers.sigmoid(fluid.layers.create_parameter(shape=[1], dtype='float32', name='bbox_rY3d_ble_params', default_initializer=fluid.initializer.Constant(value=10e-5)))
    
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
    cls = fluid.layers.reshape(x=cls, shape=[batch_size, num_classes, feat_h * num_anchors, feat_w])
    # score probabilities
    prob = fluid.layers.softmax(cls, axis=1)
    
    # reshape for consistency
    bbox_x = flatten_tensor(fluid.layers.reshape(x=bbox_x, shape=[batch_size, 1, feat_h * num_anchors, feat_w]))
    bbox_y = flatten_tensor(fluid.layers.reshape(x=bbox_y, shape=[batch_size, 1, feat_h * num_anchors, feat_w]))
    bbox_w = flatten_tensor(fluid.layers.reshape(x=bbox_w, shape=[batch_size, 1, feat_h * num_anchors, feat_w]))
    bbox_h = flatten_tensor(fluid.layers.reshape(x=bbox_h, shape=[batch_size, 1, feat_h * num_anchors, feat_w]))

    bbox_x3d = flatten_tensor(fluid.layers.reshape(x=bbox_x3d, shape=[batch_size, 1, feat_h * num_anchors, feat_w]))
    bbox_y3d = flatten_tensor(fluid.layers.reshape(x=bbox_y3d, shape=[batch_size, 1, feat_h * num_anchors, feat_w]))
    bbox_z3d = flatten_tensor(fluid.layers.reshape(x=bbox_z3d, shape=[batch_size, 1, feat_h * num_anchors, feat_w]))
    bbox_w3d = flatten_tensor(fluid.layers.reshape(x=bbox_w3d, shape=[batch_size, 1, feat_h * num_anchors, feat_w]))
    bbox_h3d = flatten_tensor(fluid.layers.reshape(x=bbox_h3d, shape=[batch_size, 1, feat_h * num_anchors, feat_w]))
    bbox_l3d = flatten_tensor(fluid.layers.reshape(x=bbox_l3d, shape=[batch_size, 1, feat_h * num_anchors, feat_w]))
    bbox_rY3d = flatten_tensor(fluid.layers.reshape(x=bbox_rY3d, shape=[batch_size, 1, feat_h * num_anchors, feat_w]))
        
    # bundle
    bbox_2d = fluid.layers.concat(input=[bbox_x, bbox_y, bbox_w, bbox_h], axis=2)
    bbox_3d = fluid.layers.concat(input=[bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d], axis=2)

    feat_size = [feat_h, feat_w]

    cls = flatten_tensor(cls)
    prob = flatten_tensor(prob)

    if phase == "train":
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
       
def m3d_rpn(input, conf, backbone, phase='train'):
    # Backbone
    if backbone.lower() == "densenet121":
        backbone_res = DenseNet121().net(input)
    # if backbone.lower() == "resnet101":
    #     backbone_res = ResNet101().net(input) # TODO
    
    # RPN
    return rpn_module(backbone_res, conf, phase) 





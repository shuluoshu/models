"""
model_3d_dilate
"""

from lib.rpn_util import *
from models.backbone.densenet import DenseNet121 
from models.backbone.resnet import ResNet101 
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

def rpn_module(input, conf, phase):
    # settings
    num_classes = len(conf['lbls']) + 1
    num_anchors = conf['anchors'].shape[0]
    
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
    return rpn_module(backbone_res, conf, phase) # phase TODO

"""
resnet_dilate
"""

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal
from paddle.fluid.layers import utils
from paddle.fluid import core, dygraph_utils


class DeformConv2DV2(fluid.dygraph.Layer):
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
                 dtype='float32'):
        assert param_attr is not False, "param_attr should not be False here."
        super(DeformConv2DV2, self).__init__()
        self._num_channels = num_channels
        self._num_filters = num_filters
        self._filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
        self._stride = utils.convert_to_list(stride, 2, 'stride')
        self._padding = utils.convert_to_list(padding, 2, 'padding')
        self._dilation = utils.convert_to_list(dilation, 2, 'dilation')
        self._groups = groups
        self._deformable_groups = deformable_groups
        self._im2col_step = im2col_step
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._dtype = dtype

        if self._groups is None:
            num_filter_channels = self._num_channels
        else:
            if self._num_channels % self._groups != 0:
                raise ValueError("num_channels must be divisible by groups.")
            num_filter_channels = self._num_channels // self._groups

        filter_shape = [self._num_filters, num_filter_channels
                        ] + self._filter_size

        def _get_default_param_initializer():
            filter_elem_num = self._filter_size[0] * self._filter_size[
                1] * self._num_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std, 0)

        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            default_initializer=_get_default_param_initializer())

        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input, offset, mask):
        attrs = (
            'strides',
            self._stride,
            'paddings',
            self._padding,
            'dilations',
            self._dilation,
            'groups',
            self._groups or 1,
            'deformable_groups',
            self._deformable_groups or 1,
            'im2col_step',
            self._im2col_step if self._im2col_step else 64, )
        assert mask is not None, "mask should be given if modulated is True"
        out = core.ops.deformable_conv(input, offset, mask, self.weight,
                                       *attrs)

        return dygraph_utils._append_bias_in_dygraph(out, self.bias, 1)

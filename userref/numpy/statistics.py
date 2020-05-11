# -*- coding: utf-8 -*-
"""
Created on 2020/5/1 17:43 by PyCharm

@author: xumiz
"""
# %% 模块导入
import numpy as np

# %% 次序统计
# numpy.amax(a, axis=None, out=None, keepdims=None, initial=None, where=None)
#     返回数组的最大值或沿轴的最大值(传播nan)
#     maximum(a[0], a[1])快于amax(a, axis=0)
#     参数：
#         a: array_like
#         axis: None or int or tuple of ints
#             默认平铺数组
#         out: ndarray, None
#             存储结果的位置
#         keepdims: bool
#             设置为True，结果维度保留大小为1，且将对输入数组进行正确的广播
#         initial: scalar
#             输出元素的最小值
#         where: array_like of bool
#             元素进行比较以获得最大值
#     返回： ndarray or scalar
# numpy.amin(a, axis=None, out=None, keepdims=None, initial=None, where=None)
#     返回数组的最大值或沿轴的最小值(传播nan)
# numpy.nanmax(a, axis=None, out=None, keepdims=None)
#     返回数组的最大值或沿轴的最大值(忽略nan)
# numpy.nanmin(a, axis=None, out=None, keepdims=None)
#     返回数组的最大值或沿轴的最小值(忽略nan)
np.amax([[0, 1], [2, 3]])
np.amax([[0, 1], [2, 3]], axis=0)
np.amax([[0, 1], [2, 3]], axis=1)
np.amax([[0, 1], [2, 3]], where=[False, True], initial=-1, axis=0)

np.amax([0, 1, np.nan, 3, 4])
np.amax([0, 1, np.nan, 3, 4], where=~np.isnan([0, 1, np.nan, 3, 4]),
        initial=-1)
np.nanmax([0, 1, np.nan, 3, 4])

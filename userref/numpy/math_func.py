# -*- coding: utf-8 -*-
"""
Created on 2020/5/1 17:25 by PyCharm

@author: xumiz
"""
# %% 模块导入
import numpy as np

# %% 算术运算
# numpy.add(x1, x2, out=None, where=True, casting='same_kind', order='K',
#           dtype=None, subok=True)
#     求和，等价于 x1 + x2 (广播操作)
#     参数：
#         x1, x2: array_like
#             如果x1，x2形状不一致，,它们必须能够被广播成一个公共的形状
#     返回：ndarray or scalar
# numpy.divide(x1, x2, out=None, where=True, casting='same_kind', order='K',
#              dtype=None, subok=True)
#     标准除法
# numpy.floor_divide(x1, x2, out=None, where=True, casting='same_kind',
#                    order='K', dtype=None, subok=True)
#     求商，等价于 x1 // x2
# numpy.remainder(x1, x2, out=None, where=True, casting='same_kind',
#                 order='K', dtype=None, subok=True)
#     求余，等价于 x1 % x2 (广播操作,与除数x2有相同的符号)
# numpy.divmod(x1, x2, where=True, casting='same_kind', order='K', dtype=None,
#              subok=True)
#     求商余，等价于(x // y, x % y)
# numpy.modf(x, where=True, casting='same_kind', order='K', dtype=None,
#            subok=True)
#     按元素顺序返回数组的小数部分和整数部，如果数字是负数，则分数和整数部分都是负数
x1 = np.arange(1, 4)
x2 = np.arange(-4, 5).reshape((3, 3))
print(x1, x2, sep='\n')

np.add(x1, x2)

np.divide(np.add(x1, x2), [1, 2, -2])
np.floor_divide(np.add(x1, x2), [1, 2, -2])
np.remainder(np.add(x1, x2), [1, 2, -2])
np.divmod(np.add(x1, x2), [1, 2, -2])
np.modf([0, 3.5, -3.5])

# %% 取整
# numpy.floor(x, out=None, where=True, casting='same_kind', order='K',
#             dtype=None, subok=True)
#     向下取整
#     参数：
#         x1, x2: array_like
#             如果x1，x2形状不一致，,它们必须能够被广播成一个公共的形状
#     返回：ndarray or scalar
# numpy.ceil(x, out=None, where=True, casting='same_kind', order='K',
#            dtype=None, subok=True)
#     向上取整
# numpy.trunc(x, out=None, where=True, casting='same_kind', order='K',
#             dtype=None, subok=True)
# numpy.fix(x, out=None)
#     向0取整
# numpy.rint(x, out=None, where=True, casting='same_kind', order='K',
#            dtype=None, subok=True)
#     四舍五入
np.floor([-1.5, -1.4, -0.2, 0.2, 1.4, 1.5, 2.0])
np.ceil([-1.5, -1.4, -0.2, 0.2, 1.4, 1.5, 2.0])
np.trunc([-1.5, -1.4, -0.2, 0.2, 1.4, 1.5, 2.0])
np.rint([-1.5, -1.4, -0.2, 0.2, 1.4, 1.5, 2.0])
np.fix([-1.5, -1.4, -0.2, 0.2, 1.4, 1.5, 2.0])

# numpy.around(a, decimals=0, out=None)
#     四舍五入到给定的小数数目,对于中间值，NumPy舍入为最接近的偶数
np.around([0.37, 1.64])
np.around([0.37, 1.64], decimals=1)

# %% 其他
# numpy.maximum(x1, x2, out=None, where=True, casting='same_kind', order='K',
#               dtype=None, subok=True)
#     比较两个数组并返回一个新数组，其元素取两个数组中的较大者（传播nan值）
#     等价于 np.where(x1 >= x2, x1, x2)，但此方法更快速并可适当的广播
#     参数：
#         x1, x2: array_like
#         out: ndarray, None, or tuple of ndarray and None,
#             存储结果的位置
#     返回： ndarray or scalar
# numpy.minimum(x1, x2, out=None, where=True, casting='same_kind', order='K',
#               dtype=None, subok=True)
#     比较两个数组并返回一个新数组，其元素取两个数组中的较小者（传播nan值）
# fmax(x1, x2, out=None, where=True, casting='same_kind', order='K',
#      dtype=None, subok=True)
#     比较两个数组并返回一个新数组，其元素取两个数组中的较大者（忽略nan值）
# fmin(x1, x2, out=None, where=True, casting='same_kind', order='K',
#      dtype=None, subok=True)
#     比较两个数组并返回一个新数组，其元素取两个数组中的较小者（忽略nan值）
np.maximum([2, 3, 4], [1, 5, 2])
np.maximum([[1, 0], [0, 1]], [0.5, 2])
np.maximum(-np.Inf, 1)
np.maximum([np.nan, 0, np.nan], [0, np.nan, np.nan])
np.fmax([np.nan, 0, np.nan], [0, np.nan, np.nan])

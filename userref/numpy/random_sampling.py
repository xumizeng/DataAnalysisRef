# -*- coding: utf-8 -*-
"""
Created on 2020/4/17 14:27 by PyCharm

@author: xumiz
"""
# %% 模块导入
import numpy as np

# %% 随机数生成器(numpy.random.Generator)构造函数
# numpy.random.default_rng()
#     用默认的位生成器(PCG64)构造一个新的生成器
#     返回
#         out: <class 'numpy.random.Generator'>
rng = np.random.default_rng()
print(rng)

# %% 生成器方法
# standard_normal(size=None, dtype='d', out=None)
#     从标准正态分布中抽取样本 (mean=0, stdev=1)
#     参数：
#         size: int or tuple of ints
#             输出的形状
#         dtype: {str, dtype}
#             数据类型，例如：‘d’ (‘float64’) 或 ‘f’ (‘float32’)
#         out： ndarray
#             用于放置结果的可选输出数组
#     返回
#         out: float 或 ndarra
# 样本来自N(均值mean=0, 标准差stdev=1)
rng.standard_normal()
rng.standard_normal(size=(3, 4))

# 样本来自 N(均值mean=3, 标准差stdev=2.5)
3 + 2.5 * rng.standard_normal(size=(3, 4))

# normal(loc=0.0, scale=1.0, size=None)
#     从正态(高斯)分布中随机抽取样本
#     参数：
#         loc: float or array_like of floats
#             正态分布的平均值
#         scale: float or array_like of floats
#             正态分布的标准差
#         size: int or tuple of ints
#             输出的形状
#     返回
#         out: scalar 或 ndarray
# 样本来自 N(均值mean=3, 标准差stdev=2.5)
rng.normal(3, 2.5, size=(3, 4))

# integers(low, high=None, size=None, dtype='int64', endpoint=False)
#     返回指定范围内的随机整数
#     参数：
#         low: int or array-like of ints
#             从分布中抽取的最小整数
#         high: int or array-like of ints
#             从分布中抽取的最大整数
#         size: int or tuple of ints
#             输出的形状
#         dtype: {str, dtype}
#             数据类型
#         endpoint: bool
#             为真，则从 [low, high] 采样，而非默认的 [low, high)
#     返回: scalar 或 ndarray
rng.integers(1, size=10)
rng.integers(2, size=10)
rng.integers(5, size=(2, 4))

rng.integers(1, [3, 5, 10])
rng.integers([1, 5, 7], 10)

rng.integers([1, 3, 5, 7], [[10], [20]], dtype=np.uint8)

# shuffle(x, axis=0)
#     通过打乱序列的内容来修改序列
#     参数：
#         x: array_like
#         axis: int
#             只支持ndarray对象
#     返回: None
arr = np.arange(10)
rng.shuffle(arr)
print(arr)

arr = np.arange(9).reshape((3, 3))
rng.shuffle(arr)
print(arr)

arr = np.arange(9).reshape((3, 3))
rng.shuffle(arr, axis=1)
print(arr)

# permutation(x, axis=0)
#     随机排列一个序列
#     参数：
#         x: int or array_like
#             ndarray: 复制并随机打乱数组元素
#             int: 随机排列np.arange(x)
#     返回: ndarray
rng.permutation(10)
arr = np.arange(9).reshape((3, 3))
rng.permutation(arr)
rng.permutation(arr, axis=1)

# choice(a, size=None, replace=True, p=None, axis=0)
#     参数：
#         a: 1-D array-like or int
#             ndarray: 从数组中生成一个随机样本
#             int: 从 np.arange(a) 中生成一个随机样本
#         size: int or tuple of ints
#             输出的形状
#         replace: bool
#             默认采样时进行替换而非打乱采样顺序
#         p: 1-D array-like
#             默认a中所有元素的概率均匀分布
#         shuffle: bool
#     返回: scalar 或 ndarray
rng.choice(5, 3)  # 等价于 rng.integers(0,5,3)
rng.choice(5, 3, replace=False)  # 等价于  rng.permutation(np.arange(5))[:3]
rng.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])

arr = ['a', 'b', 'c', 'd']
rng.choice(arr, 5, p=[0.5, 0.1, 0.1, 0.3])

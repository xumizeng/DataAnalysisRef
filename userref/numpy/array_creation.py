# -*- coding: utf-8 -*-
"""
Created on 2020/4/8 13:19 by PyCharm

@author: xumiz
"""
# %% 模块导入
import numpy as np
import numpy.ma as ma

# %% numpy.array 构造器
# numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)
#     创建一个所含项目类型相同，大小相同的数组
#     参数:
#         object: array_like
#             数组
#         dtype: data-type
#             数据类型，没有给出，类型将被确定为保存对象所需的最小类型
#         copy: bool
#             默认复制输入数组
#         subok: bool
#             默认返回的数组强制为基类数组，否则将传递子类
#         ndmin: int
#             指定结果数组应有的最小维数
#     返回:
#         out: <class 'numpy.ndarray'>  ndarray数组
# 创建一个多维数组
np.array([[1, 2], [3.0, 4]])

# 指定数据类型
np.array([1, 2, 3], dtype=np.float)

# copy=Fasle 时对数组操作可能改变数据源
d = np.arange(4)
arr = np.array(d, copy=False)
arr[1] = 0
print(arr)
print(d)

# 从子类创建一个数组
x = ma.array([1, 2, 3], mask=[0, 0, 1])
np.array(x, subok=True)
np.array(x)

# 指定最小尺寸
np.array([1, 2, 3], ndmin=2)

# %% numpy.arange 构造器
# numpy.arange([start, ]stop, [step, ]dtype=None)
#     返回包含给定间隔内的均匀间隔值的数组，可用于整数步长
#     参数:
#         dtype: data-type
#             数据类型，没有给出，类型将被确定为保存对象所需的最小类型
#     返回:
#         out: <class 'numpy.ndarray'>  ndarray数组
np.arange(3, 7, 2)

# %% numpy.linspace 构造器
# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False,
#                dtype=None, axis=0)
#     返回包含给定间隔内的均匀间隔值的数组，可用于非整数步长
#     参数:
#         num: int
#             生成的样本数量
#         endpoint: bool
#             默认包含最后一个样本
#         retstep: bool
#             默认不返回步长
#         dtype: data-type
#             数据类型，没有给出，类型将被确定为保存对象所需的最小类型
#     返回:
#         out: <class 'numpy.ndarray'>  ndarray数组 或 含步长的元组
np.linspace(3, 7, num=3)
np.linspace(2.0, 3.0, num=5)
np.linspace(2.0, 3.0, num=5, endpoint=False)
np.linspace(2.0, 3.0, num=5, retstep=True)

# %% ndarray 属性
# ndarray.shape       数组形状(out: <class 'tuple'>)
# ndarray.ndim        数组维数(out: <class 'int'>)
# ndarray.size        数组中元素数目(out: <class 'int'>)
arr = np.array([[1, 2], [3, 4]])
print(arr.shape, arr.ndim, arr.size)

# ndarray.base        获取数组的原始对象
#                     (out: <class 'NoneType'> 或 <class 'numpy.ndarray'>)
x = np.array([1, 2, 3, 4])
y = x[2:]
print(x.base, y.base is x)

# ndarray.dtype       数据类型(out: <type 'numpy.dtype'>)
arr = np.array([1, 2, 3], dtype=int)
print(arr.dtype.type is np.int32)

# class numpy.flatiter
#     通过 ndarray.flat 返回的数组一维化迭代器，允许使用基本切片或高级索引
#     返回
#         out: <class 'numpy.flatiter'>)
x = np.arange(1, 7).reshape(2, 3)
for i in x.flat:
    print(i)

# %% ndarray 方法
# copy(order='C')
#     返回数组的一个副本
#     除了order参数有不同的默认值，等同于numpy.copy(a, order='K')
#     返回:
#         out: <class 'numpy.ndarray'>  ndarray数组
arr = np.array([0, 1, 2])
x = arr
y = arr.copy()
print(arr is x, y is arr)
x[1] = 6
print(arr is x)
x = 5
print(arr is x)

# .reshape(shape, order='C')
#     返回一个新形状数组
#     除了允许形状作为单独参数传入，等同于numpy.reshape(a, newshape, order='C')
#     即 a.reshape(10, 11) 等同于 a.reshape((10, 11))
#     参数:
#         shape: int or tuple of ints
#             形状维度可以是-1，表示该维度从数组的长度和其余维度推断
#     返回:
#         out: <class 'numpy.ndarray'>  ndarray数组
np.arange(1, 7).reshape(2, 3)
np.arange(1, 7).reshape(2, 3).reshape(6)
np.arange(1, 7).reshape(3, -1)

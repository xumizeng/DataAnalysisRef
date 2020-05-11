# -*- coding: utf-8 -*-
"""
Created on 2020/4/21 12:05 by PyCharm

@author: xumiz
"""
# %% 模块导入
import pandas as pd
import numpy as np

# %% GroupBy 构造器
# DataFrame 方法：
# groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
#         group_keys=True, squeeze=False, observed=False)
#     对DataFrame进行分组
#     参数：
#         by: mapping, function, label, or list of labels
#             用于确定groupby的组
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             沿着行或列拆分
#         level: int, level name, or sequence of such
#             如果轴是一个多层索引，则按一个或多个特定级别分组
#         as_index: bool
#             对于聚合的输出，默认返回以组标签为索引的对象
#         sort: bool
#             按group_keys排序，关闭此选项性能更好，Groupby保留每个组中的行顺序
#         group_keys: bool
#             默认将组键添加到索引中以标识块
#         squeeze: bool
#             如果可能，降低返回类型的维数
#     返回： DataFrameGroupBy
# 基本索引类型
rng = np.random.default_rng()
df = pd.DataFrame({'A': ['a', 'b', 'a', 'b',
                         'a', 'b', 'a', 'a'],
                   'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C': rng.integers(5, size=8),
                   'D': rng.integers(5, size=8)})
print(df)
df.groupby(['A']).sum()
df.groupby(['A', 'B']).sum()

# 多层索引类型
arr = [['a', 'a', 'c', 'c'], ['d', 'b', 'd', 'b']]
index = pd.MultiIndex.from_arrays(arr, names=('idx1', 'idx2'))
df = pd.DataFrame({'col': [30, 35, 30, 20]}, index=index)
print(df)
df.groupby(level=0).sum()
df.groupby(level='idx2').sum()

col = pd.MultiIndex.from_arrays([['A0', 'A1', 'A1', 'A0'],
                                 ['B0', 'B1', 'B0', 'B1']])
idx = pd.MultiIndex.from_arrays([['C0', 'C0', 'C1', 'C1'],
                                 ['D0', 'D1', 'D0', 'D1']])
df = pd.DataFrame(rng.integers(9, size=[4, 4]), index=idx, columns=col)

df.groupby(level=1, axis=1).sum()
df.stack()
df.stack().groupby(level=1).sum()

# 分类数据
cat = pd.Categorical(['a', 'a', 'b', 'b'], categories=['b', 'a', 'c'])
df = pd.DataFrame({'A': cat, 'B': ['c', 'd', 'c', 'd'], 'C': [1, 2, 3, 4]})
print(df)
df.groupby('A').sum()
df.groupby(['A', 'B']).sum()

#  Grouper对象
# class pandas.Grouper
# Grouper(key=None, level=None, freq=None, axis=0, sort=False)
#     允许用户为对象指定groupby指令
#     参数：
#         key: str
#             选择目标的分组列
#         level: name/number
#             如果轴是一个多层索引，则按一个或多个特定级别分组
#         freq: str / frequency object
#             如果目标选择(通过key或level)是一个类似于datetime的对象，
#             那么它将根据指定的频率分组
#         sort: bool
#             按group_keys排序，关闭此选项性能更好，Groupby保留每个组中的行顺序
#     返回： DataFrameGroupBy
df = pd.DataFrame({'A': ['A0'] * 5 + ['B0'] * 4,
                   'B': ['B0'] + ['B0', 'B0', 'B1', 'B1'] * 2,
                   'C': ['C0', 'C1', 'C1'] + ['C0', 'C0', 'C1'] * 2,
                   'D': rng.integers(9, size=9),
                   'E': rng.integers(9, size=9),
                   'F': [pd.Timestamp(2020, i, 1) for i in range(1, 6)] + [
                       pd.Timestamp(2020, i, 15) for i in range(1, 5)]})
print(df)
df.groupby(['A']).sum()
df.groupby(pd.Grouper(key='A')).sum()
df.groupby(pd.Grouper(key='F', freq='M')).sum()

# %% GroupBy方法：计算
# GroupBy.mean(self, *args, **kwargs)
#     计算每组的平均值，排除丢失的值
#     返回： Series or DataFrame
# GroupBy.sum(self, **kwargs)
#     计算每组值的和，排除丢失的值
df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
                   'B': [np.nan, 2, 3, 4, 5],
                   'C': [1, 2, 1, 1, 2]})
print(df)
df.groupby('A').mean()
df.groupby(['A', 'B']).mean()
df.groupby('A')['B'].mean()

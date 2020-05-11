# -*- coding: utf-8 -*-
'''
Created on 2020/4/11 10:26 by PyCharm

@author: xumiz
'''
# %% 模块导入
import pandas as pd
import numpy as np

# %% pandas.Index 构造器
# class pandas.Index
# Index(data=None, dtype=None, copy=False, name=None, tupleize_cols=True,
#       **kwargs)
#     创建一个不可变索引
#     参数：
#         data: array-like, 1-dimensional
#             数据源，索引实例只能包含可 hashable 的对象
#         dtype: np.dtype：
#             输出数据类型为'int64', 'uint64' 或'object'，默认自行推断
#         name: object
#             存储在索引中的名称
#         tupleize_cols: bool
#             默认尝试在可能的情况下创建一个多层索引
#     返回：
#         out: <'pandas.Index'> 索引
# 构造基本索引
a = pd.Index(['c', 'b', 'a'])
b = pd.Index(['c', 'e', 'd'], name='A')
print(a, b, sep='\n')

# %% Index方法：修改
# set_names(self, names, level=None, inplace=False)
#     设置索引或多层索引名称
#         返回：
#             out: <'pandas.Index'>
# rename(self, name, inplace=False)
#     更改索引或多索引名称
# unique(self, level=None)
#     返回索引中唯一的值
# nunique(self, dropna=True)
#     返回对象中唯一元素的数目
tuples = [(1, 'one'), (1, 'one'),
          (3, 'one'), (2, 'two'),
          (3, 'one'), (2, 'two')]
idx = pd.Index(tuples, names=['A', 'B'])
print(idx)
idx.set_names(['lvl1', 'lvl2'])
idx.set_names('lvl3', level=1)
idx.rename(['lvl4', 'lvl5'])

idx.unique()
idx.unique(level=0)
idx.unique(level=1)

idx.nunique(dropna=False)

# %% Index方法：时间序列操作
# shift(self, periods=1, freq=None)
#     按所需的时频增量数移动索引,仅对类似于datetime的索引类实现，即DatetimeIndex,
#     dindex 以及 TimedeltaIndex
#     参数：
#         periods: int
#             要移动的周期数
#         freq: DateOffset, tseries.offsets, timedelta, or str
#             频率增量移位，未指定则索引通过其自身的freq属性进行移位
#     返回:
#         out: 索引类型
idx = pd.date_range('1/1/2020', periods=5, freq='MS')
idx.shift(10)
idx.shift(10, freq='D')

# %% Index方法： 集合操作
# intersection(self, other, sort=False)
#     返回两个索引的交集,结果索引默认不做排序
# union(self, other, sort=None)
#     返回两个索引的并集，结果索引默认将按升序排序
# difference(self, other, sort=None)
#     返回一个新索引，该索引中的元素不在other中，结果索引默认将按升序排序
# symmetric_difference(self, other, result_name=None, sort=None)
#     返回两个索引的对称差集，结果索引默认将按升序排序
a = pd.Index(['c', 'b', 'a', 'f'])
b = pd.Index(['c', 'e', 'd', 'b'], name='A')
print(a, b, sep='\n')

# 等同于 a & b
print(a.intersection(b))
# 等同于 a | b
print(a.union(b))
# 等同于 a ^ b
print(a.symmetric_difference(b))

a.difference(b)
b.difference(a)

# %% 数字索引：RangeIndex
# class pandas.RangeIndex 构造器
# RangeIndex(start=None, stop=None, step=None, dtype=None, copy=False,
#            name=None)
#     创建一个单调整数范围的不可变索引
#     参数：
#         name: object
#             存储在索引中的名称
#     返回：
#         out: <'pandas.RangeIndex'>索引
pd.RangeIndex(5)
pd.RangeIndex(5, name='alpha')

# %% 分类索引：CategoricalIndex
# class pandas.CategoricalIndex 构造器
# CategoricalIndex(data=None, categories=None, ordered=None, dtype=None,
#                  copy=False, name=None)
#     创建一个基于 Categorical 的索引,不可进行数字运算
#     参数：
#         data: array-like (1-dimensional)
#             分类数据
#         categories: Index-like (unique)
#             类别，其顺序会定义分类数据的输出顺序
#             默认由数据源剔除重复后的唯一值组成，不在类别中的值输出时将被NaN替换
#         ordered: bool
#             默认不对生成的categorical排序
#         dtype: CategoricalDtype
#             用于此分类的CategoricalDtype实例，不能与类别或顺序一起使用
#         name: object
#             存储在索引中的名称
#     返回：
#         out: <'pandas.CategoricalIndex'> 分类索引
pd.CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'])
pd.CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'], ordered=True,
                    categories=['c', 'a'])

# 从一个Categorical实例化
i = pd.CategoricalIndex(pd.Categorical([1, 2, 3, 4], categories=[4, 2, 3, 1]))
print(i)
df = pd.DataFrame({'A': ['a', 'b', 'c', 'd'], 'B': [4, 2, 3, 1]}, index=i)
print(df)
df.sort_index()

# 指定分类数据的数据类型
t = pd.CategoricalDtype(categories=['b', 'a'], ordered=True)
pd.CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'], dtype=t)

# %% 多重索引：MultiIndex
# MultiIndex类方法
# classmethod MultiIndex.from_arrays(arrays, sortorder=None, names=object)
#     将数组转换为多层索引
#     参数：
#         arrays: list / sequence of array-likes
#             数据源
#         names : list / sequence of str, optional
#             索引中每层的名称
#     返回：
#         out: <'pandas.MultiIndex'> 分类索引
index = pd.MultiIndex.from_arrays([[1, 1, 2, 2], ['a', 'b', 'a', 'b']],
                                  names=('A', 'B'))
pd.DataFrame(np.arange(8).reshape(4, 2), index=index)

# classmethod MultiIndex.from_tuples(tuples, sortorder=None, names=None)
#     将元组列表转换为多层索引
#     参数：
#         tuples: list / sequence of tuple-likes
#             数据源
#         names: list / sequence of str, optional
#             索引中每层的名称
#     返回：
#         out: <'pandas.MultiIndex'> 分类索引
tuples = [(1, 'red'), (1, 'blue'), (2, 'red'), (2, 'blue')]
index = pd.MultiIndex.from_tuples(tuples, names=('number', 'color'))
pd.DataFrame(np.arange(8).reshape(4, 2), index=index)

# classmethod MultiIndex.from_product(iterables, sortorder=None, names=object)
#     将笛卡尔乘积转换为多层索引
#     参数：
#         iterables: list / sequence of iterables
#             数据源
#         names: list / sequence of str, optional
#             索引中每层的名称
#     返回：
#         out: <'pandas.MultiIndex'> 分类索引
idx = pd.MultiIndex.from_product([['A0', 'A1', 'A2'],
                                  ['B0', 'B1'],
                                  ['C0', 'C1']])
col = pd.MultiIndex.from_product([['a', 'b'], ['x', 'y']])
rng = np.random.default_rng()
df = pd.DataFrame(rng.integers(0, 9, size=(len(idx), len(col))),
                  index=idx, columns=col)
print(df)

# slice(None) 选择该leve的所有内容
print(df.loc[(slice('A1', 'A2'), slice(None), ['C0']), :])
print(df.loc[:, (slice(None), ['y'])])

idx = pd.IndexSlice
print(df.loc[idx['A1':'A2', :, ['C0', 'C1']], idx[:, 'y']])
print(df.loc[idx[df[('a', 'y')] > 5, :, ['C0']], idx[:, 'y']])

print(df.loc[(slice(None), slice(None), ['C0']), :])
df.loc[(slice(None), slice(None), ['C0']), :] = 0
print(df)

# classmethod MultiIndex.from_frame(df, sortorder=None, names=None)
#     从DataFrame创建一个多层索引
#     参数：
#         df: DataFram
#             数据源
#         names: list / sequence of str, optional
#             如果没有提供名称，则使用列名，如果列是多层索引，则使用列名的元组
#             如果是一个序列，用给定的序列覆盖名称
#     返回：
#         out: <'pandas.MultiIndex'> 分类索引
df = pd.DataFrame(np.arange(8).reshape(4, 2))
index = pd.MultiIndex.from_frame(df, names=['A', 'B'])
pd.DataFrame(np.arange(8).reshape(4, 2), index=index)

# get_level_values(self, level)
#     返回指定级别的标签值
idx = pd.MultiIndex.from_arrays((list('abc'), list('def')), names=['A', 'B'])
idx.get_level_values(0)
idx.get_level_values('B')

# set_levels(self, levels, level=None, inplace=False, verify_integrity=True)
#   在MultiIndex上设置新级别
tuples = [(1, 'one'), (1, 'one'),
          (3, 'one'), (2, 'two'),
          (2, 'one'), (3, 'two')]
idx = pd.MultiIndex.from_tuples(tuples, names=['A', 'B'])
print(idx)
idx.set_levels([['a', 'b', 'c'], [1, 2]])
idx.set_levels(['a', 'b', 'c'], level=0)
idx.set_levels(['a', 'b'], level='B')

# %% 日期索引：DatetimeIndex
# class pandas.DatetimeIndex
# DatetimeIndex(data=None, freq=None, tz=None, normalize=False, closed=None,
#               ambiguous='raise', dayfirst=False, yearfirst=False, dtype=None,
#               copy=False, name=None)
#     创建一个日期索引
#     参数：
#         data: array-like (1-dimensional)
#             用于构造索引的datetime数据
#         freq: str or pandas offset object
#             频率
#         name: object
#             存储在索引中的名称
#         dayfirst: bool
#             True: 解析数据中的日期以天为第一顺序
#         yearfirst: bool
#             True: 解析数据中的日期以年为第一顺序
#     返回：
#         out: <class 'pandas.DatetimeIndex'>
pd.DatetimeIndex(['2020-02-12', '2020-02-15'], name='date')
pd.DatetimeIndex(['10-02-12', '10-02-15'])
pd.DatetimeIndex(['10-02-12', '10-02-15'], dayfirst=True)
pd.DatetimeIndex(['10-02-12', '10-02-15'], yearfirst=True)

# DatetimeIndex 方法：
# to_pydatetime(self, *args, **kwargs)
#     将 DatetimeIndex 转换为 datetime.datetime
#     返回：ndarray
idx = pd.DatetimeIndex(['10-02-12', '10-02-15'])
idx.to_pydatetime()

# PeriodIndex 方法：
# to_timestamp(self, freq=None, how='start', copy=True)
#     Series中的 PeriodIndex 转换为 DatetimeIndex(如果没有传递，则从索引中推断)
#     参数：
#         freq: str or DateOffset
#             默认值为“D”
#         how: {‘s’, ‘e’, ‘start’, ‘end’}
#             默认为周期的开始
#     返回:
#         out: <'pandas.Series'>
idx = pd.period_range('1/1/2020', periods=3, freq='M')
print(idx)
idx.to_timestamp()
idx.to_timestamp(how='end')

# %% 时间增量索引：TimedeltaIndex
# class pandas.TimedeltaIndex
# TimedeltaIndex(data=None, unit=None, freq=None, closed=None,
#                dtype=dtype('<m8[ns]'), copy=False, name=None)
#     创建一个周期索引
#     参数：
#         data: array-like (1-dimensional)
#             用于构造索引的timedelta数据
#         unit: str
#             可能值：
#             ‘Y’, ‘M’, ‘W’, ‘D’, ‘T’, ‘S’, ‘L’, ‘U’, or ‘N’
#             ‘days’ or ‘day’
#             ‘hours’, ‘hour’, ‘hr’, or ‘h’
#             ‘minutes’, ‘minute’, ‘min’, or ‘m’
#             ‘seconds’, ‘second’, or ‘sec’
#             ‘milliseconds’, ‘millisecond’, ‘millis’, or ‘milli’
#             ‘microseconds’, ‘microsecond’, ‘micros’, or ‘micro’
#             ‘nanoseconds’, ‘nanosecond’, ‘nanos’, ‘nano’, or ‘ns’
#         freq: str or pandas offset object
#             频率
#         name: object
#             存储在索引中的名称
#     返回：
#         out: <class 'pandas.TimedeltaIndex'>
pd.TimedeltaIndex(['1 days', '2 days'], freq='D')
pd.TimedeltaIndex([1, 2], unit='D', freq='D')

# %% 周期索引：PeriodIndex
# class pandas.PeriodIndex
# PeriodIndex(data=None, ordinal=None, freq=None, tz=None, dtype=None,
#             copy=False, name=None, **fields)
#     创建一个周期索引
#     参数：
#         data: array-like (1d int np.ndarray or PeriodArray)
#             用于构造索引的周期数据
#         freq: str or period object
#             频率
#         name: object
#             存储在索引中的名称
#         **fields: {year, month, quarter, day, hour, minute, second}
#                   int, array, or Series
#     返回：
#         out: <class 'pandas.PeriodIndex'>
pd.PeriodIndex(['2020-01', '2020-02', '2020-03'], freq='M')

# PeriodIndex 方法：
# asfreq(self, freq, how=None)
#     将周期索引按指定的频率转换
#     参数：
#         freq: DateOffset or str
#             频率
#         how: {‘START’/‘S’/‘BEGIN’, ‘E’/‘END’/‘FINISH’}
#             仅适用于 PeriodIndex
#     返回:
#         out: <'pandas.DataFrame'>
idx = pd.period_range('2020-01-01', '2025-01-01', freq='A')
print(idx)
idx.asfreq('M')
idx.asfreq('M', how='s')

# DatetimeIndex 方法：
# to_period(self, freq=None, copy=True)
#     将 DatetimeIndex 转换为具有所需频率的 PeriodIndex,(如果没有传递，
#     则从索引中推断)
idx = pd.DatetimeIndex(['2020-03-31', '2020-05-31', '2020-08-31'])
print(idx)
idx.to_period('M')

idx = pd.date_range('1/1/2020', periods=5, freq='MS')
print(idx)
idx.to_period(freq='M')
idx.to_period()

# %% 区间索引：IntervalIndex
# IntervalIndex 类方法
# classmethod IntervalIndex.from_arrays(left, right, closed='right', name=None,
#                                       copy=False, dtype=None)
#     由定义左边界和右边界的两个数组构造一个区间索引
#     参数：
#         left: array-like (1-dimensional)
#             左界
#         right: array-like (1-dimensional)
#             右界
#         closed : {‘right’, ‘left’, ‘both’, ‘neither’}
#             控制区间的封闭
#     返回：
#         out: <'pandas.IntervalIndex'> 区间索引
pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3])

# classmethod IntervalIndex.from_breaks(breaks, closed='right', name=None,
#                                       copy=False, dtype=None)
#     从一个数组分割出区间索引
#     参数：
#         breaks: array-like (1-dimensional)
#             左界
#         closed : {‘right’, ‘left’, ‘both’, ‘neither’}
#             控制区间的封闭
#     返回：
#         out: <'pandas.IntervalIndex'> 区间索引
pd.IntervalIndex.from_breaks([0, 1, 2, 3])

# classmethod IntervalIndex.from_tuples(data, closed='right', name=None,
#                                       copy=False, dtype=None)
#     从类元组构造一个区间索引
#     参数：
#         data: array-like (1-dimensional)
#             左界
#         closed : {‘right’, ‘left’, ‘both’, ‘neither’}
#             控制区间的封闭,默认左开右闭
#     返回：
#         out: <'pandas.IntervalIndex'> 区间索引
pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])

# -*- coding: utf-8 -*-
"""
Created on 2020/4/21 12:19 by PyCharm

@author: xumiz
"""
# %% 模块导入
import numpy as np
import pandas as pd

# %% Resampler 构造器
# DataFrame 方法：
# resample(self, rule, axis=0, closed=None, label=None, convention='start',
#          kind=None, loffset=None, base=0, on=None, level=None)
#     重新取样时间序列数据
#     参数：
#         rule: DateOffset, Timedelta or str
#             表示目标转换的偏移字符串或对象
#         closed: {‘right’, ‘left’}
#             控制区间闭合
#             频率偏移默认 ‘left’ ，除了 ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’,  ‘W’
#         label: {‘right’, ‘left’}
#             控制区间闭合
#             默认 ‘left’ ，除了 ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’,  ‘W’
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             进行向上或向下采样的目标轴
#         convention: ‘start’, ‘end’, ‘s’, ‘e’}
#             控制使用 rule 的开始或结束，仅对 PeriodIndex 生效
#         kind: {‘timestamp’, ‘period’}
#             传递 'timestamp' 将结果索引转换为 DateTimeIndex
#             传递 'period' 将结果索引转换为 PeriodIndex
#             默认情况下，保留输入表示
#         loffset: timedelta
#             调整重新采样的时间标签
#         on: str
#             使用列代替索引进行重新采样
#         level: str or int
#            对于多索引，用于重新采样的级别
#     返回: DatetimeIndexResampler
df = pd.DataFrame({'col1': [1, 1, 9, 1, 4, 8, 7, 9],
                   'col2': [5, 6, 4, 1, 5, 1, 4, 5]})
df['col3'] = pd.date_range('01/01/2020', periods=8, freq='W')
print(df)
df.resample('M', on='col3').sum()

d = dict(col1=[1, 1, 9, 3, 4, 8, 7, 9], col2=[5, 6, 4, 1, 5, 1, 4, 5])
dtidx = pd.date_range('1/1/2020', periods=4, freq='D')
idx = pd.MultiIndex.from_product([dtidx, ['A', 'B']])
df = pd.DataFrame(data=d, index=idx, copy=True)
print(df)
df.resample('D', level=0).sum()

index = pd.period_range('2020-01-01', freq='A', periods=2)
df = pd.DataFrame([[1, 2], [3, 4]], index=index)
print(df)
df.resample('Q', convention='start').asfreq()
df.resample('Q', convention='end').asfreq()

# Series 方法：
# resample(self, rule, axis=0, closed=None, label=None, convention='start',
#          kind=None, loffset=None, base=0, on=None, level=None)
#     重新取样时间序列数据
#     参数：
#         rule: DateOffset, Timedelta or str
#             表示目标转换的偏移字符串或对象
#         closed: {‘right’, ‘left’}
#             控制区间闭合
#             频率偏移默认 ‘left’ ，除了 ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’,  ‘W’
#         label: {‘right’, ‘left’}
#             输出索引中采用的区间边界
#             边界默认 ‘left’ ，除了 ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’,  ‘W’
#         convention: ‘start’, ‘end’, ‘s’, ‘e’}
#             控制使用 rule 的开始或结束，仅对 PeriodIndex 生效
#         kind: {‘timestamp’, ‘period’}
#             传递 'timestamp' 将结果索引转换为 DateTimeIndex
#             传递 'period' 将结果索引转换为 PeriodIndex
#             默认情况下，保留输入表示
#         loffset: timedelta
#             调整重新采样的时间标签
#         level: str or int
#            对于多索引，用于重新采样的级别
#     返回:
#         out: DatetimeIndexResampler
idx = pd.date_range('1/1/2020', periods=5, freq='T')
ser = pd.Series(range(5), index=idx)
print(ser)
ser.resample('3T', label='left', closed='left').sum()
ser.resample('3T', label='left', closed='right').sum()
ser.resample('3T', label='right', closed='right').sum()

index = pd.period_range('2020-01-01', freq='A', periods=2)
ser = pd.Series([1, 2], index=index)
print(ser)
ser.resample('Q', convention='start').asfreq()
ser.resample('Q', convention='end').asfreq()

# %% Resampler方法：函数应用
# apply(self, func, *args, **kwargs)
#     沿着指定轴应用一个函数,执行任何类型的操作
#     参数：
#         func: function
#             函数(传递给函数的对象是 Series)
#         *args
#             传递给func的位置参数
#         **kwds
#             传递给func的关键字参数
#     返回: scalar, Series or DataFrame
# aggregate(self, func, *args, **kwargs)
#     沿着指定轴应用一个或多个函数进行聚合，只执行聚合类型操作,通常使用聚合的别名agg
# transform(self, arg, *args, **kwargs)
#     沿着指定轴应用一个或多个函数进行转换,只执行转换类型操作
df = pd.DataFrame({'col1': [1, 1, 9, 1, 4, 8, 7, 9],
                   'col2': [5, 6, 4, 1, 5, 1, 4, 5]})
df['col3'] = pd.date_range('01/01/2020', periods=8, freq='W')
print(df)
df.resample('M', on='col3').apply(np.sum)
df.resample('M', on='col3').apply(['sum', 'mean', 'max'])

df.resample('M', on='col3').agg(np.sum)
df.resample('M', on='col3').agg(['sum', 'mean', 'max'])

df.resample('M', on='col3').transform(np.sum)
df.resample('M', on='col3').transform(lambda x: (x - 1))

# %% Resampler方法：计算
# mean(self, _method='mean', *args, **kwargs)
#     计算每组的平均值，排除丢失的值
#     返回： Series or DataFrame
d = dict(col1=[1, 1, 9, 1, np.nan, 8, 7, 9],
         col2=[5, 6, 4, 1, 5, 1, 4, 5],
         col3=[5, 6, 4, 1, 5, 1, 4, 5])
idx = pd.date_range('01/01/2020', periods=8, freq='W')
df = pd.DataFrame(data=d, index=idx, copy=True)
print(df)
df.resample('M').mean()

# sum(self, _method='sum', min_count=0, *args, **kwargs)
#     计算每组值的和，排除丢失的值
d = dict(col1=[1, 1, 9, 1, np.nan, 8, 7, 9],
         col2=[5, 6, 4, 1, 5, 1, 4, 5],
         col3=[5, 6, 4, 1, 5, 1, 4, 5])
idx = pd.date_range('01/01/2020', periods=8, freq='W')
df = pd.DataFrame(data=d, index=idx, copy=True)
print(df)

df.resample('M').sum()
df.resample('M').sum(min_count=4)

# %% Resampler方法：升采样
# asfreq(self, fill_value=None)
#     将时间索引按指定的频率转换
#     参数：
#         fill_value: scalar
#             用于升采样期间应用的缺失值，不会填充已经存在的NaNs
#     返回:
#         out: <'pandas.DataFrame'> 或 <'pandas.Series'>
idx = pd.date_range('1/1/2020', periods=5, freq='T')
ser = pd.Series(range(5), index=idx)
print(ser)
ser.resample('30S').asfreq()
ser.resample('30S').asfreq(fill_value=10)

# fillna(self, method, limit=None)
#     填充由升采样引入的缺失值，原始数据中缺失的值将不会被修改
#     参数：
#         method: {‘pad’/‘ffill’, ‘backfill’/‘bfill’, ‘nearest’}
#             如下所述
#         limit: int
#             填充值的数量限制
#     返回: Series 或 DataFrame
# bfill(self, limit=None)
#     在重新采样的数据中用后值填充新的缺失值
#     参数：
#         limit: int
#             填充值的数量限制
#     返回: Series 或 DataFrame
# pad(self, limit=None)
#     在重新采样的数据中用前值填充新的缺失值
# nearest(self, limit=None)
#     在重新采样的数据中用最接近的值填充新的缺失值
idx = pd.date_range('1/1/2020', periods=3, freq='T')
ser = pd.Series(range(3), index=idx)
print(ser)
ser.resample('15s').fillna("backfill")

ser.resample('15S').bfill()
ser.resample('15S').pad()
ser.resample('15S').nearest()

ser.resample('15S').bfill(limit=2)
ser.resample('15S').pad(limit=2)
ser.resample('15S').nearest(limit=1)

df = pd.DataFrame({'a': [2, np.nan, 6], 'b': [1, 3, 5]},
                  index=pd.date_range('20200101', periods=3, freq='h'))
print(df)
df.resample('30min').backfill()
df.resample('15min').pad(limit=2)
df.resample('15min').nearest(limit=2)

# interpolate(self, method='linear', axis=0, limit=None, inplace=False,
#              limit_direction='forward', limit_area=None, downcast=None,
#              **kwargs)
#     根据不同的方法插入值，对于多索引DataFrame/Series，只支持method='linear'
#     参数：
#         method: str
#             填充方法
#             ‘linear’: 插值在缺失的数据点上执行线性插值,仅对数值类型有效
#             ‘pad’: 沿轴将有效值向前/后的缺失值填充，对数字，字符串均有效
#                    注意应用此方法，轴方向不同于默认方向
#         axis{0 or ‘index’, 1 or ‘columns’, None}
#             目标轴，默认 ‘index’
#         inplace: bool
#             默认操作对象非输入数据源，inplace=True 操作输入数据，返回None
#         limit: int
#             填充的连续NaN值的最大数目,默认尽可能填充
#         limit_direction{‘forward’, ‘backward’, ‘both’}
#             填充的连续NaN值的方向
#         limit_area{None, ‘inside’, ‘outside’}
#             填充的连续NaN值的区域
#         None: 无填充约束
#             ‘inside’: 只填充被有效值包围的nan值
#             ‘outside’: 只填充有效值之外的nan值
#         downcast: dict
#             定义转换类型
#         **kwargs
#             传递给插值函数的关键字参数
#     返回:
#         out: <'pandas.DataFrame'> 或 <class 'NoneType'>
d = [[0.0, np.nan, -1.0, 1.0],
     [np.nan, 2.0, np.nan, np.nan],
     [2.0, 3.0, np.nan, 9.0],
     [np.nan, 4.0, -4.0, 16.0]]
index = pd.date_range('20200101', periods=4, freq='h')
df = pd.DataFrame(d, columns=list('abcd'), index=index, copy=True)
print(df)

df.resample('30min').asfreq()
df.resample('30min').interpolate()
df.resample('30min').interpolate(limit_direction='backward')
df.resample('30min').interpolate(limit_direction='both')

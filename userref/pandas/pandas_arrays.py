# -*- coding: utf-8 -*-
"""
Created on 2020/4/8 9:57 by PyCharm

@author: xumiz
"""
# %% 模块导入
import pandas as pd
import numpy as np

# %% Pandas arrays 构造函数
# pandas.array(data, dtype=None, copy=True)
#     创建一个扩展数组 ExtensionArray
#     参数：
#         data: Sequence of objects, 1-dimensional
#             数据源，当数据是 Index 或 Series 时，将从数据中提取底层数组
#         dtype: str, np.dtype, or ExtensionDtype
#             强制指定数据类型，默认解析器自行获取或推断ExtensionDtype,详见附录
#         copy: bool
#             复制数据
#     返回：
#         out: 扩展数组
pd.array([1, 2, np.nan])
pd.array(['a', None, 'c'])
pd.array([pd.Period('2020', freq='D'), pd.Period('2020', freq='D')])
pd.array(['a', 'b', 'a'], dtype='category')

# panda若未推断出一个专用的扩展类型将返回PandasArray
pd.array([1.1, 2.2])

# dtype指定类型( datetime 或 timedelta 除外)为np.dtype，以确保返回PandasArray
pd.array(['a', None, 'c'], dtype=np.dtype('<U1'))
pd.array(['2015', '2016'], dtype=np.dtype('datetime64[ns]'))
pd.array(['1H', '2H'], dtype='timedelta64[ns]')

# copy=False 时可能改变数据源
np_arr = np.array([1, 2, np.nan])
pd_arr = pd.array(np_arr, copy=False)
pd_arr[2] = 3
print(pd_arr, np_arr, sep='\n')

# %% Datetime数据
# class pandas.Timestamp
# Timestamp(ts_input=object, freq=None, tz=None, unit=None, year=None,
#           month=None, day=None, hour=None, minute=None, second=None,
#           microsecond=None, nanosecond=None, tzinfo=None)
#     返回一个用于替换python datetime.datetime对象的 Timestamp 时间戳
#     参数：
#         ts_input: datetime-like, str, int, float
#             要转换为时间戳的值
#         unit: str [ ‘D’, ‘h’, ‘m’, ‘s’, ‘ms’, ‘us’, ‘ns’]
#             ts_input类型为int或float时用于转换的单元
#         year, month, day: int
#             日期
#         hour, minute, second, microsecond: int, default 0
#             时间
#     返回：
#         out: <'pandas.Timestamp'> 时间戳
# 从str, int, float转换时间戳
pd.Timestamp('2020-01-01T12')
pd.Timestamp(0, unit='d')
pd.Timestamp(0.5, unit='s')

# 通过位置或关键字传递（两者不可混在一起）
pd.Timestamp(2020, 1, 1, 12)
pd.Timestamp(year=2017, month=1, day=1, hour=12)

# %% 时间增量数据
# class pandas.Timedelta
# Timedelta(value=object, unit=None, **kwargs)
#     返回时间增量，即两个日期或时间之间的差值
#     参数：
#         value : Timedelta, timedelta, np.timedelta64, string, or integer
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
#     返回：
#         out: <class 'pandas.Timedelta'>
pd.Timedelta('12 days')
pd.Timedelta(12, unit='D')
pd.Timedelta(12, unit='W')

# %% 时间间隔数据
# class pandas.Period
# Period(value=None, freq=None, ordinal=None, year=None, month=None,
#        quarter=None, day=None, hour=None, minute=None, second=None)
#     返回一段时间
#     参数：
#         value : Period or str
#             时间段
#         freq : str
#             一个周期字符串对象,默认为1天，常用字符串见附录
#     返回：
#         out: <'pandas.Period'> 时间段
pd.Period('2018-03-11', freq='M')

# class pandas.PeriodDtype
# PeriodDtype(freq=None)
#     返回一个 Period 数据的 ExtensionDtype
#     参数：
#         freq: str or DateOffset
#             一个周期字符串对象,默认为1天，常用字符串见附录
#     返回：
#         out: <'pandas.PeriodDtype'>
pd.PeriodDtype(freq='D')
pd.PeriodDtype(freq=pd.offsets.MonthEnd())

# %% 区间数据
# class pandas.Interval
# Interval(left, right, closed=‘right’)
#     返回一个有界区间
#     参数：
#         left: orderable scalar
#             左界
#         right: orderable scalar
#             右界
#         closed : {‘right’, ‘left’, ‘both’, ‘neither’}
#             控制区间的封闭
#     返回：
#         out: <'pandas.Interval'> 有界区间
pd.Interval(1, 5.2)

# 控制区间封闭
# noinspection PyArgumentList
iv = pd.Interval(1, 5, closed='left')
print(iv, 1 in iv, 5 in iv, iv.length)

# 运算操作
pd.Interval(1, 5) + 3
pd.Interval(1, 5) * 10.0

# 创建时间区间
# noinspection PyArgumentList
pd.Interval(pd.Timestamp('2017-01-01 00:00:00'),
            pd.Timestamp('2018-01-01 00:00:00'), closed='left')

# class pandas.IntervalDtype
# IntervalDtype(subtype=None)
#     指定 Interval 数据的ExtensionDtype
#     参数：
#         subtype: str, np.dtype
#             区间界限的dtype
#     返回：
#         out: <'pandas.IntervalDtype'>
pd.IntervalDtype(subtype='int64')

# %% 分类数据
# class pandas.Categorical
# Categorical(values, categories=None, ordered=None, dtype=None,
#             fastpath=False)
#     以经典 R/S-plus 方式表示一个分类变量,不可进行数字运算
#     参数：
#         values: list-like
#             分类数据
#         categories: Index-like (unique)
#             类别，其顺序会定义分类数据的输出顺序
#             默认由数据源剔除重复后的唯一值组成，不在类别中的值输出时将被NaN替换
#         ordered: bool
#             默认不对生成的categorical排序
#         dtype: CategoricalDtype
#             用于此分类的CategoricalDtype实例，不能与类别或顺序一起使用
#     返回：
#         out: <'pandas.Categorical'> 分类对象
pd.Categorical([1, 2, 3, 1, 2, 3])
pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])

# 有序的分类可以根据自定义的分类顺序进行排序，并且可以有最值
pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'], ordered=True,
               categories=['c', 'b', 'a'])
pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'], ordered=True,
               categories=['b', 'a'])

raw_cat = pd.Categorical(['a', 'b', 'c', 'a'], categories=['b', 'c', 'd'],
                         ordered=False)
print(raw_cat)
pd.Series(raw_cat)
pd.DataFrame({'A': ['a', 'b', 'c', 'a'], 'B': raw_cat})

# 指定分类数据的数据类型
cat_type = pd.CategoricalDtype(categories=['b', 'a'], ordered=True)
pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'], dtype=cat_type)

# Categorical 方法
# unique(self)
#     返回序列对象的唯一值
#     返回：ExtensionArray
pd.Categorical(list('baabc')).unique()
pd.Categorical(list('baabc'), categories=list('abc'), ordered=True).unique()

# factorize(self, na_sentinel=-1)
#     将对象编码为枚举类型或分类变量
#     参数：
#         sort: bool, default True
#             对uniques进行排序，并对code进行洗牌来保持顺序
#         na_sentinel: int
#             缺失值标记
#     返回：(codes: ndarray, uniques: Index)
codes, uniques = pd.Categorical(['b', None, 'a', 'c', 'b']).factorize()
print(codes, uniques, sep='\n')

# classmethod Categorical.from_codes(codes, categories=None, ordered=None,
#                                    dtype=None)
#     从codes和categories中创建一个 Categorical
rng = np.random.default_rng()
splitter = rng.choice([0, 1], size=5, p=[0.5, 0.5])
print(splitter)
pd.Categorical.from_codes(splitter, categories=["a", "b"])

# class pandas.CategoricalDtype
# CategoricalDtype(categories=None, ordered=False)
#     指定分类数据的类型
#     参数：
#         categories: sequence
#             类别，唯一并且不能包含任何空值
#             默认由数据源剔除重复后的唯一值组成，不在类别中的值输出时将被NaN替换
#         ordered: bool
#             默认不对生成的categorical排序
#         dtype: CategoricalDtype
#             用于此分类的CategoricalDtype实例
#     返回：
#         out: <'pandas.CategoricalDtype'>
pd.CategoricalDtype()
pd.CategoricalDtype(categories=['b', 'c', 'd'], ordered=True)

# %% 附录一: Pandas 数组扩展类型
# Scalar Type                     Array Type
# pandas.Interval                 pandas.arrays.IntervalArray
# pandas.Period                   pandas.arrays.PeriodArray
# datetime.datetime               pandas.arrays.DatetimeArray
# datetime.timedelta              pandas.arrays.TimedeltaArray
# (none)                          pandas.Categorical
# (none)                          pandas.SparseArray
# int                             pandas.arrays.IntegerArray
# str                             pandas.arrays.StringArray
# bool                            pandas.arrays.BooleanArray

# %% 附录二：常用日期偏移量及相关的频率字符串列表
# Date Offset         FreStr          Description
# DateOffset          None            通用偏移量类，默认为1天
# Week                'W'             一个星期，可选固定在一周中的某一天
# WeekOfMonth         'WOM'           每个月的第一个星期的第x天
# LastWeekOfMonth     'LWOM'          每月最后一周的第x天
# MonthEnd            'M'             月末
# MonthBegin          'MS'            月初
# SemiMonthEnd        'SM'            每月15日(或其他日期)和月末
# SemiMonthBegin      'SMS'           15日(或其他日期)和月初
# QuarterEnd          'Q'             季度末
# QuarterBegin        'QS'            季度初
# YearEnd             'A'             年末
# YearBegin           'AS' or 'BYS'   年初
# Day                 'D'             一天
# Hour                'H'             一小时
# Minute              'T' or 'min'    一分钟
# Second              'S'             一秒钟
# Milli               'L' or 'ms'     一毫秒
# Micro               'U' or 'us'     一微秒
# Nano                'N'             一纳秒

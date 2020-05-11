# -*- coding: utf-8 -*-
"""
Created on 2020/4/11 16:15 by PyCharm

@author: xumiz
"""
# %% 模块导入
import pandas as pd
import numpy as np

# %% 数据操作
# cut(x, bins, right=True, labels=None, retbins=False, precision=3,
#     include_lowest=False, duplicates='raise')
#     将值转换成离散区间
#     参数：
#         x: array-like, 1-dimensional
#             数据源
#         bins: int, sequence of scalars, or IntervalIndex
#             定义区间形式：
#                 int: 等宽区间的数目（x的范围两边扩展1%，以包括最小值和最大值）
#                 sequence of scalars: 定义了被分割后每一个区间边界（此时x不扩展1%）
#                 IntervalIndex: 定义准确区间（区间索引不可重叠）
#         right : bool
#             默认区间左开右闭，否则左闭右开，对 bins=IntervalIndex 无效
#         labels : array or False
#             为返回区间指定标签，且影响输出区间的类型，对 bins=IntervalIndex 无效
#             参数值为False，函数只返回区间的整数指示符，即x中的数据在第几个区间内
#         retbins : bool
#             默认不返回分割后的区间
#         duplicates : {default 'raise', 'drop'}
#             默认不允许存在重复区间， duplicate =drop，将删除多余重复区间
#     返回：
#         out: 数组对象，表示每个值所对应的区间，类型取决于数据源 x 及 标签 labels
#         labels: False ，返回 <'numpy.ndarray'>
#         labels: array/None，x:Series 返回 <'pandas.Series'>
#             否则为<'pandas.Categorical'>
arr = np.array([1, 7, 5, 4, 6, 3])

# bin: int, sequence of scalars, or IntervalIndex
pd.cut(arr, 3)
pd.cut(arr, np.array([0, 3, 6, 9]))
bins = pd.IntervalIndex.from_tuples([(0, 3), (3, 6), (6, 9)], closed='left')
pd.cut(arr, bins)

# labels: None, array, False
pd.cut(arr, 3, right=False)
pd.cut(arr, 3, labels=["bad", "medium", "good"])
pd.cut(arr, 3, labels=False)

# x: Series
ser = pd.Series(arr, index=['a', 'b', 'c', 'd', 'e', 'f'], copy=True)
pd.cut(ser, 3)
pd.cut(ser, [0, 3, 6, 9], labels=False)
pd.cut(ser, [0, 3, 6, 9, 9], labels=False, retbins=True, duplicates='drop')

# x: DataFrame
rng = np.random.default_rng()
df = pd.DataFrame({'value': rng.integers(0, 100, size=20)})
print(df)

labels = ["{0} - {1}".format(i, i + 9) for i in range(0, 100, 10)]
print(labels)
df['group'] = pd.cut(df['value'], range(0, 105, 10), right=False)
print(df)
df['group'] = pd.cut(df['value'], range(0, 105, 10), right=False,
                     labels=labels)
print(df)

# qcut(x, q, labels=None, retbins=False, precision=3, duplicates='raise')
#     根据分位数将样本转换成离散区间
#     参数：
#         x: array-like, 1-dimensional
#             数据源
#         q: int or list-like of int
#             分位数
#         labels : array or False
#             为返回区间指定标签，参数值为False，函数只返回区间的整数指示符
#         retbins : bool
#             默认不返回分割后的区间
#         duplicates : {default 'raise', 'drop'}
#             默认不允许存在重复区间， duplicate =drop，将删除多余重复区间
#     返回：
#         out: 数组对象，表示每个值所对应的区间，类型取决于数据源 x 及 标签 labels
#         labels: False ，返回 <'numpy.ndarray'>
#         labels: array/None，x:Series 返回 <'pandas.Series'>
#             否则为<'pandas.Categorical'>
arr = np.array([1, 7, 5, 4, 6, 3])

# q: int or list-like of int
pd.qcut(arr, 4)
pd.qcut(arr, [0, .25, .5, .75, 1.])

# labels: None, array, False
pd.qcut(arr, 3)
pd.qcut(arr, 3, labels=["bad", "medium", "good"])
pd.qcut(arr, 3, labels=False)

# x: Series
ser = pd.Series(arr, index=['a', 'b', 'c', 'd', 'e', 'f'], copy=True)
pd.qcut(ser, 3)
pd.qcut(ser, 3, labels=False)
pd.qcut(ser, 3, labels=False, retbins=True, duplicates='drop')

# get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None,
#             sparse=False, drop_first=False, dtype=None)
#     将分类变量转换为虚拟指标
#     参数：
#         data: array-like, Series, or DataFrame
#             数据
#         prefix: str, list of str, or dict of str
#             列名前缀
#         prefix_sep: str
#             分隔符
#         dummy_na : bool
#             添加一个列来指示NaNs
#         columns: list-like
#             默认转换所有列
#         drop_first: bool
#             移除第一个分类
#     返回：DataFrame
df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', np.nan],
                   'C': [1, 2, 3]})
print(df)

pd.get_dummies(df)
pd.get_dummies(df, columns=['A'])
pd.get_dummies(df['A'])
pd.get_dummies(df, prefix=['col1', 'col2'])
pd.get_dummies(df, drop_first=True)
pd.get_dummies(df, dummy_na=True)

# concat(objs, axis=0, join='outer', ignore_index=False, keys=None,
#        levels=None, names=None, verify_integrity=False, sort=False,
#        copy=True)
#     沿着特定轴连接panda对象
#     参数：
#         objs: a sequence or mapping of Series or DataFrame objects
#             操作对象
#         axis : {0/'index', 1/'columns'}
#             操作目标轴
#         join : {'inner', 'outer'}
#             处理其他轴(或多个轴)上的索引按并集处理，而非交集
#         ignore_index : bool
#             为真，沿着连接轴不使用原有索引值
#         keys : sequence
#             在数据的最外层添加多层索引
#         levels : list of sequences
#             用于构造多层索引的特定级别
#         names : list
#             多层索引中级别的名称
#         verify_integrity : bool
#             默认不检查新连接的轴是否包含重复项
#         sort : bool
#             join is ‘outer’，如果对非连接轴进行排序，join='inner'时无效
#     返回：取决于输入对象
df1 = pd.DataFrame([['B0', 'A0'], ['B1', 'A1']], columns=['B', 'A'])
df2 = pd.DataFrame([['B2', 'A2'], ['B3', 'A3']], columns=['B', 'A'])
df3 = pd.DataFrame([['B4', 'A4', 'C4'], ['B5', 'A5', 'C5'],
                    ['B6', 'A6', 'C6']], columns=['B', 'A', 'C'])
print(df1, df2, df3, sep='\n')

pd.concat([df1, df2])
pd.concat([df1, df2], sort=True)
pd.concat([df1, df2], keys=['x', 'y'])
pd.concat([df1, df3])
pd.concat([df1, df3], ignore_index=True)
pd.concat([df1, df3], join="inner")

pd.concat([df1, df2], axis=1)
pd.concat([df1, df3], axis=1)
pd.concat([df1, df3], axis=1, join="inner")

# 分类数据
s1 = pd.Series(['a', 'b'], dtype='category')
s2 = pd.Series(['a', 'b', 'a'], dtype='category')
s3 = pd.Series(['b', 'c'], dtype='category')
pd.concat([s1, s2])
pd.concat([s1, s3])
pd.concat([s1, s3]).astype('category')

# concat()(以及append())对数据进行了完整的复制，不断地重用这个函数请使用列表
lst = [pd.DataFrame([i], columns=['A']) for i in range(5)]
pd.concat(lst, ignore_index=True)

# merge_ordered(left, right, on=None, left_on=None, right_on=None,
#               left_by=None, right_by=None, fill_method=None,
#               suffixes=('_x', '_y'), how='outer')
#     用可选的填充方法执行合并
#     参数：
#         on : label or list
#             要连接的字段名，默认重叠列
#         left_on : abel or list, or array-like
#             left 的连接键
#         right_on : label or list, or array-like
#             right 的连接键
#         left_by : bool
#             将左侧的DataFrame按组列进行分组，并将右侧的DataFrame逐块合并
#         right_by : bool
#             按组列对右DataFrame进行分组，然后用左DataFrame逐块合并
#         fill_method : {‘ffill’, None}
#             数据插值法
#         how : {‘left’, ‘right’, ‘outer’, ‘inner’}
#             定义合并的方式：
#                 left: 使用 self 的连接键
#                 right: 使用 right 的连接键
#                 outer: 使用 self 和 right 连接键的并集
#                 inner: 使用 self 和 right 连接键的交集
#     返回：DataFrame
df1 = pd.DataFrame({'A': ['A2', 'A1', 'A0', 'A3'],
                    'B': [1, 2, 3, 4],
                    'D': ['D0', 'D1', 'D2', 'D3']})
df2 = pd.DataFrame({'A': ['A2', 'A1', 'A4'],
                    'C': [4, 5, 6]})
print(df1, df2, sep='\n')

pd.merge(df1, df2, how='outer')
pd.merge_ordered(df1, df2)
# left_by='D' ，df2缺失的'D'会被分别填充为 D0...D3,合并到输出中
df3 = pd.merge_ordered(df1, df2, left_by='D')
print(df3)

# 对df3['A']重新排序
df3['A'] = df3['A'].astype('category')
print(df3['A'])
df3['A'].cat.reorder_categories(pd.merge(df1, df2, how='outer')['A'],
                                inplace=True, ordered=True)
print(df3['A'])
print(df3.sort_values('A'))
print(df3)
pd.merge_ordered(df1, df2, fill_method='ffill', left_by='D')

# merge_asof(left, right, on=None, left_on=None, right_on=None,
#            left_index=False, right_index=False, by=None, left_by=None,
#            right_by=None, suffixes=('_x', '_y'), tolerance=None,
#            allow_exact_matches=True, direction='backward')
#     执行asof合并，匹配最近的键
#     参数：
#         left: DataFrame
#         right: DataFrame
#         on : label
#             要连接的字段名，必须在两个DataFrame中都能找到
#         left_on : abel or list, or array-like
#             left 的连接键
#         right_on : label or list, or array-like
#             right 的连接键
#         by : column name or list of column names
#             在执行合并操作之前匹配这些列
#         left_by : bool
#             将左侧的DataFrame按组列进行分组，并将右侧的DataFrame逐块合并
#         right_by : bool
#             按组列对右DataFrame进行分组，然后用左DataFrame逐块合并
#         suffixes : 2-length sequence (tuple, list, …)
#             应用于重叠列名的后缀
#         tolerance : int or Timedelta
#             asof公差
#         allow_exact_matches : bool
#             默认允许匹配相同值
#         direction : ‘backward’ (default), ‘forward’, or ‘nearest’
#             选择先前的、后续的或最接近的匹配
#     返回：DataFrame
df1 = pd.DataFrame({'K': [1, 5, 10], 'A': ['A1', 'A2', 'A3']})
df2 = pd.DataFrame({'K': [1, 2, 3, 6, 7], 'B': ['B1', 'B2', 'B3', 'B4', 'B5']})
print(df1, df2, sep='\n')

pd.merge_asof(df1, df2)
pd.merge_asof(df1, df2, direction='forward')
pd.merge_asof(df1, df2, direction='nearest')

pd.merge_asof(df1, df2, allow_exact_matches=False)
pd.merge_asof(df1.set_index('K'), df2.set_index('K'), left_index=True,
              right_index=True)

df1 = pd.DataFrame({'T': pd.to_datetime(['20190525 13:30:00.023',
                                         '20190525 13:30:00.038',
                                         '20190525 13:30:00.048',
                                         '20190525 13:30:00.048',
                                         '20190525 13:30:00.048']),
                    'G': ['B', 'B', 'A', 'A', 'C'],
                    'V2': ['S1', 'S2', 'S3', 'S4', 'S5'],
                    'V3': ['T1', 'T2', 'T3', 'T4', 'T5']})

df2 = pd.DataFrame({'T': pd.to_datetime(['20190525 13:30:00.023',
                                         '20190525 13:30:00.023',
                                         '20190525 13:30:00.030',
                                         '20190525 13:30:00.041',
                                         '20190525 13:30:00.048',
                                         '20190525 13:30:00.049',
                                         '20190525 13:30:00.072',
                                         '20190525 13:30:00.075']),
                    'G': ['A', 'B', 'B', 'B', 'A', 'C', 'A', 'B'],
                    'V4': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'],
                    'V5': ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8']})
print(df1, df2, sep='\n')

pd.merge_asof(df1, df2, on='T', by='G')
pd.merge_asof(df1, df2, on='T', by='G', tolerance=pd.Timedelta('2ms'))

# wide_to_long(df, stubnames, i, j, sep='', suffix='\d+')
#     将 DataFrame 从宽格式改为长格式
#     参数：
#         stubnames : str or list-like
#             列名分离后的前半部分，连同原始列名对应的数据列保留在DataFrame中
#         i : str or list-like
#             指定列，用作构建多层索引index(i, j)
#         j : str
#             列名分离后的后半部分，用作构建多层索引index(i, j)
#         sep : str
#             分隔符，用于分离列名
#             如：列名A-suffix1, A-suffix2，可通过指定sep= ' - '去掉连字符
#         suffix : str
#             捕获所需后缀的正则表达式
#     返回：DataFrame
df1 = pd.DataFrame({'A_0': list('abc'),
                    'A_1': list('def'),
                    'B_0': rng.integers(5, size=3),
                    'B_1': rng.integers(5, 10, size=3),
                    'C_one': rng.integers(10, 15, size=3),
                    'id': ['row1', 'row2', 'row3']})
print(df1)
pd.wide_to_long(df1, stubnames='A_', i='id', j='N')
pd.wide_to_long(df1, stubnames='A', i='id', j='N', suffix='_\d+')
pd.wide_to_long(df1, stubnames=['A', 'B'], i='id', j='N', sep='_')
pd.wide_to_long(df1, stubnames='C', i=['id', 'A_0', 'A_1'],
                j='N', sep='_', suffix='\w+')

# 使用unstack还原
df2 = pd.wide_to_long(df1, stubnames='B', i=['A_0', 'A_1'], j='N', sep='_')
print(df2, df1, sep='\n')
df2 = df2.unstack()
print(df2, df2.columns, sep='\n')
df2.columns = df2.columns.map('{0[0]}_{0[1]}'.format)
print(df2)
df2.reset_index(inplace=True)
df2 = df2.T.drop_duplicates().T
print(df2)
df2.rename(columns={'C_one_0': 'C_one', 'id_0': 'id'}).sort_index(axis=1)
print(df1)

# crosstab(index, columns, values=None, rownames=None, colnames=None,
#          aggfunc=None, margins=False, margins_name='All', dropna=True,
#          normalize=False)
#     默认计算数据列中的元素按索引组合后出现的频率
#     参数：
#         values: array-like
#             按aggfunc聚合的数组
#         index, columns:: array-like, Series, or list of arrays/Series
#             构造新DataFrame的行列
#         rownames, colnames: sequence
#             轴名称
#         aggfunc: function
#             聚合函数
#         margins: bool
#             为真时，数据部分行与列都会添加聚合汇总
#         margins_name: str
#             聚合汇总使用的标签名
#         dropna: bool
#             默认丢弃条目都缺失的行或列
#         normalize: bool, {‘all’, ‘index’, ‘columns’}, or {0,1}
#             通过将所有值除以值的和来规范化：
#             'all'/True：对所有值进行规范化
#             'index'/0：对每一行进行规范化
#             'columns'/1：将对每个列进行规范化
#             False：不进行规范化
#             margins=True，对margins值规范化
df = pd.DataFrame({'A': ['A0'] * 5 + ['B0'] * 4,
                   'B': ['B0'] + ['B0', 'B0', 'B1', 'B1'] * 2,
                   'C': ['C0', 'C1', 'C1'] + ['C0', 'C0', 'C1'] * 2,
                   'D': rng.integers(9, size=9),
                   'E': rng.integers(9, size=9)})
print(df)

df.pivot_table(df, index=['C'], columns=['A'], fill_value=0, aggfunc='size')
pd.crosstab(index=df['C'], columns=[df['A']])
pd.crosstab(index=df['C'], columns=[df['A'], df['B']])
pd.crosstab(index=df['C'], columns=[df['A'], df['B']],
            rownames=['r_lv0'], colnames=['c_lv0', 'c_lv1'])
pd.crosstab(index=df['C'], columns=df['A'], margins=True)
pd.crosstab(index=df['C'], columns=[df['A'], df['B']], margins=True,
            margins_name='M0').rename(columns={'': 'M1'}, level=1)

# 数据规范化
pd.crosstab(index=df['C'], columns=[df['A'], df['B']], normalize=0)
pd.crosstab(index=df['C'], columns=df['A'], normalize=True, margins=True)
pd.crosstab(index=df['C'], columns=df['A'], normalize=1, margins=True)

# 传递一个值组和一个聚合函数
pd.pivot_table(df, values='D', index=['C'], columns=['A', 'B'], aggfunc=np.sum)
pd.crosstab(values=df['D'], index=df['C'], columns=[df['A'], df['B']],
            aggfunc=np.sum)

# 应用于分类对象，任何传递的类别都包含在交叉列表中，即使实际数据不包含特定类别
cat1 = pd.Categorical(['A0', 'A1'], categories=['A0', 'A1', 'A2'])
cat2 = pd.Categorical(['B0', 'B1'], categories=['B0', 'B1', 'B2'])
pd.crosstab(index=cat1, columns=cat2)
pd.crosstab(index=cat1, columns=cat2, dropna=False)

# %% 缺失数据顶级函数
# isna(obj)
#     检测 array-like 对象的缺失值
#     参数：
#         obj: scalar or array-like
#             标量或类数组对象
#     返回：标量参数(包括字符串) 返回 <class 'bool'>
#           Index索引返回 <class 'numpy.ndarray'>
#           Series/DataFrame/ndarrays 按输入类型返回
# notna(obj)
#     检测 array-like 对象的非缺失值
pd.isna(pd.NA)
pd.notna(pd.NA)

index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None, "2017-07-08"])
pd.isna(index)
pd.notna(index)

df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
pd.isna(df)
pd.notna(df)
pd.isna(df[1])
pd.notna(df[1])

# %% 顶级转换函数
# to_numeric(arg, errors='raise', downcast=None)
#     将参数转换为数值类型
#     参数：
#         arg: scalar, list, tuple, 1-d array, or Series
#         errors: {‘ignore’, ‘raise’, ‘coerce’}
#             ‘raise’: 无效的解析将引发异常
#             ‘coerce’: 无效的解析将被设置为NaN
#             ‘ignore’: 效的解析将返回输入
#         downcast: {‘integer’, ‘signed’, ‘unsigned’, ‘float’}
#             将结果数据向下转换为尽可能小的数值dtype
#     返回：类型取决于输入。Series 返回out: <class 'pandas.Series'>
#             否则为 ndarray
ser = pd.Series(['1.0', '2', -3])
pd.to_numeric(ser, downcast='float')
pd.to_numeric(ser, downcast='signed')

ser = pd.Series(['apple', '1.0', '2', -3])
pd.to_numeric(ser, errors='ignore')
pd.to_numeric(ser, errors='coerce')

# %% datetime日期时间处理顶级函数
# to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None,
#             format=None, exact=True, unit=None, infer_datetime_format=False,
#             origin='unix', cache=True)
#     将参数转换为 datetime
#     参数：
#         arg: int, float, str, datetime, list, tuple, 1-d array,
#              Series, DataFrame, dict-like
#         errors: {‘ignore’, ‘raise’, ‘coerce’}
#             ‘raise’: 无效的解析将引发异常
#             ‘coerce’: 无效的解析将被设置为NaN
#             ‘ignore’: 效的解析将返回输入
#         dayfirst: bool
#             以天日期为第一解析顺序
#         yearfirst: bool
#             以年日期为第一解析顺序
#         format: str
#             指定字符串格式化时间
#         infer_datetime_formatbool
#             为真且未指定格式，则尝试推断日期时间字符串的格式，提高解析速度
#     返回：返回类型取决于输入类型
dt = ['2000-03-31 00:00:00', '2000-05-31 00:00:00', '2000-08-31 00:00:00']
pd.to_datetime(dt)

df = pd.DataFrame({'year': [2019, 2020], 'month': [2, 3], 'day': [4, 5]})
print(df)
pd.to_datetime(df)

# pandas.to_timedelta(arg, unit='ns', errors='raise')
#     将参数转换为时间增量
#     参数：
#         arg: str, timedelta, list-like or Series
#         errors: {‘ignore’, ‘raise’, ‘coerce’}
#             ‘raise’: 无效的解析将引发异常
#             ‘coerce’: 无效的解析将被设置为NaN
#             ‘ignore’: 效的解析将返回输入
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
#     返回：返回类型取决于输入:
pd.to_timedelta('1 days 06:05:01.00003')
pd.to_timedelta(['1 days 06:05:01.00003', '15.5us', 'nan'])
pd.to_timedelta(np.arange(5), unit='d')

# date_range(start=None, end=None, periods=None, freq=None, tz=None,
#            normalize=False, name=None, closed=None, **kwargs)
#     返回一个固定频率的DatetimeIndex
#     参数：
#         start: str or datetime-like
#             日期左界
#         end: str or datetime-like
#             日期右界
#         periods: int
#             周期数
#         freq: str or DateOffset
#             频率字符串，默认为’D’，datetime为1天
#         name: str
#             索引名称
#         closed: {None, ‘left’, ‘right’}
#             控制区间封闭,默认闭区间
#     返回：
#         out: <class 'pandas.DatetimeIndex'>
# 日期时间索引
pd.date_range(start=pd.Timestamp('2019-01-01'), end=pd.Timestamp('2019-01-09'),
              freq='2D')
pd.date_range(start=pd.Timestamp('2019-01-01'), end=pd.Timestamp('2019-01-09'),
              periods=4)
pd.date_range(start=pd.Timestamp('2017-01-01'), periods=3, freq='MS')

pd.date_range(start=pd.Timestamp('2019-01-01'), end=pd.Timestamp('2019-01-09'),
              closed='left')

# timedelta_range(start=None, end=None, periods=None, freq=None, name=None,
#                 closed=None)
#     返回一个固定频率的 TimedeltaIndex
#     参数：
#         start: str or timedelta-like
#             日期左界
#         end: str or timedelta-like
#             日期右界
#         periods: int
#             周期数
#         freq: str or DateOffset
#             频率字符串，默认为’D’，datetime为1天
#         name: str
#             索引名称
#         closed: {None, ‘left’, ‘right’}
#             控制区间封闭,默认闭区间
#     返回：
#         out: <class 'pandas.TimedeltaIndex'>
pd.timedelta_range(start=pd.Timedelta(1, unit='W'), periods=4)
pd.timedelta_range(start=pd.Timedelta(1, unit='W'), periods=4, closed='right')

pd.timedelta_range(start='1 day', end='2 days', freq='6H')
pd.timedelta_range(start='1 day', end='5 days', periods=4)

# period_range(start=None, end=None, periods=None, freq=None, name=None)
#     返回一个固定频率的周期频率
#     参数：
#         start: str or period-like
#             周期左界
#         end: str or period-like
#             周期右界
#         periods: int
#             周期数
#         freq: str or DateOffset
#             频率字符串，默认为’D’，datetime为1天
#         name: str
#             索引名称
#     返回：
#         out: <class 'pandas.PeriodIndex'>
pd.period_range(start='2020-01-01', end='2021-01-01', freq='M')
pd.period_range(start=pd.Period('2020Q1', freq='Q'),
                end=pd.Period('2020Q2', freq='Q'), freq='M')

# %% 区间处理顶级函数
# interval_range(start=None, end=None, periods=None, freq=None, name=None,
#                closed='right')
#     返回一个固定间隔的索引
#     参数：
#         start: numeric or datetime-like
#             区间左界
#         end: numeric or datetime-like
#             区间右界
#         periods: int
#             周期数
#         freq: numeric, str, or DateOffset
#             区间长度，默认数值为1，datetime为1天
#         name: str
#             区间索引的名称
#         closed: {‘left’, ‘right’, ‘both’, ‘neither’}
#             控制区间封闭
#     返回：
#         out: <'pandas.IntervalIndex'> 索引
# 数字索引
pd.interval_range(start=0, end=8, periods=4)
pd.interval_range(start=0, end=8, freq=2)
pd.interval_range(start=0, freq=2, periods=4)
pd.interval_range(end=8, freq=2, periods=4)

# 日期时间索引
pd.interval_range(start=pd.Timestamp('2019-01-01'),
                  end=pd.Timestamp('2019-01-09'), freq='2D')
pd.interval_range(start=pd.Timestamp('2019-01-01'),
                  end=pd.Timestamp('2019-01-09'), periods=4)
pd.interval_range(start=pd.Timestamp('2017-01-01'), periods=3, freq='MS')

# 控制区间开闭
pd.interval_range(end=5, periods=4, closed='both', name='numeric')

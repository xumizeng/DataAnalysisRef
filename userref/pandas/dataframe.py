# -*- coding: utf-8 -*-
"""
Created on 2020/4/6 10:36 by PyCharm

@author: xumiz
"""
# %% 模块导入
import pandas as pd
import numpy as np
from pathlib import Path
import stat
import shutil

# import matplotlib

# matplotlib.use('Agg')
# from matplotlib import pyplot as plt

# %% DataFrame 对象
# class DataFrame
# DataFrame(data=None, index=None, columns=None, dtype=None,
#                  copy= False)
#     构造一个二维、大小可变、可能异构的 DataFrame 表格数据
#     参数：
#         data: ndarray , Iterable, dict, or DataFrame
#             数据源
#         index: Index or array-like
#             输出的行标签，默认为RangeIndex(0, 1, 2, …, n)
#         columns: Index or array-like
#             输出的列标签，默认为RangeIndex(0, 1, 2, …, n)
#         dtype: dtype
#             强制的数据类型，默认解析器自行推断
#         copy: bool
#             从输入中复制数据，影响 DataFrame / 2d ndarray 类型的输入
#             根据数据类型的不同，创建可能需要复制数据，即使copy=False
#     返回:
#         out: <'pandas.DataFrame'>
# 构造 DataFrame

# 索引和选择数据
rng = np.random.default_rng()
df1 = pd.DataFrame(rng.integers(-3, 9, size=(5, 4)),
                   index=pd.date_range('20200101', periods=5),
                   columns=list('ABCD'))
print(df1)

# 基本索引使用方括号
df2 = df1.copy()
print(df2)
print(df2['A'])  # Series
print(df2[::-2], sep='\n')  # DataFrame

df2.loc[:, ['B', 'A']] = df2.loc[:, ['A', 'B']].to_numpy()  # 列赋值
df2[3:] = 0  # 行赋值
print(df2)
df2['E'] = df2['A']
print(df2)
df2.loc[pd.Timestamp('2020-01-06')] = 5
print(df2)

print(df2[df2.lt(0)])
# 等同于 df2[df2['A'].lt(0)]
print(df2.query('A < 0'))
# 等同于 df2[df2['A'].lt(0) & df2['A'].gt(-2)]
print(df2.query('A < 0 & A > -2'))
# 等同于 df2[df2['A'].lt(0) | df2['A'].gt(5)]
print(df2.query('A < 0 | A > 5'))
# 等同于 df2[~df2['A'].lt(0)]
print(df2.query('~ (A < 0)'))

# 指定数据类型
# dtype='category'等价于dtype=CategoricalDtype()
df = pd.DataFrame({'A': list('abca'), 'B': list('bccd')}, dtype="category")
print(df, df['A'], df.dtypes, sep='\n')

cat_type = pd.CategoricalDtype(categories=['b', 'c', 'd'], ordered=True)
df = pd.DataFrame({'A': list('abca'), 'B': list('bccd')}, dtype=cat_type)
print(df, df['A'], df.dtypes, sep='\n')

# %% DataFrame 属性
# loc     通过标签或布尔数组访问一组行和列
# iloc    通过纯整数位置访问一组行和列
#         通过.loc和.iloc设置Series和DataFrame数据时，会自动对齐所有轴
# 返回单个值
df2 = df1.copy()
print(df2)
print(df2.loc['2020-01-04', 'A'])  # 等同于 df2['A']['2020-01-04']
df2.loc['2020-01-04', 'A'] = 9
print(df2)

# 以Series类型返回指定行
df2 = df1.copy()
print(df2)
print(df2.loc['2020-01-02'])
# 等同于 df2.loc['2020-01-02'].loc['B':]
print(df2.loc['2020-01-02', 'B':])
# 等同于 df2.loc['2020-01-02'].loc[['B', 'D']]
print(df2.loc['2020-01-02', ['B', 'D']])
# 等同于 df2.loc['2020-01-02'].loc[df2.loc['2020-01-02'].gt(0)]
print(df2.loc['2020-01-02', df2.loc['2020-01-02'].gt(0)])
df2.loc['2020-01-02', df2.loc['2020-01-02'].gt(0)] = 0
print(df2)

# 以Series类型返回指定列
df2 = df1.copy()
print(df2)
# 等同于 df2['A']
print(df2.loc[:, 'A'])
# 等同于 df2['A'].loc['20200102':'20200104']
print(df2.loc['20200102':'20200104', 'A'])  # 字符串在切片可以转换为索引的类型
# 等同于 df2['A'].loc[[pd.Timestamp('20200102'), pd.Timestamp('20200104')]]
print(df2.loc[[pd.Timestamp('20200102'), pd.Timestamp('20200104')], 'A'])
# 等同于 df2['A'].loc[df2['A'].gt(0)]
print(df2.loc[df2['A'].gt(0), 'A'])
df2.loc[df2['A'].gt(0), 'A'] = 0
print(df2)

# 以DataFrame类型返回指定行
df2 = df1.copy()
print(df2)
print(df2.loc[[pd.Timestamp('20200102'), pd.Timestamp('20200104')]])
print(df2.loc['20200102':'20200104'])
# 等同于 df2[df2['A'].lt(0)]
print(df2.loc[df2['A'].lt(0)])
df2.loc[df2['A'].lt(0)] = 0
print(df2)

# 以DataFrame类型返回指定列
df2 = df1.copy()
print(df2)

print(df2.loc[:, ['A', 'D']])
print(df2.loc[:, 'A':'C'])
print(df2.loc[:, df2.loc[pd.Timestamp('20200102')].gt(0)])
df2.loc[:, df2.loc[pd.Timestamp('20200102')].gt(0)] = 0
print(df2)

# 以DataFrame类型返回指定行列
df2 = df1.copy()
print(df2)
print(df2.loc[[pd.Timestamp('20200102'), pd.Timestamp('20200104')],
              ['A', 'C']])
print(df2.loc['20200102':'20200104', 'B':'C'])
print(df2.loc[df2['A'].lt(0), df2.loc[pd.Timestamp('20200102')].gt(0)])
df2.loc[df2['A'].lt(0), df2.loc[pd.Timestamp('20200102')].gt(0)] = 0
print(df2)

# 多层索引
index = pd.MultiIndex.from_product([['A', 'B', 'C'], ['mark1', 'mark2']])
col = pd.MultiIndex.from_product([['one', 'two'], ['first', 'second']])
values = rng.integers(0, 9, size=(6, 4))
df = pd.DataFrame(values, index=index, columns=col, copy=True)
print(df)

print(df.loc['A'])
print(df.loc['A', 'mark1'])
print(df.loc[[('A', 'mark1')]])
print(df.loc[('A', 'mark2'):('C', 'mark1')])

print(df.loc[:, ('one', 'second')])

# at      访问行/列标签对的单个值
# iat     通过整数位置访问行/列对的单个值
df2 = df1.copy()
print(df2)
print(df2.at['2020-01-02', 'B'])
df2.at[pd.Timestamp('2020-01-06'), 'E'] = 7
print(df2)

# %% DataFrame方法：转换
# astype(self, dtype, copy=True, errors='raise')
#     将对象转换为指定的dtype
#     参数：
#         dtype: data type, or dict
#             数据类型
#         errors: {‘raise’, ‘ignore’},
#             控制对提供的dtype的无效数据的异常引发
#             raise: 允许引发异常
#             ignore: 错误时返回原始对象
#     返回:
#         out: <'pandas.DataFrame'>
df = pd.DataFrame({'A': list('abca'), 'B': list('bccd')})
df_cat = df.astype('category')
print(df_cat, df_cat['A'], df_cat.dtypes, sep='\n')

df["C"] = df["A"].astype('category')
print(df, df['C'], df.dtypes, sep='\n')

cat_type = pd.CategoricalDtype(categories=['b', 'c', 'd'], ordered=True)
df["D"] = df["A"].astype(cat_type)
print(df, df['D'], df.dtypes, sep='\n')

# convert_dtypes(self, infer_objects=True, convert_string=True,
#                convert_integer=True, convert_boolean=True)
#     将对象dtypes转换为可能的最佳类型
#     参数：
#         infer_objects: bool, default True
#             将对象dtypes转换为可能的最佳类型
#         convert_string: bool, default True
#             将对象dtypes转换为StringDtype()
#         convert_integer: bool, default True
#             将对象dtypes转换为整数扩展类型
#         convert_boolean: bool, defaults True
#             将对象dtypes转换为BooleanDtypes()
#     返回: DataFrame
# infer_objects(self)
#     将对象dtypes转换为可能的最佳类型
df = pd.DataFrame({'a': pd.Series([1, 2, 3], dtype=np.dtype('O')),
                   'b': pd.Series(['x', 'y', 'z'], dtype=np.dtype('O')),
                   'c': pd.Series([True, False, np.nan], dtype=np.dtype('O')),
                   'd': pd.Series(['h', 'i', np.nan], dtype=np.dtype('O')),
                   'e': pd.Series([1, np.nan, 2], dtype=np.dtype('float')),
                   'f': pd.Series([np.nan, 1.5, 2], dtype=np.dtype('float')),
                   })
print(df)
df.convert_dtypes()
df.infer_objects()
print(df.dtypes, df.convert_dtypes().dtypes, sep='\n')
print(df.dtypes, df.infer_objects().dtypes, sep='\n')

# to_numpy(self, dtype=None, copy=False)
#     将 DataFrame 转换为 NumPy 数组
#     参数：
#         dtype: str or numpy.dtype
#             要传递给numpy.asarray()的dtype
#     返回:
#         out: <'numpy.ndarray'>
df2 = df1.copy()
print(df2)
df2.to_numpy()

# %% DataFrame方法：索引
# head(self, n=5)
#     返回前n行
#     参数：
#         n: int
#             要选择的行数
#     返回:
#         out: <'pandas.DataFrame'>
# tail(self, n=5)
#     返回最后n行
df2 = df1.copy()
print(df2)
df2.head(3)
df2.head(-3)
df2.tail(3)
df2.tail(-3)

# get(self, key, default=None)
#     从对象中获取给定键的项，如果没有找到，返回默认值
df2 = df1.copy()
print(df2)
df2.get('A')
df2['A'].get('2020-01-01')
df2.get('2020-01-01', default='未找到！')

# xs(self, key, axis=0, level=None, drop_level=True)
#     选择数据，但不能用来设置值
tuples = [('A0', 'B0', 'C0'), ('A0', 'B1', 'C0'),
          ('A0', 'B2', 'C1'), ('A1', 'B4', 'C0')]
index = pd.MultiIndex.from_tuples(tuples, names=('lvl1', 'lvl2', 'lvl3'))
df = pd.DataFrame(np.arange(len(index) * 2).reshape(len(index), 2),
                  index=index, columns=['x', 'y'])
print(df)

df.xs('A0')
df.xs(('A0', 'B1'))
df.xs('B0', level=1)
df.xs('B0', level='lvl2')
df.xs(('A1', 'C0'), level=[0, 'lvl3'])
df.xs(('A1', 'C0'), level=[0, 'lvl3'], drop_level=False)
df.xs('x', axis=1)

# lookup(self, row_labels, col_labels)
#     给定行和列标签的等长数组，提取每组(row, col) 对的值
#         返回：numpy.ndarray
df2 = df1.copy()
print(df2)

df2.lookup([pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'),
            pd.Timestamp('2020-01-04'), pd.Timestamp('2020-01-03')],
           ['B', 'C', 'A', 'D'])

# idxmin(self, axis=0, skipna=True, *args, **kwargs)
#     返回最小值的行标签,如果多个值等于最小值，则返回具有该值的第一行标签
#         返回：Series
# idxmax(self, axis=0, skipna=True, *args, **kwargs)
d = [[0.0, np.nan, -1.0, 1.0],
     [np.nan, 2.0, np.nan, np.nan],
     [2.0, 3.0, np.nan, 9.0],
     [np.nan, 4.0, -4.0, 16.0]]
df = pd.DataFrame(d, columns=list('abcd'), copy=True)
df.idxmax()
df.idxmin()
df.idxmax(axis=1)
df.idxmin(axis=1)

# sample(self, n=None, frac=None, replace=False, weights=None,
#        random_state=None, axis=None)
#     从目标轴返回项目的随机样本
#     参数：
#         n: int, Default = 1 if frac = None
#             返回的项数
#         frac: float
#             返回的占比
#         replace: bool
#             默认不允许同一行的多次采样（如果 frac > 1，则应将置换设置为True）
#         weights: str or ndarray-like
#             默认权重相同
#         random_state: int or numpy.random.RandomState
#             设置随机数生成器种子或 numpy RandomState 对象，使得随机数得以重现
#     返回: DataFrame
df2 = df1.copy()
print(df2)

df2.sample(n=3)
df2.sample(frac=0.6)
df2.sample(n=5, replace=False)
df2.sample(n=5, replace=True)
df2.sample(n=3, weights=[0, 0, 0.2, 0.2, 0.6])

df2.sample(n=3, axis=1)
df2.sample(frac=0.6, axis=1)
df2.sample(n=4, replace=False, axis=1)
df2.sample(n=4, replace=True, axis=1)
df2.sample(n=2, weights=[0, 0, 0.4, 0.6], axis=1)

df2.sample(n=3, random_state=2)
df2.sample(n=3, random_state=2)

# take(self, indices, axis=0, is_copy=None, **kwargs)
#     沿着轴返回给定位置索引中的元素
df = pd.DataFrame([('a', 'b', 3.0), ('c', 'b', 4.0),
                   ('d', 'e', 8), ('f', 'e', np.nan)],
                  columns=['A', 'B', 'C'], index=[0, 2, 3, 1])
df.set_index(['B', 'A'], inplace=True)
print(df)

df.take([0, 3])
df.take([-1, -2])
df.take([0], axis=1)

# where(self, cond, other=nan, inplace=False, axis=None, level=None,
#       errors='raise', try_cast=False)
#     替换条件为假时的值
#     参数：
#         cond: bool Series/DataFrame, array-like, or callable
#         other: scalar, Series/DataFrame, or callable
#             对于DataFrame中的每个数据，cond为假则使用other中的对应数据
#     返回: DataFrame
# mask(self, cond, other=nan, inplace=False, axis=None, level=None,
#      errors='raise', try_cast=False)
#     替换条件为真时的值
rng = np.random.default_rng()
idx = pd.date_range('20200101', periods=4)
df = pd.DataFrame(rng.integers(-3, 9, size=(4, 4)), index=idx,
                  columns=list('ABCD'))
print(df)
df.where(df.gt(0), -df)
df.where(df.gt(0), 0)
df.where(df.gt(0), df['A'], axis=0)
df.where(df.gt(0), df.loc['2020-01-01'], axis=1)
df.where(lambda z: z > 4, lambda z: z + 10)

# query(self, expr, inplace=False, **kwargs)
#     使用布尔表达式查询DataFrame的列
#     返回: DataFrame
rng = np.random.default_rng()
index = pd.RangeIndex(9, -1, -1, name='a')
df = pd.DataFrame(rng.integers(0, 9, size=(10, 3)),
                  columns=list('ab') + ['c c'], index=index)
print(df)

# 列名中含有空格或运算符使用反引号
df.query('(a < b) & (b < `c c`)')

# 查询表达式中使用index表示未命名索引,如果索引名与列名重叠，则列名具有优先级
df.query('(index == 3)')
df.query('(index  < a) & (a < b)')
df.query('(a == 0) & (a < b)')

# 使用“@”字符前缀来引用环境中的变量
x = 6
df.query('(a < b) & (b < @x)')

# 多层索引
m = pd.Series(['row1', 'row2']).sample(8, replace=True)
n = pd.Series(['v1', 'v2']).sample(8, replace=True)
index = pd.MultiIndex.from_arrays([m, n])
df = pd.DataFrame(rng.integers(0, 9, size=(8, 2)), index=index)
df.sort_index(level=0, inplace=True)
print(df)
df.query('ilevel_0 == "row1"')
df.query('ilevel_1 == "v2"')

df.index.set_names(['A', 'B'], inplace=True)
print(df)
df.query('A == "row1"')
df.query('B == "v2"')

# 将相同的查询传递给两个 DataFrame
df1 = pd.DataFrame(rng.integers(0, 9, size=(5, 2)), columns=list('ab'))
df2 = pd.DataFrame(rng.integers(0, 9, size=(5, 2)), columns=list('ab'))
print(df1, df2, sep='\n')

for i in map(lambda frame: frame.query('0 <= a <= b <= 5'), [df1, df2]):
    print(i)

# in和not in操作符
df = pd.DataFrame({'a': list('abbcddeeff'), 'b': list('aaabbbcccc'),
                   'c': rng.integers(0, 9, size=10),
                   'd': rng.integers(0, 9, size=10)})
print(df)

df.query('a in b')
df.query('a not in b')
df.query('(a in b) & (c < d)')
df.query('b not in ["a", "c"]')
df.query('[1, 2, 6] in c')

# 布尔运算符
df = pd.DataFrame(rng.integers(0, 9, size=(5, 2)), columns=list('ab'))
df['c'] = rng.integers(0, 9, size=5) > 5
print(df)
df.query('c')
df.query('~c')

# duplicated(self, subset=None, keep='first')
#     识别重复行
#     参数：
#         subset: column label or sequence of labels
#             指定标识重复项的列，默认情况下使用所有列
#         keep: {‘first’, ‘last’, False}
#             first : 将重复项标记为True(第一次出现的除外)
#             last : 将重复项标记为True(最后一次出现的除外)
#             False : 将所有重复项标记为True
#     返回: Series
# drop_duplicates(self, subset=None, keep='first', inplace=False,
#                 ignore_index=False)
#     删除重复行（忽略索引）
#     参数：
#         keep: {‘first’, ‘last’, False}
#             first : 删除重复项(第一次出现的除外)
#             last : 删除重复项(最后一次出现的除外)
#             False : 删除所有重复项
#         ignore_index: bool
#             默认保留原索引
#     返回: DataFrame
df = pd.DataFrame({'a': ['one', 'one', 'two', 'two', 'two', 'three', 'four'],
                   'b': ['x', 'y', 'x', 'y', 'x', 'x', 'x'],
                   'c': rng.integers(0, 9, size=7)},
                  index=['a', 'a', 'b', 'c', 'b', 'a', 'c'])

print(df)
df.duplicated('a')
print(df[~df.duplicated('a')])
print(df.drop_duplicates('a'))

print(df.drop_duplicates('a', keep='last'))
print(df.drop_duplicates('a', keep=False))
print(df.drop_duplicates(['a', 'b']))
print(df.drop_duplicates(['a', 'b'], ignore_index=True))

df.index.duplicated()
print(df[~df.index.duplicated()])

# %% DataFrame方法：缺失数据处理
# isna(self)
#     检测缺失值，'' 或 numpy.inf 不视为缺失值，除非设置：
#     pandas.options.mode.use_inf_as_na = True
#     返回:
#         out: <'pandas.DataFrame'>
# notna(self)
#     检测非缺失值
d = [[5, pd.NaT, 'A', None],
     [6, pd.Timestamp('2019-05-27'), 'B', 'C'],
     [np.NaN, pd.Timestamp('2019-04-25'), '', 'D']]
df = pd.DataFrame(d, columns=['a', 'b', 'c', 'd'], copy=True)
df.isna()
df.notna()

# dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False)
#     删除缺失值
#     参数：
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             目标轴
#         how: {‘any’, ‘all’}
#             默认存在NA值，则删除该行/列
#                 ‘any’ : 存在NA值，则删除该行/列
#                 ‘all’ : 所有值都是NA，则删除该行/列
#         thresh: int
#             控制保留至少含有多少有效值（非na值）的行，默认全是有效值才保留
#         subset: array-like
#             定义在哪些列中查找丢失的值，默认所有列
#         inplace: bool
#             默认操作对象非输入数据源
#     返回:
#         out: <'pandas.DataFrame'>
d = [[5, pd.NaT, 'A', None],
     [6, pd.Timestamp('2019-05-27'), 'B', 'C'],
     [np.NaN, pd.Timestamp('2019-04-25'), '', 'D']]
df = pd.DataFrame(d, columns=['a', 'b', 'c', 'd'], copy=True)
df.dropna()
df.dropna(axis=1)

df.dropna(how='all')

df.dropna(thresh=3)
df.dropna(subset=['a', 'd'])

df.dropna(inplace=True)
print(df)

# fillna(self, value=None, method=None, axis=None, inplace=False, limit=None,
#        downcast=None)
#     使用指定的方法填充NA/NaN值
#     参数：
#         value: scalar, dict, Series, or DataFrame
#             填充值
#         method: {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}
#             填充方法
#             backfill / bfill: 沿轴将有效值向前的缺失值填充
#             pad / ffill: 沿轴将有效值向后的缺失值填充
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             目标轴,默认为0
#         inplace: bool
#             默认操作对象非输入数据源，inplace=True 操作输入数据，返回None
#         limit: int
#             向前/向后填充的连续NaN值的最大数目,默认尽可能填充
#         downcast: dict
#             定义向下转换类型
#     返回:
#         out: <'pandas.DataFrame'> 或 <class 'NoneType'>
d = [[np.nan, 2, 1, np.nan],
     [3, np.nan, np.nan, np.nan],
     [np.nan, np.nan, np.nan, 3],
     [np.nan, 3, np.nan, 2]]
df = pd.DataFrame(data=d, columns=list('ABCD'), copy=True)
print(df)

# 用标量替换所有的NaN元素
df.fillna(0)

# 按列指定替换元素
values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df.fillna(value=values)
values = {'C': 2}
df.fillna(value=values)

# 使用Pandas Object填充
df.mean()
df.fillna(df.mean())
df.fillna(df.mean()['B':'C'])

# 向后或向前填充非空值
df.fillna(method='ffill')
df.fillna(method='bfill')

# 向后或向前填充一个非空值
df.fillna(method='ffill', limit=1)
df.fillna(method='bfill', limit=1)

# 就地更改数据源
df.fillna(method='ffill', axis=1, inplace=True)
print(df)
df.fillna(method='bfill', axis=1, inplace=True)
print(df)

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
#             'quadratic': 处理一个以增长率增长的时间序列
#             'pchip': 处理值近似于一个累积分布函数的值
#             'akima': 用平滑绘图的目标填充缺失的值
#         axis: {0 or ‘index’, 1 or ‘columns’, None}
#             目标轴，默认 ‘index’
#         inplace: bool
#             默认操作对象非输入数据源，inplace=True 操作输入数据，返回None
#         limit: int
#             填充的连续NaN值的最大数目,默认尽可能填充
#         limit_direction: {‘forward’, ‘backward’, ‘both’}
#             填充的连续NaN值的方向
#         limit_area: {None, ‘inside’, ‘outside’}
#             填充的连续NaN值的区域:
#                 None: 无填充约束
#                 ‘inside’: 只填充被有效值包围的nan值
#                 ‘outside’: 只填充有效值之外的nan值
#         downcast: dict
#             定义转换类型
#         **kwargs
#             传递给插值函数的关键字参数
#     返回: DataFrame 或 NoneType
d = [[0.0, np.nan, -1.0, 1.0],
     [np.nan, 2.0, np.nan, np.nan],
     [2.0, 3.0, np.nan, np.nan],
     [np.nan, 4.0, -4.0, 16.0]]
df = pd.DataFrame(d, columns=list('abcd'), copy=True)
print(df)

df.interpolate()
df.interpolate(limit=1)
df.interpolate(limit_direction='backward')
df.interpolate(limit_direction='both', axis=1)
df.interpolate(limit_direction='both', limit=1)
df.interpolate(limit_direction='both', limit_area='inside', limit=1)

idx = pd.date_range('1/1/2020', periods=4, freq='M')
df = pd.DataFrame(d, columns=list('abcd'), index=idx, copy=True)
print(idx, df, sep='\n')
df.interpolate()
df.interpolate(method='time')
a = df.at['2020-04-30', 'd'] - df.at['2020-01-31', 'd']
b = (idx[3] - idx[0]).days
c = (idx[1] - idx[0]).days
print(a, b, c, a / b * c + df.at['2020-01-31', 'd'])

df = pd.DataFrame(d, columns=list('abcd'), index=[0, 1, 5, 10], copy=True)
print(df)
df.interpolate()
df.interpolate(method='values')
a = df.at[10, 'd'] - df.at[0, 'd']
b = df.index[3] - df.index[0]
c = df.index[1] - df.index[0]
print(a, b, c, a / b * c + df.at[0, 'd'])
df.plot()

ser = pd.Series(np.arange(1, 2.75, .25) ** 2 + rng.standard_normal(7))
missing = np.array([1, 2, 3])
ser[missing] = np.nan
print(ser)
methods = ['linear', 'quadratic', 'cubic', 'barycentric', 'pchip', 'akima']
df = pd.DataFrame({m: ser.interpolate(method=m) for m in methods})
print(df)
df.plot()

methods = ['spline', 'polynomial']
df = pd.DataFrame({m: ser.interpolate(method=m, order=2) for m in methods})
print(df)
df.plot()

# replace(self, to_replace=None, value=None, inplace=False, limit=None,
#         regex=False, method='pad')
#     用value替换to_replace
#     参数：
#         to_replace: str, regex, list, dict, Series, int, float, or None
#             被替换的值
#         value: scalar, dict, list, str, regex
#             替换值，当value=None且to_replace是标量、列表或元组时，replace使用
#             method='pad' 进行替换
#         limit: int
#             替换的最大数目,默认尽可能替换
#         regex: bool or same types as to_replace
#             默认to_replace、value不解释为正则表达式
#         method: {‘pad’, ‘ffill’, ‘bfill’, None}
#             bfill: 沿轴将有效值向前的缺失值填充
#             pad / ffill: 沿轴将有效值向后的缺失值填充
#     返回: DataFrame
# Scalar
df = pd.DataFrame({'A': [0, 1, 2, 3, 4],
                   'B': [5, 6, 7, 8, 9],
                   'C': ['a', '.', 'c', 'd', '.']})
print(df)
df.replace(0, 5)
df.replace('d', None)
df.replace('.', None)

# String
df.replace('.', np.nan)

# List-like
df.replace([0, 1, 3, 4], 4)
df.replace([0, 1, 2, 3], [4, 3, 2, 1])
df.replace([1, 2], method='bfill')

# dict-like
df.replace({0: 10, 1: 100})
df.replace({'A': 0, 'B': 5}, 100)
df.replace({'A': {0: 100, 4: 400}})
df.replace({'d': None})

# Regular expression
df = pd.DataFrame({'A': ['bat', 'foo', 'bait'],
                   'B': ['abc', 'bar', 'xyz']})
print(df)
df.replace(to_replace=r'^ba.$', value='new', regex=True)
df.replace(regex=r'^ba.$', value='new')
df.replace({'A': r'^ba.$'}, {'A': 'new'}, regex=True)
df.replace(regex={r'^ba.$': 'new', 'foo': 'xyz'})
df.replace(regex=[r'^ba.$', 'foo'], value='new')

# %% DataFrame方法：排序
# sort_index(self, axis=0, level=None, ascending=True, inplace=False,
#            kind='quicksort', na_position='last', sort_remaining=True,
#            ignore_index=False)
#     根据标签(沿轴)对对象排序
#     参数：
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             要排序的轴
#         level: int or level name or list
#             对指定索引级别中的值排序
#         ascending: bool
#             默认升序排列
#         inplace: bool
#             默认不对输入源进行操作
#         kind: {‘quicksort’, ‘mergesort’, ‘heapsort’}
#             选择排序算法，归并排序('mergesort')是唯一稳定的算法。对于DataFrames，
#             仅在对单个列或标签进行排序时应用此选项
#         na_position: {‘first’, ‘last’}
#             定义缺失值位置，对多层索引无效
#     返回: DataFrame or None
# 基本索引
df = pd.DataFrame({None: ['A', 'A', 'B', np.nan, 'D', 'C'],
                   'col1': [2, 1, 9, 8, 7, 4],
                   'col3': [0, 1, 9, 4, 2, 3]}, index=[None] + list('DACFE'))
print(df)
df.sort_index()
df.sort_index(ascending=False)
df.sort_index(axis=1)
df.sort_index(na_position='first')

# 多层索引
arrays = [['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
          ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y']]
arr = list(zip(*arrays))
print(arr)
rng.shuffle(arr)
print(arr)
df = pd.DataFrame(rng.integers(0, 9, size=(len(arr), 2)),
                  index=pd.MultiIndex.from_tuples(arr))
print(df)

df.sort_index()
df.sort_index(level=1)

# sort_values(self, by, axis=0, ascending=True, inplace=False,
#             kind='quicksort', na_position='last', ignore_index=False)
#     根据沿任意轴的值排序
#     参数：
#         by: str or list of str
#             要排序的名称或名称列表
#                 axis=0，by可包含索引级别、列标签
#                 axis=1，by可包含列级别、索引标签
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             要排序的轴
#         ascending: bool
#             默认升序排列升序
#         inplace: bool
#             默认不对输入源进行操作
#         kind: {‘quicksort’, ‘mergesort’, ‘heapsort’}
#             选择排序算法，归并排序('mergesort')是唯一稳定的算法。对于DataFrames，
#             仅在对单个列或标签进行排序时应用此选项
#         na_position: {‘first’, ‘last’}
#             定义缺失值位置，对多层索引无效
#     返回:
#         out: <'pandas.DataFrame'> or None
df = pd.DataFrame({'col2': ['A', 'A', 'B', np.nan, 'D', 'C'],
                   'col1': [2, 1, 9, 8, 7, 4],
                   'col3': [0, 1, 9, 4, 2, 3]}, index=[None] + list('DACFE'))
df.sort_values(by=['col2'])
df.sort_values(by=['col2'], ascending=False)
df.sort_values(by=['col2', 'col1'])
df.sort_values(by='col2', ascending=False, na_position='first')
df.sort_values(axis=1, by='C')

cat = pd.Categorical(list('bbeebaa'), categories=['e', 'a', 'b'], ordered=True)
df = pd.DataFrame({'A': cat, 'B': [1, 2, 1, 2, 2, 1, 2]})
print(df, df['A'], sep='\n')
df.sort_values(by=['A', 'B'])
df['A'] = df['A'].cat.reorder_categories(['a', 'b', 'e'])
print(df['A'])
df.sort_values(by=['A', 'B'])

# %% DataFrame方法：改变形状
# stack(self, level=-1, dropna=True)
#     将指定的级别从列堆叠到索引，新的索引级别已排序
#     参数：
#         level: int, str, list
#         dropna: bool
#             默认删除结果中数据原有缺失值的行
#     返回: DataFrame or Series
# unstack(self, level=-1, fill_value=None)
#     将指定的级别从索引堆叠到列，新的列级别已排序
#     参数：
#         level: int, str, list
#         dropna: bool
#             如果unstack产生了丢失的值，用这个值替换NaN, 数据原有缺失值不予替换
#     返回: DataFrame or Series
df = pd.DataFrame([[0, 1], [2, 3]], index=['row1', 'row2'],
                  columns=['col1', 'col2'])
print(df)
df.stack(0)
df.unstack(0)

col = pd.MultiIndex.from_tuples([('C', 'D'), ('B', 'A')])
df = pd.DataFrame([[1, 2], [2, 4]], index=['row1', 'row2'], columns=col)
print(df)
df.stack(0)
df.stack(1)
df.stack((0, 1))
df.stack((1, 0))

idx = pd.MultiIndex.from_tuples([('C', 'D'), ('B', 'A')])
df = pd.DataFrame([[1, 2], [2, 4]], index=idx, columns=['col1', 'col2'])
print(df)
df.unstack(0)
df.unstack(1)
df.unstack((0, 1))
df.unstack((1, 0))

col = pd.MultiIndex.from_arrays([['A0', 'A1', 'A1', 'A0'],
                                 ['B0', 'B1', 'B0', 'B1']])
idx = pd.MultiIndex.from_arrays([['C0', 'C0', 'C1', 'C1'],
                                 ['D0', 'D1', 'D0', 'D1']])
df = pd.DataFrame(rng.integers(9, size=[4, 4]), index=idx, columns=col)
print(df)
df.stack()
df.stack().sum(axis=1)
# 等同于 groupby(level=1, axis=1).sum()
df.stack().sum(axis=1).unstack()
print(df)
df.sum()
df.sum().unstack(0)

# 处理缺失值
col = pd.MultiIndex.from_tuples([('C', 'D'), ('B', 'A')])
df = pd.DataFrame([[None, 2], [2, 4]], index=['row1', 'row2'], columns=col)
print(df)
df.stack(0)
df.stack(0, dropna=False)

idx = pd.MultiIndex.from_tuples([('C', 'D'), ('B', 'A')])
df = pd.DataFrame([[None, 2], [2, 4]], index=idx, columns=['col1', 'col2'])
print(df)
df.unstack(0)
df.unstack(0, fill_value=True)

# melt(self, id_vars=None, value_vars=None, var_name=None, value_name='value',
#      col_level=None)
#     将 DataFrame 从宽格式改为长格式
#     参数：
#         id_vars: tuple, list, or ndarray
#             用作id的列
#         value_vars: tuple, list, or ndarray
#             未指定，则使用未设置为id_vars的所有列
#         var_name: scalar
#             未指定，则使用 frame.columns.name 或 ‘variable’.
#         value_name: scalar
#             用于“值”列的名称
#         col_level: int, str
#             应用于多索引
#     返回: DataFrame
df = pd.DataFrame({'A': ['A0', 'A1'], 'B': ['B0', 'B1'],
                   'X': ['X0', 'X1'], 'Y': ['Y0', 'Y1']})
print(df)

df.melt()
df.melt(id_vars=['A', 'B'], var_name='变量', value_name='值')
df.melt(id_vars=['A', 'B'], value_vars='X')

idx = pd.MultiIndex.from_arrays([list('ABXY'), list('CDEF')])
df = df.reindex(columns=idx, level=0)
print(df)
df.melt(col_level=0, id_vars=['A', 'B'])
df.melt(col_level=1, id_vars=['C', 'D'])
df.melt(id_vars=[('A', 'C')])
df.melt(id_vars=[('A', 'C'), ('B', 'D')])

# explode(self, column)
#     将 指定列中的 list-like 对象的每个元素转换为行，复制索引值
df = pd.DataFrame({'A': [[1, 2, 3], 'foo', [], [3, 4]],
                   'B': ['B0', 'B1', 'B2', 'B3']})
print(df)
df.explode('A')

df = pd.DataFrame({'A': ['a,b,c', 'd,e,f'],
                   'B': [1, 2]})
print(df)
df.assign(A=df['A'].str.split(','))
df.assign(A=df['A'].str.split(',')).explode('A')

# pivot(self, index=None, columns=None, values=None)
#     返回按给定索引/列值组织的重新构造的DataFrame
#     参数：
#         index: str or object
#             构造新DataFrame的索引，未指定则使用现有索引
#         columns: str or object
#             构造新DataFrame的列
#         values: str, object or a list of the previous
#             构造新DataFrame的值，未指定，将使用所有剩余的列值
#     返回: DataFrame
df = pd.DataFrame({'R': ['R1', 'R1', 'R1', 'R2', 'R2', 'R2'],
                   'C': ['C1', 'C2', 'C3', 'C1', 'C2', 'C3'],
                   'X': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6'],
                   'Y': ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6']})
print(df)
df.pivot(columns='C')
df.pivot(index='R', columns='C')
df.pivot(index='R', columns='C', values='Y')

# pivot_table(self, values=None, index=None, columns=None, aggfunc='mean',
#             fill_value=None, margins=False, dropna=True, margins_name='All',
#             observed=False)
#     创建一个类似电子表格样式的DataFrame数据透视表
#     参数：
#         values: column to aggregate
#             构造新DataFrame的值，未指定，将使用所有剩余的列值
#         index: str or object
#             构造新DataFrame的索引，未指定则使用现有索引
#         columns: column, Grouper, array, or list of the previous
#             构造新DataFrame的列
#         aggfunc: function, list of functions, dict, default numpy.mean
#             聚合函数
#         fill_value: scalar
#             替换缺失的值
#         dropna: bool
#             默认丢弃全为NaN的列
#         margins: bool
#             为真时，数据部分行与列都会添加聚合汇总
#     返回: DataFrame
df = pd.DataFrame({'A': ['A0'] * 5 + ['B0'] * 4,
                   'B': ['B0'] + ['B0', 'B0', 'B1', 'B1'] * 2,
                   'C': ['C0', 'C1', 'C1'] + ['C0', 'C0', 'C1'] * 2,
                   'D': rng.integers(9, size=9),
                   'E': rng.integers(9, size=9),
                   'F': [pd.Timestamp(2020, i, 1) for i in range(1, 6)] + [
                       pd.Timestamp(2020, i, 15) for i in range(1, 5)]})
print(df)
pd.pivot_table(df, values='D', index=['A', 'B'], aggfunc=np.sum)
pd.pivot_table(df, index=['A', 'B'], columns=['C'], aggfunc=np.sum)
pd.pivot_table(df, index=['A', 'B'], columns=['C'], aggfunc=np.sum,
               fill_value=0)
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum)
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'],
               aggfunc=np.sum, fill_value=0)
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'],
               aggfunc=[np.sum, np.mean], fill_value=0)
pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
               aggfunc={'D': np.sum, 'E': np.sum})
pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
               aggfunc={'D': np.sum, 'E': [min, max, np.sum]})

# 使用Grouper作为索引和列关键字
print(df)
idx = pd.Grouper(freq='M', key='F')
pd.pivot_table(df, values='D', index=idx, columns='C')

# 写入Excel
datapath = Path('../data')
datapath.mkdir()

excel_path = Path('../data/test.xlsx')
df_copy = pd.pivot_table(df, index=['A', 'B'], columns=['C'], margins=True,
                         aggfunc=np.sum)
print(df_copy)
with pd.ExcelWriter(excel_path) as writer:
    df_copy.to_excel(writer, sheet_name='表一')

# 分类数据
cat = pd.Categorical(['a', 'a', 'b', 'b'], categories=['b', 'a', 'c'])
df = pd.DataFrame({'A': cat, 'B': ['c', 'd', 'c', 'd'], 'C': [1, 2, 3, 4]})
print(df)
pd.pivot_table(df, values='C', index=['A', 'B'])


# 清空数据文件夹
def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal"""
    Path(path).chmod(stat.S_IWRITE)
    func(path)


shutil.rmtree(datapath, onerror=remove_readonly)

# reorder_levels(self, order, axis=0)
#     参数：
#         order: list of int or list of str
#             要交换的索引的级别
#     使用输入顺序重新排列索引级别
idx = pd.MultiIndex.from_product([['A', 'B'], ['y', 'x']],
                                 names=['lvl1', 'lvl2'])
df = pd.DataFrame(rng.integers(0, 9, size=(len(idx), 2)), index=idx)
print(df)
df.reorder_levels([1, 'lvl1'])

# swaplevel(self, i=-2, j=-1, axis=0)
#     交换特定轴上多层索引中的级别
#     参数：
#         i, j: int or str
#             要交换的索引的级别
#     返回: DataFrame
idx = pd.MultiIndex.from_product([['A', 'B'], ['y', 'x']],
                                 names=['lvl1', 'lvl2'])
df = pd.DataFrame(rng.integers(0, 9, size=(len(idx), 2)), index=idx)
print(df)
df.swaplevel(0, 'lvl2')

# %% DataFrame方法：转置
# transpose(self, *args, copy=False)
#     转置行列
#     参数：
#         copy: bool
#             是否在转置后复制数据
#             对于混合类型或扩展类型的DataFrames，总会创建一个副本
#     返回:
#         out: <'pandas.DataFrame'>
d = np.array([[1, 9.5, 2, 0], [4, 8, 9, 0]])
col = ['col1', 'col2', 'col3', 'col4']
df1 = pd.DataFrame(data=d, columns=col)
print(df1)
df2 = df1.transpose()
df2[0][1] = 30.5
print(df2, df1, d, sep='\n')

d = np.array([[1, 9.5, 2, 0], [4, 8, 9, 0]])
df1 = pd.DataFrame(data=d, columns=col, copy=True)
print(df1)
df2 = df1.transpose()
df2[0][1] = 30.5
print(df2, df1, d, sep='\n')

d = np.array([[1, 9.5, 2, 0], [4, 8, 9, 0]])
df1 = pd.DataFrame(data=d, columns=col)
print(df1)
df2 = df1.transpose(copy=True)
df2[0][1] = 30.5
print(df2, df1, d, sep='\n')

# %% DataFrame方法：合并
# append(self, other, ignore_index=False, verify_integrity=False, sort=False)
#     将other的行追加到末尾，返回一个新对象，不在调用者中的其他列将作为新列添加
#     参数：
#         other: DataFrame or Series/dict-like object, or list of these
#             要附加的数据
#         ignore_index : bool
#             为真，沿着连接轴不使用原有索引值
#         verify_integrity : bool
#             默认不检查新连接的轴是否包含重复项
#         sort : bool
#             对列进行排序
#     返回：DataFrame
df1 = pd.DataFrame([[1, 2], [3, 4]], columns=list('BA'))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('BA'))

df1.append(df2)
df1.append(df2, ignore_index=True)
df1.append(df2, sort=True)

# assign(self, **kwargs)
#     将新列分配给DataFrame
df = pd.DataFrame({'X': [4, 2]}, index=['A', 'B'])
print(df)

df.assign(Y=[3, 5])
df.assign(Y=[3, 5], X=[6, 8])

df.assign(Y=lambda z: z['X'] * 3 / 2 - 2)
df.assign(Y=lambda z: z['X'] * 3 / 2 - 2, Z=lambda z: z['Y'] * 3 / 2 + 3)

# join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False)
#     通过索引或者指定的列连接两个DataFrame
#     参数：
#         other: DataFrame, named Series, or list of DataFrame
#             如果传递了一个Series，则必须设置将用作列名的属性 name
#         on : str, list of str, or array-like
#             默认通过索引连接两个DataFrame，否则使用参数 on指定的列
#         how : {‘left’, ‘right’, ‘outer’, ‘inner’}
#             定义连接的方式：
#                 left: 使用self的索引/列
#                 right: 使用other的索引
#                 outer: 使用self的索引/列和other的索引的并集，并对其进行排序
#                 inner: 使用self的索引/列和other的索引的交集，并对其进行排序
#         lsuffix : str
#             重叠列中 self 列名的后缀
#         rsuffix : str
#             重叠列中 other 列名的后缀
#         sort : bool
#             按字典顺序排列数据，为False，则连接键的顺序取决于how关键字
#     返回：DataFrame
idx = pd.Index(['A', 'B', 'C', 'D'])
d1 = idx.to_numpy()
d2 = np.floor_divide(rng.integers(5, size=(3, 4)), ([2, 1, 2, 1]))
d = np.add(d1, d2.astype('str'))
print(d1, d2, d, sep='\n')
df1 = pd.DataFrame(data=d, columns=idx, copy=True)
df2 = pd.DataFrame(rng.integers(5, size=(3, 2))).add_prefix('V')
print(df1, df2, sep='\n')
df1.join(df2)

# 连接索引
df = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                   'B': ['B0', 'B1', 'B2']},
                  index=['K4', 'K1', 'K2'])
other = pd.DataFrame({'A': ['C0', 'C2', 'C3'],
                      'C': ['D0', 'D2', 'D3']},
                     index=['K4', 'K2', 'K3'])
print(df, other, sep='\n')

# 等价于 merge(df, other, left_index=True, right_index=True, how='left')
df.join(other, lsuffix='_x', rsuffix='_y')
df.join(other, lsuffix='_l', rsuffix='_r', sort=True)
df.join(other, lsuffix='_l', rsuffix='_r', how='right')
df.join(other, lsuffix='_l', rsuffix='_r', how='inner')
df.join(other, lsuffix='_l', rsuffix='_r', how='outer')

# 连接索引和列
idx = pd.MultiIndex.from_tuples([('K0', 'K0'), ('K1', 'K0'),
                                 ('K2', 'K0'), ('K2', 'K1')])
df = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                   'B': ['B0', 'B1', 'B2', 'B3'],
                   'key1': ['K0', 'K0', 'K1', 'K2'],
                   'key2': ['K0', 'K1', 'K0', 'K1']})
other = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3'],
                      'E': ['E0', 'E1', 'E2', 'E3']},
                     index=idx)
print(df, other, sep='\n')

df.join(other, on=['key1', 'key2'])

# 将单个索引连接到多个索引
index = pd.MultiIndex.from_tuples([('K0', 'Y0'), ('K1', 'Y1'),
                                   ('K2', 'Y2'), ('K2', 'Y3')],
                                  names=['key', 'Y'])
df = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                   'B': ['B0', 'B1', 'B2']},
                  index=pd.Index(['K0', 'K1', 'K2'], name='key'))
other = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']},
                     index=index)
print(df, other, sep='\n')

df.join(other)
df.join(other, how='inner')

# merge(self, right, how='inner', on=None, left_on=None, right_on=None,
#       left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'),
#       copy=True, indicator=False, validate=None)
#     使用数据库风格的连接合并DataFrame(对应的顶级函数：pandas.merge)
#     参数：
#         right: DataFrame or named Series
#             如果传递了一个Series，则必须设置将用作列名的属性 name
#         on : str, list of str, or array-like
#             默认以重叠的列名当做连接键
#         how : {‘left’, ‘right’, ‘outer’, ‘inner’}
#             定义合并的方式：
#                 left: 使用 self 的连接键
#                 right: 使用 right 的连接键
#                 outer: 使用 self 和 right 连接键的并集
#                 inner: 使用 self 和 right 连接键的交集
#         left_on : abel or list, or array-like
#             self 的连接键
#         right_on : label or list, or array-like
#             right 的连接键
#         left_index : bool
#             使用 self 的索引作为连接键
#         right_index : bool
#             使用 right 的索引作为连接键
#         sort : bool
#             按字典顺序排列数据，为False，则连接键的顺序取决于how关键字
#         suffixes : tuple of (str, str)
#             应用于重叠列名的后缀，若要对重叠列引发异常，使用(False, False)
#         indicator :bool or str
#             为真，向输出中添加一个名为“_merge”的列，其中包含关于每行源的信息
#     返回：DataFrame
# 左右DataFrame列名完全不同
df1 = pd.DataFrame({'key1': ['K0', 'K4', 'K1', 'K2'],
                    'key2': ['K0', 'K1', 'K0', 'K1'],
                    'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3']})
df2 = pd.DataFrame({'key3': ['K0', 'K1', 'K3', 'K2'],
                    'key4': ['K0', 'K0', 'K0', 'K0'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']})
print(df1, df2, sep='\n')

df1.merge(df2, left_on='key1', right_on='key3')
df1.merge(df2, left_on='key1', right_on='key3', how='outer')
df1.merge(df2, left_on='key1', right_on='key3', how='left')
df1.merge(df2, left_on='key1', right_on='key3', how='right')
df1.merge(df2, left_on='key1', right_on='key3', how='outer', indicator=True)
df1.merge(df2, left_on='key1', right_on='key3', how='outer',
          indicator='indicator_column')

# 左右DataFrame列名部分相同
df1 = pd.DataFrame({'key1': ['K0', 'K4', 'K1', 'K2'],
                    'key2': ['K0', 'K1', 'K0', 'K1'],
                    'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3']})
df2 = pd.DataFrame({'key1': ['K0', 'K1', 'K3', 'K2'],
                    'key2': ['K0', 'K0', 'K0', 'K0'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']})
print(df1, df2, sep='\n')
df1.merge(df2)
# 使用validate参数自动检查他们的合并键中是否存在意外的重复项
# df1.merge(df2, on='key2', validate="one_to_one")
df1.merge(df2, on='key2', validate="many_to_many")

# 左右DataFrame按索引连接
df1 = pd.DataFrame({'col1': ['D0', 'B0', 'A0', 'C0'],
                    'col2': ['D1', 'B1', 'A1', 'C1']},
                   index=list('DBAC'))
df2 = pd.DataFrame({'col1': ['F0', 'C0', 'A0', 'E0'],
                    'col3': ['F1', 'C1', 'A1', 'E1']},
                   index=list('FCAE'))
print(df1, df2, sep='\n')

df1.merge(df2, left_index=True, right_index=True)
df1.merge(df2, left_index=True, right_index=True, how='outer')

# 左右DataFrame按列-索引连接
index = pd.MultiIndex.from_tuples([('K0', 'K0'), ('K1', 'K0'),
                                   ('K2', 'K0'), ('K2', 'K1')])
df = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                   'B': ['B0', 'B1', 'B2', 'B3'],
                   'key1': ['K0', 'K0', 'K1', 'K2'],
                   'key2': ['K0', 'K1', 'K0', 'K1']})
other = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3'],
                      'E': ['E0', 'E1', 'E2', 'E3']},
                     index=index)
print(df, other, sep='\n')

df.merge(other, left_on=['key1', 'key2'], right_index=True, how='left')

# 将单个索引连接到多个索引
index = pd.MultiIndex.from_tuples([('K0', 'Y0'), ('K1', 'Y1'),
                                   ('K2', 'Y2'), ('K2', 'Y3')],
                                  names=['key', 'Y'])
df = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                   'B': ['B0', 'B1', 'B2']},
                  index=pd.Index(['K0', 'K1', 'K2'], name='key'))
other = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']},
                     index=index)
print(df, other, sep='\n')

result = df.reset_index().merge(other.reset_index(), on=['key'], how='inner')
print(result)
result.set_index(['key', 'Y'])

# %% DataFrame方法：时间序列相关
# asfreq(self, freq, method=None, how=None, normalize=False, fill_value=None)
#     将时间索引按指定的频率转换
#     参数：
#         freq: DateOffset or str
#             频率
#         method: {‘backfill’/‘bfill’, ‘pad’/‘ffill’, None}
#             填充方法
#             backfill / bfill: 沿轴将有效值向前一个缺失值填充
#             pad / ffill: 沿轴将有效值向后一个缺失值填充
#         how: {‘start’, ‘end’}
#             仅适用于 PeriodIndex
#         fill_value: scalar
#             不会填充已经存在的NaNs
#     返回:
#         out: <'pandas.DataFrame'>
d = [0.0, None, 2.0, 3.0]
idx = pd.date_range('1/1/2020', periods=4, freq='T')
df = pd.DataFrame(data=d, index=idx, columns=['col'])
print(df)
df.asfreq(freq='30S')
df.asfreq(freq='30S', fill_value=9.0)
df.asfreq(freq='30S', method='bfill')
df.asfreq(freq='30S', method='ffill')

# shift(self, periods=1, freq=None, axis=0, fill_value=None)
#     移动数据DataFrame数据
#     参数：
#         periods: int
#             要移动的周期数
#         freq: DateOffset, tseries.offsets, timedelta, or str
#             行索引非时间序列
#                 freq不可指定，根据axis的设置，移动数据，行列索引不变
#             行索引为时间序列（axis参数无效）：
#                 freq未指定则索引通过其自身的freq属性进行移位
#                 freq被指定，行索引滚动periods*freq偏移量，列索引及数据不会移动
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             目标轴
#         fill_value: object
#             用于填充新引入的缺失值，
#             对于数值型数据，使用 np.nan
#             datetime、timedelta或period数据等，使用 NaT
#             扩展dtypes使用 self.dtype.na_value
#     返回:
#         out: <'pandas.DataFrame'>
df = pd.DataFrame({'Col1': [10, 20, 15, 30, 45], 'Col2': [13, 23, 18, 33, 48],
                   'Col3': [17, 27, 22, 37, 52]}, index=list('ABCDE'))
df.shift(periods=3)
df.shift(periods=-3)
df.shift(periods=1, axis='columns')
df.shift(periods=3, fill_value=0)

rng = np.random.default_rng()
idx = pd.date_range('20200101', periods=6)
df = pd.DataFrame(rng.standard_normal((6, 4)), index=idx, columns=list('ABCD'))
df.shift(5, freq='D')
df.shift(14, freq='2D')
df.shift(5)

# tshift(self, periods=1, freq=None, axis=0)
# 移动时间索引
#     参数：
#         periods: int
#             要移动的周期数
#         freq: DateOffset, tseries.offsets, timedelta, or str
#             行索引为时间序列（axis参数无效）：
#                 freq未指定则索引通过其自身的freq属性进行移位
#                 freq被指定，行索引滚动periods*freq偏移量，列索引及数据不会移动
#     返回:
#         out: <'pandas.DataFrame'>
rng = np.random.default_rng()
idx = pd.date_range('20200101', periods=6)
df = pd.DataFrame(rng.standard_normal((6, 4)), index=idx, columns=list('ABCD'))
df.tshift(5, freq='D')
df.tshift(14, freq='2D')

# to_period(self, freq=None, axis=0, copy=True)
#     DataFrame 中的 DatetimeIndex 转换为具有所需频率的 PeriodIndex,(如果没有传递，
#     则从索引中推断)
#     返回:
#         out: <'pandas.DataFrame'>
rng = np.random.default_rng()
idx = pd.date_range('1/1/2020', periods=3, freq='MS')
col = pd.date_range('1/1/2021', periods=3, freq='M')
df = pd.DataFrame(rng.integers(7, size=(len(idx), len(col))),
                  index=idx, columns=col)
print(df)
df.to_period(freq='M')
df.to_period(axis=1)

# to_timestamp(self, freq=None, how='start', axis=0, copy=True)
#     DataFrame 中的 PeriodIndex 转换为 timestamps(如果没有传递， 则从索引中推断)
#     参数：
#         how: {‘s’, ‘e’, ‘start’, ‘end’}
#             默认为周期的开始
#     返回:
#         out: <'pandas.DataFrame'>
rng = np.random.default_rng()
idx = pd.period_range('1/1/2020', periods=3, freq='M')
col = pd.period_range('1/1/2021', periods=3, freq='M')
df = pd.DataFrame(rng.integers(7, size=(len(idx), len(col))),
                  index=idx, columns=col)
print(df)
df.to_timestamp(axis=1)
df.to_timestamp(how='end')

# %% DataFrame方法：二元运算符函数
# sub(self, other, axis='columns', level=None, fill_value=None)
#     等价于 - (dataframe - other)
#     参数：
#         other: scalar, sequence, Series, or DataFrame
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             默认按列进行比较
#         level: int or label
#             跨级别广播，匹配传递的多层索引级别上的索引值
#         fill_value: float or None
#             在计算之前，用这个值填充现有的缺失值(NaN)和DataFrame成功对齐所需的
#             任何新元素，如果两个对应的DataFrame位置中的数据都缺失，结果也将丢失
#         返回：
#             out: <class 'pandas.DataFrame'>
# add(self, other, axis='columns', level=None, fill_value=None) / radd
#     等价于 +
# mul(self, other, axis='columns', level=None, fill_value=None) / rrmul
#     等价于 *
# div(self, other, axis='columns', level=None, fill_value=None) / rdiv
#     等价于 /
# truediv(self, other, axis='columns', level=None, fill_value=None) / rtruediv
#     等价于 /
# floordiv(self, other, axis='columns', level=None, fill_value=None) /rfloordiv
#     等价于 //
# mod(self, other, axis='columns', level=None, fill_value=None) / rmod
#     等价于 %
# pow(self, other, axis='columns', level=None, fill_value=None) / rpow
#     等价于 **
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [30, 20, 10]}, index=list('abc'))
df.sub(1)
df.sub([1, 2], axis=1)
df.sub([1, 2, 3], axis=0)

df.sub(pd.Series([1, 2], index=['col1', 'col2']), axis=1)
df.sub(pd.Series([1, 2, 3], index=list('abc')), axis=0)

other = pd.DataFrame({'col1': [1, 2, 3]}, index=list('abc'))
df.sub(other)
df.sub(other, fill_value=10)

# eq(self, other, axis='columns', level=None)
#     等价于 ==
#     参数：
#         other: scalar, sequence, Series, or DataFrame
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             默认按列进行比较
#         level: int or label
#             跨级别广播，匹配传递的多层索引级别上的索引值
#         返回：
#             out: <class 'pandas.DataFrame'>
# ne(self, other, axis='columns', level=None)
#     等价于 =!
# le(self, other, axis='columns', level=None)
#     等价于 <=
# lt(self, other, axis='columns', level=None)
#     等价于 <
# ge(self, other, axis='columns', level=None)
#     等价于 >=
# gt(self, other, axis='columns', level=None)
#     等价于 >
df = pd.DataFrame({'col1': [250, 150, 100], 'col2': [100, 250, 300]},
                  index=['A', 'B', 'C'])
x = df.eq(100)
print(x, df[x], sep='\n')

x = df.eq(pd.Series([100, 250], index=["col1", "col2"]), axis=1)
print(x, df[x], sep='\n')
x = df.eq(pd.Series([100, 250], index=["A", "D"]), axis=0)
print(x, df[x], sep='\n')

x = df.eq([250, 100], axis=1)
print(x, df[x], sep='\n')
x = df.eq([250, 250, 100], axis=0)
print(x, df[x], sep='\n')

other = pd.DataFrame({'col2': [300, 250, 100, 150]}, index=list('ABCD'))
x = df.eq(other, axis=1)
print(x, df[x], sep='\n')

# isin(self, values)
#     检测DataFrame中每个元素是否与传递的值中的元素完全匹配
#     参数：
#         value: siterable, Series, DataFrame or dict
#     返回：DataFrame
index = pd.MultiIndex.from_product([['A', 'B'], ['row1', 'row2', 'row3']])
df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6],
                   'col2': ['a', 'b', 'd', 'e', 'd', 'a'],
                   'col3': ['a', 'c', 'a', 'b', 'd', 'b']},
                  index=index)
print(df)

df.isin(['a', 'b', 1, 3])
print(df[df.isin(['a', 'b', 1, 3])])

df.isin({'col2': ['a', 'b'], 'col1': [1, 3]})
print(df[df.isin({'col2': ['a', 'b'], 'col1': [1, 3]})])

# combine(self, other, func, fill_value=None, overwrite=True)
#     与另一个DataFrame的按列合并，升序排列
#     参数：
#         func: function
#             用于逐列合并两个dataframes列
#         fill_value: scalar value
#             将列传递给合并函数之前填充NaNs的值
#         overwrite: bool
#             self中不存在于other中的列默认填充NaN
#     返回：DataFrame
# combine_first(self, other)
#     使用来自other的非na值进行填充self的空值
# update(self, other, join='left', overwrite=True, filter_func=None,
#        errors='ignore')
#     使用来自other的非na值进行就地修改self
#     参数：
#         overwrite: bool
#             默认使用来自other的非na值进行就地修改self，设置为False,则只更新
#             原始DataFrame中的NA值
#         func: function
#             对于应该更新的值，返回True
#         errors: {‘raise’, ‘ignore’}
#             self和other数据在同一位置包含非na数据时默认忽略ValueError错误
df1 = pd.DataFrame([[1, 5, 2], [np.nan, np.nan, 3],
                    [3, 2, 2]],
                   columns=list('ABD'), index=['row1', 'row2', 'row3'])
df2 = pd.DataFrame([[2, 2, np.nan], [5, 1, 4], [2, 1, np.nan]],
                   columns=list('ABC'), index=['row2', 'row3', 'row4'])
print(df1, df2, sep='\n')

df1.combine(df2, func=lambda s1, s2: s1 if s1.sum() < s2.sum() else s2)
df1.combine(df2, func=np.minimum)
df1.combine(df2, func=np.fmin)
df1.fillna(3).combine(df2.fillna(3), func=np.fmin)
df1.combine(df2, func=np.fmin, fill_value=3, overwrite=False)
df1.combine(df2, func=np.fmin, fill_value=3)

print(df1, df2, sep='\n')
df1.combine_first(df2)

print(df1, df2, sep='\n')
df3 = df1.copy()
df3.update(df2)
print(df3)

print(df1, df2, sep='\n')
df3 = df1.copy()
df3.update(df2, overwrite=False)
print(df3)

# %% DataFrame方法：函数应用
# apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds)
#     沿着数据流的轴应用一个函数,执行任何类型的操作
#     参数：
#         func: function
#             函数(传递给函数的对象是 Series)
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             应用该函数的轴
#         raw: bool
#             False :将每一行或每一列作为一个 Series 传递给函数
#             True : 函数将接收ndarray对象（如果是NumPy聚合函数，将获得更好的性能）
#         result_type: {‘expand’, ‘reduce’, ‘broadcast’, None}
#                 ‘expand’ : 类似列表的结果将被扩展到Dataframe的列
#                 ‘reduce’ : 类似列表的结果将优先返回一个 Series
#                 ‘broadcast’ : 广播到原始DataFrame形状，原索引和列都将被保留
#         args
#             传递给func的位置参数
#         **kwds
#             传递给func的关键字参数
#     返回: Series or DataFrame
#         func输出为标量：
#             只在 result_type: ‘broadcast’ 时返回 <class 'pandas.DataFrame'>
#         func输出为 list-like(包括array) ：
#             只在  axis=1, raw=False, result_type= None / ‘reduce’
#             时返回 <class 'pandas.Series'>
#         func输出为 Series ：
#             返回 <class 'pandas.DataFrame'>
df = pd.DataFrame(np.arange(1, 7).reshape(3, 2), columns=['A', 'B'])

# 使用numpy通用函数
df.apply(np.square)
df.apply(np.square, axis=1)

# 使用numpy聚合函数
df.apply(np.sum)
df.apply(np.sum, axis=1)

# 应用于分类数据
cat = pd.Categorical([1, 2, 3, 2])
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'd'], 'C': cat})
print(df)

df.apply(lambda row: type(row['C']), axis=1)
df.apply(lambda cl: cl.dtype, axis=0)

# applymap(self, func)
#     DataFrame的每个元素应用一个函数
#     参数：
#         func: callable
#             函数，从单个值返回单个值
#     返回
#         out: <class 'pandas.DataFrame'>
df = pd.DataFrame(np.arange(1, 7).reshape(3, 2), columns=['A', 'B'])
df.applymap(np.square)

# aggregate(self, func, axis=0, *args, **kwargs)
#     沿着指定轴应用一个或多个函数进行聚合，只执行聚合类型操作,通常使用聚合的别名agg
#     参数：
#         func: function, str, list or dict
#             函数，接受的类型为：
#                 函数，函数字符串名，函数名的列表，轴标签的字典
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             应用该函数的轴
#         *args
#             传递给func的位置参数
#         **kwargs
#             传递给func的关键字参数
#     返回：Series or DataFrame
#         <class 'pandas.Series'> : 使用单个函数调用
#         <class 'pandas.DataFrame'> : 使用多个函数或列表化调用
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]],
                  columns=['A', 'B', 'C'])
df.agg(sum)
df.agg(np.square)
df.agg('sum', axis=1)

df.agg([sum])
df.agg(['sum'], axis=1)
df.agg([sum, 'min'])
df.agg({'A': [sum], 'B': ['min', 'max']})

# transform(self, func, axis=0, *args, **kwargs)
#     沿着指定轴应用一个或多个函数进行转换,只执行转换类型操作
#     参数：
#         func: function, str, list or dict
#             函数，接受的类型为：
#                 函数，函数字符串名，函数名的列表，轴标签的字典
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             应用该函数的轴
#         *args
#             传递给func的位置参数
#         **kwargs
#             传递给func的关键字参数
#     返回：<class 'pandas.DataFrame'>
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]],
                  columns=['A', 'B', 'C'])
df.transform(lambda z: z + 1)
df.transform([np.sqrt, np.square])

# %% DataFrame方法：计算
# max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs)
#     返回目标轴的最大值
#     参数：
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             目标轴
#         level: int or level name
#             如果轴是一个多层索引，则沿着特定的级别计数，并折叠成一个 Series
#         numeric_only: bool
#             只包含浮点/整型/布尔型列。默认遍历尝试，然后只使用数字数据
#         **kwargs
#             传递给func的关键字参数
#     返回：<class 'pandas.Series'>,如果指定级别返回<class 'pandas.DataFrame'>
# min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs)
#     返回目标轴的最小值
# mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs)
#     返回目标轴的均值
d = [[3, 4, 5], [24.0, 'fly', 10],
     [8.5, 'run', np.nan], [np.nan, 'jump', 15]]
idx = pd.MultiIndex.from_tuples([('row1', 'falcon'), ('row1', 'parrot'),
                                 ('row2', 'lion'), ('row2', 'monkey')],
                                names=['level1', 'level2'])
df = pd.DataFrame(data=d, index=idx, columns=['A', 'B', 'C'], copy=True)
df.max()
df.max(level=0)
df.max(axis=1)
df.max(numeric_only=True)

# sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0,
#     **kwargs)
#     返回目标轴所含数值之和
#     参数：
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             目标轴
#         level: int or level name
#             如果轴是一个多层索引，则沿着特定的级别计数，并折叠成一个 Series
#         min_count: int
#             执行操作所需的有效值数量。如果有效值数量小于min_count，则结果为NA
#         **kwargs
#             传递给func的关键字参数
#     返回：<class 'pandas.Series'>,如果指定级别返回<class 'pandas.DataFrame'>
d = [[3, 4, 5], [24.0, 'fly', 10],
     [8.5, 'run', np.nan], [np.nan, 'jump', 15]]
idx = pd.MultiIndex.from_tuples([('row1', 'falcon'), ('row1', 'parrot'),
                                 ('row2', 'lion'), ('row2', 'monkey')],
                                names=['level1', 'level2'])
df = pd.DataFrame(data=d, index=idx, columns=['A', 'B', 'C'], copy=True)
df.sum()
df.sum(level=0)
df.sum(axis=1)
df.sum(numeric_only=True)

pd.DataFrame([np.nan]).sum(min_count=1)

# cumsum(self, axis=None, skipna=True, *args, **kwargs)
#     返回目标轴所含数值累积和
#     参数：
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             目标轴，default 0
#     返回：<class 'pandas.Series'>,如果指定级别返回<class 'pandas.DataFrame'>
# cummax(self, axis=None, skipna=True, *args, **kwargs)
#     返回目标轴所含数值累积累积最大值
# cummin(self, axis=None, skipna=True, *args, **kwargs)
#     返回目标轴所含数值累积乘积
# cumprod(self, axis=None, skipna=True, *args, **kwargs)
df = pd.DataFrame([[2.0, 1.0], [3.0, np.nan], [1.0, 0.0]], columns=list('AB'))
df.cumsum()
df.cumsum(axis=1)

# all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs)
#     检测所有元素都为真
index = pd.MultiIndex.from_product([['A', 'B'], ['row1', 'row2', 'row3']])
df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6],
                   'col2': ['a', 'b', 'd', 'e', 'd', 'a'],
                   'col3': ['a', 'c', 'a', 'b', 'd', 'b']},
                  index=index)
print(df)

df.isin(['a', 'b', 1, 3])
df.isin(['a', 'b', 1, 3]).all(axis=None)
df.isin(['a', 'b', 1, 3]).all(axis=0)
df.isin(['a', 'b', 1, 3]).all(axis=1)
print(df[df.isin(['a', 'b', 1, 3]).all(axis=1)])

# %% DataFrame方法：重建索引
# add_prefix(self, prefix)
#     为列标签添加前缀
# add_suffix(self, prefix)
#     为列标签添加后缀
pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]}).add_prefix('col_')

# rename(self, mapper=None, index=None, columns=None, axis=None, copy=True,
#        inplace=False, level=None, errors='ignore')
#     设置索引或列名称
#     参数：
#         index, columns: dict-like or function
#             目标轴
#     返回：DataFrame
# rename_axis(self, mapper=None, index=None, columns=None, axis=None,
#             copy=True, inplace=False)
#     设置索引或列轴名称
#     参数：
#         index, columns: scalar, list-like, dict-like or function
#             目标轴
#     返回：DataFrame
idx = pd.MultiIndex.from_product([['a', 'b'], ['y', 'x']],
                                 names=['lvl1', 'lvl2'])
df = pd.DataFrame(rng.integers(0, 9, size=(len(idx), 2)), index=idx)
print(df)
df.rename(columns={0: 'col1', 1: 'col2'})
print(df.rename(columns=str).columns)
df.rename(index={'A': 'row1', 'B': 'row2', 'y': 'x', 'x': 'y'})
df.rename(index=str.upper)

df.rename_axis(index={'lvl1': 'class'})
df.rename_axis(index=['abc', 'def'])
df.rename_axis(index=str.upper)
df.rename_axis(columns=['Cols'])

# reindex(self, labels=None, index=None, columns=None, axis=None, method=None,
#         copy=True, level=None, fill_value=nan, limit=None, tolerance=None)
#     重建新索引
#     参数：
#         labels: array-like
#             新索引/列
#         index, columns: array-like
#             新索引/列
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             目标轴
#         method: {None, ‘backfill’/’bfill’, ‘pad’/’ffill’, ‘nearest’}
#             填充方法，只适用于具有单调递增/递减索引的 DataFrames，并且原始NaN值
#             不会被任何值传播方案填充
#             None (default): 不填充
#             backfill / bfill: 沿轴将有效值向前的缺失值填充
#             pad / ffill: 沿轴将有效值向后的缺失值填充
#             nearest: 沿轴将最接近的有效值向缺失值填充
#         level: int or name
#             跨级别广播，匹配传递的多层索引级别上的索引值
#         fill_value: scalar(包括字符串)
#             填充值
#         limit: int
#             向前或向后填充的连续元素的最大数目
#     返回：DataFrame
# 基本索引
df = pd.DataFrame(rng.integers(0, 9, size=(5, 2)),
                  index=list('ABCDE'), columns=['X', 'Y'], copy=True)
print(df)

new_idx = pd.Index(['C', 'F', 'G', 'D', 'B'])
new_col = pd.Index(['X', 'Z'])
df.reindex(index=new_idx)
df.reindex(index=new_idx, fill_value=0)
df.reindex(columns=new_col)
df.reindex(columns=new_col, fill_value=0)

# 时间索引
date_idx = pd.date_range('1/1/2020', periods=6, freq='D')
df = pd.DataFrame({"prices": [100, 101, np.nan, 100, 89, 88]}, index=date_idx)
date_idx2 = pd.date_range('12/29/2019', periods=10, freq='D')

df.reindex(date_idx2)
df.reindex(date_idx2, method='bfill')
df.reindex(date_idx2, method='ffill')
df.reindex(date_idx2, method='nearest')

# 多层索引
idx = pd.MultiIndex.from_product([['A', 'B'], ['y', 'x']],
                                 names=['lvl1', 'lvl2'])
df = pd.DataFrame(rng.integers(0, 9, size=(len(idx), 2)), index=idx)
print(df)
df.sum(level=0)
df.sum(level=0).reindex(idx, level=0)

# align(self, other, join='outer', axis=None, level=None, copy=True,
#       fill_value=None, method=None, limit=None, fill_axis=0,
#       broadcast_axis=None)
#     将轴上的两个对象与指定的联接方法对齐
#     返回：(left, right)(DataFrame, type of other)
idx = pd.MultiIndex.from_product([['A', 'B'], ['y', 'x']],
                                 names=['lvl1', 'lvl2'])
df = pd.DataFrame(rng.integers(0, 9, size=(len(idx), 2)), index=idx)
print(df)
x1, x2 = df.align(df.sum(level=0), level=0)
print(x1, x2, sep='\n')

# reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None)
#     返回与其他对象具有匹配索引的对象
#     等同调用 reindex(index=other.index, columns=other.columns,...)
#     参数：
#         other: Object of the same data type
#             新索引/列
#         method: {None, ‘backfill’/’bfill’, ‘pad’/’ffill’, ‘nearest’}
#             填充方法，只适用于具有单调递增/递减索引的 DataFrames，并且原始NaN值
#             不会被任何值传播方案填充
#             None (default): 不填充
#             backfill / bfill: 沿轴将有效值向前的缺失值填充
#             pad / ffill: 沿轴将有效值向后的缺失值填充
#             nearest: 沿轴将最接近的有效值向缺失值填充
#         limit: int
#             向前或向后填充的连续元素的最大数目
#     返回：<class 'pandas.DataFrame'>
d1 = [[24.3, 75.7, 'high'], [31, 87.8, 'high'],
      [22, 71.6, 'medium'], [35, 95, 'medium']]
idx1 = pd.date_range(start='2020-02-12', end='2020-02-15', freq='D')
df1 = pd.DataFrame(data=d1, index=idx1, columns=['A', 'B', 'C'], copy=True)

d2 = [[28, 'low'], [35.1, 'medium']]
idx2 = pd.DatetimeIndex(['2020-02-12', '2020-02-15'])
df2 = pd.DataFrame(data=d2, index=idx2, columns=['A', 'C'])

df2.reindex_like(df1)
df2.reindex_like(df1, method='backfill')
df2.reindex_like(df1, method='ffill')

# set_index(self, keys, drop=True, append=False, inplace=False,
#           verify_integrity=False)
#     设置DataFrame索引
#     参数：
#         keys: label or array-like or list of labels/arrays
#             目标列
#         drop: bool
#             默认删除用作新索引的列
#         append: bool
#             默认不保留现有索引
#         inplace: bool
#             默认创建新对象而非修改数据源
#         verify_integrity: bool,
#             检查新索引中的重复项，设置为False将提高该方法的性能
#     返回：<class 'pandas.DataFrame'>
df = pd.DataFrame({'month': [1, 4, 7, 10],
                   'year': [2012, 2012, 2013, 2014],
                   'sale': [55, 40, 84, 31]})
df.set_index('month')
df.set_index(['year', 'month'])
df.set_index([pd.Index([1, 2, 2, 4]), 'year'])

ser = pd.Series(['A', 'B', 'C', 'D'])
df.set_index([ser, ser + ser])

df.set_index('month', drop=False)
df.set_index('month', append=True)

# reset_index(self, level=None, drop=False, inplace=False, col_level= 0,
#             col_fill='')
#     重置索引
#     参数：
#         level: int, str, tuple, or list
#             默认从索引中删除所有级别
#         drop: bool
#             默认将索引重置为整数索引，原索引插入到dataframe列中
#         inplace: bool
#             默认创建新对象而非修改数据源
#         col_level: int or str
#             如果列有多个级别，则确定将标签插入哪个级别,默认被插入到第一层
#         col_fill: object
#             如果列有多个级别，则确定如何命名其他级别,如果没有，则重复索引名
#     返回：<class 'pandas.DataFrame'>  or None
d = [['bird', 389.0], ['bird', 24.0], ['mammal', 80.5], ['mammal', np.nan]]
idx = pd.Index(['row1', 'row2', 'row3', 'row4'])
df = pd.DataFrame(data=d, index=idx, columns=['A', 'B'], copy=True)

df.reset_index()
df.reset_index(drop=True)

d = [[389.0, 'fly'], [24.0, 'fly'], [80.5, 'run'], [np.nan, 'jump']]
idx = pd.MultiIndex.from_tuples([('bird', 'falcon'), ('bird', 'parrot'),
                                 ('mammal', 'lion'), ('mammal', 'monkey')],
                                names=['class', 'name'])
col = pd.MultiIndex.from_tuples([('speed', 'max'), ('species', 'type')])
df = pd.DataFrame(data=d, index=idx, columns=col, copy=True)

df.reset_index(level='class')
df.reset_index(level='class', col_level=1)
df.reset_index(level='class', col_level=1, col_fill='A')
df.reset_index(level='class', drop=True)

# %% DataFrame方法：描述性统计
# count(self, axis=0, level=None, numeric_only=False)
#     返回每一列或每一行的非na /null值数量
#     参数：
#         axis: {0 or ‘index’, 1 or ‘columns’}
#             目标轴
#         level: int or str
#             如果轴是一个多层索引，则沿着特定的级别计数，生成一个更小的 DataFrame
#         numeric_only: bool,
#             仅包含浮点型、整型或布尔型数据
#         返回：
#             out: <'pandas.Series'> or <class 'pandas.DataFrame'>
# nunique(self, axis=0, dropna=True)
#     统计唯一值数目
tuples = [('row1', 'mark1'), ('row1', 'mark2'),
          ('row2', 'mark3'), ('row2', 'mark4'),
          ('row3', 'mark5'), ('row3', 'mark6')]
index = pd.MultiIndex.from_tuples(tuples, names=('level1', 'level2'))
values = [[2, 2], [0, 4], [None, 2], [np.nan, np.nan], [0, 1], [np.nan, 2]]
df = pd.DataFrame(values, columns=['col1', 'col2'], index=index, copy=True)
print(df)
df.count()
df.count(axis=1)
df.count(level=0)
df.count(level='level1')
df.count(level=1)
df.count(level='level2')

df.nunique()
df.nunique(dropna=False)
df.nunique(axis=1)

# describe(self, percentiles=None, include=None, exclude=None)
#     生成描述性统计
#     参数：
#         percentiles: list-like of numbers
#             包含在输出中的百分位数,默认为 [.25, .5, .75]
#         include: ‘all’, list-like of dtypes or None (default)
#             包含在结果中的数据类型的白列表
#                 ‘all’: 输入的所有列数据都将包含在输出中
#                 A list-like of dtypes: 将结果限制为提供的数据类型
#                 None: 结果将包括所有数字列
#         exclude: list-like of dtypes or None (default)
#             从结果中删除的数据类型的黑名单
#                 A list-like of dtypes: 从结果中排除提供的数据类型
#                 None: 结果不会排除任何类型
#         返回：
#             out: <class 'pandas.DataFrame'>
df = pd.DataFrame({'A': pd.Categorical(['a', 'c', 'c', np.nan],
                                       categories=['b', 'a', 'c', 'd']),
                   'B': [1, 2, 3, 4],
                   'C': ['a', 'b', 'c', 'a']})
print(df, df.dtypes, sep='\n')
df.describe()
df.describe(include=['category'])
df.describe(include='all')
df.describe(include=[np.number])
df.describe(exclude=[np.object])

df['A'].describe()

# select_dtypes(self, include=None, exclude=None)
#     根据列dtypes返回DataFrame列的一个子集
#     参数：
#         include: scalar or list-like
#             包含在结果中的数据类型的白列表:
#                 所有数值类型，使用 np.number or 'number'
#                 字符串，使用  object
#                 datetimes，使用 np.datetime64, 'datetime' or 'datetime64'
#                 时间增量，使用 np.timedelta64, 'timedelta' or 'timedelta64'
#                 pandas分类类型，使用 'category'
#         exclude: scalar or list-like
#             从结果中删除的数据类型的黑名单
#         返回：
#             out: <class 'pandas.DataFrame'>
df = pd.DataFrame({'A': [1, 2] * 3, 'B': ['a', 'b'] * 3, 'C': [1.0, 2.0] * 3})
print(df.dtypes)

df.select_dtypes(include='object')
df.select_dtypes(include=np.object)

df.select_dtypes(include='number')
df.select_dtypes(include=np.number)

df.select_dtypes(include='float64')
df.select_dtypes(include=np.float)
df.select_dtypes(include='int64')
df.select_dtypes(include=np.longlong)

df.select_dtypes(exclude='int64')
df.select_dtypes(exclude=['int64', 'object'])

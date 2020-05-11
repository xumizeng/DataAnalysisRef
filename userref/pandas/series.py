# -*- coding: utf-8 -*-
"""
Created on 2020/4/7 17:50 by PyCharm

@author: xumiz
"""
# %% 模块导入
import re
import pandas as pd
import numpy as np

# %% Series 构造器
#  class Series
# Series(data=None, index=None, dtype=None, name=None, copy=False,
#        fastpath=False)
#     带有轴标签(包括时间序列)的一维ndarray
#     参数：
#         data: array-like, Iterable, dict, or scalar value
#             数据源
#         index: array-like or Index (1d)
#             输出的行标签，默认为RangeIndex(0, 1, 2, …, n)
#         dtype: str, numpy.dtype, or ExtensionDtype, optional
#             强制的数据类型，默认解析器自行推断ExtensionDtype，详见附录
#         name: str
#             Series 的名称
#         copy: bool
#             从输入中复制数据
#     返回：
#         out: <'pandas.Series'> Series对象
# 设置行标签并命名
ser = pd.Series([1, 2, 3], index=['row1', 'row2', 'row3'])
print(ser)
ser = pd.Series([1, 2, 3], name='first')
print(ser)

# %% Series 属性
# array           获取 Series 中的值
#                 (out: <'pandas.core.arrays.numpy_.PandasArray'>)
ser = pd.Series([1, 2, np.nan])
print(ser.array)

# is_unique                   对象中的值唯一，返回True (out: <'bool'>)
# is_monotonic                对象中的值单调递增,返回True (out: <'bool'>)
# is_monotonic_increasing     对象中的值单调递增，返回True (out: <'bool'>)
# is_monotonic_decreasing     对象中的值单调递减，返回True (out: <'bool'>)
ser = pd.Series([1, 2, 8, 9])
print(ser.is_unique, ser.is_monotonic, ser.is_monotonic_increasing,
      ser.is_monotonic_decreasing)

# %% Series方法：转换
# to_numpy(self, dtype=None, copy=False, na_value=no_default,**kwargs)
#     获取 ndarray 类型数据源
#     参数：
#         dtype: str or numpy.dtype
#             数据类型
#         copy: bool
#             创建一个副本
#         na_value: Any
#             用于丢失值的值，默认值取决于dtype和数组的类型
#     返回：
#         out: <'numpy.ndarray'> ndarray数组
ser = pd.Series([1, 2, None])
ser.to_numpy(copy=True)

# copy=Fasle 时对数组操作可能改变数据源
ser = pd.Series([1, 2, None])
ser.to_numpy()[2] = 10
print(ser)

# 设置缺失值
ser = pd.Series([1, 2, None])
ser.to_numpy(na_value=23)
print(ser)

# %% Series方法：函数应用
# map(self, arg, na_action=None)
#     根据输入映射Series的值
#     参数：
#         arg: function, collections.abc.Mapping subclass or Series
#             对应的映射
#         na_action: {None, ‘ignore’}
#             默认将函数应用到缺失的值
#     返回：Series
rng = np.random.default_rng()
df2 = pd.DataFrame({'a': ['one', 'one', 'two', 'three', 'two', 'one', 'six'],
                    'b': ['x', 'y', 'y', 'x', 'y', 'x', 'x'],
                    'c': rng.integers(-3, 9, 7)})
print(df2)
df2['a'].map({'one': 'a', 'two': 'b'})
df2['a'].map('Map values: {}'.format)
df2['a'].map(lambda x: x.startswith('t'))
print(df2)
print(df2[df2['a'].map(lambda x: x.startswith('t'))])

# %% Series方法：计算
# unique(self)
#     返回序列对象的唯一值
#     返回：ndarray
pd.Series(['b', 'b', 'a', 'c', 'b']).unique()

df = pd.DataFrame({'A': list('abca'), 'B': list('bccd')})
print(df)
pd.Series(df.to_numpy().ravel()).unique()

# factorize(self, sort=False, na_sentinel=-1)
#     将对象编码为枚举类型或分类变量
#     参数：
#         sort: bool, default True
#             对uniques进行排序，并对code进行洗牌来保持顺序
#         na_sentinel: int
#             缺失值标记
#     返回：(codes: ndarray, uniques: Index)
codes, uniques = pd.Series(['b', None, 'a', 'c', 'b']).factorize()
print(codes, uniques, sep='\n')

codes, uniques = pd.Series(['b', None, 'a', 'c', 'b']).factorize(sort=True)
print(codes, uniques, sep='\n')

# %% Series方法：描述性统计
# value_counts(self, normalize=False, sort=True, ascending=False, bins=None,
#              dropna=True)
#     返回一个包含唯一值计数的 Series，结果对象将按频次降序排列
#     参数：
#         normalize: bool
#             返回对象默认不包含唯一值的相对频率（值除以值的和）
#         sort: bool, default True
#             按频次排序
#         ascending: bool, default False
#             默认按降序排列
#         bins: int, optional
#             分组到半开区间中，只对数值数据有效
#         dropna: bool, default True
#             默认不包含NaN的计数
#     返回：Series
rng = np.random.default_rng()
s = pd.Series(rng.integers(7, size=10))
print(s)

s.value_counts()
s.value_counts(normalize=True)
s.value_counts(bins=3)
s.value_counts(dropna=False)

cat = pd.Categorical(['a', 'b', 'c', 'c'], categories=['c', 'a', 'b', 'd'])
s = pd.Series(cat)
print(s)
s.value_counts()

# %% Series方法：str访问器
# str.lower(self)
#     将序列/索引中的字符串转换为小写形式
#     返回：Series
# str.upper(self)
#     将序列/索引中的字符串转换为大写形式
# str.title(self)
#     将系列/索引中的字符串转换为主题字符串
# str.capitalize(self)
#     将序列/索引中的字符串首个字符转换为大写形式
# str.swapcase(self)
#     将序列/索引中的字符串大小写互换
# str.casefold(self)
#     将序列/索引中的字符串更彻底地转换为小写形式
s = pd.Series(['lower', 'CAPITALS', np.nan, 'SwApCaSe'], dtype='string')
print(s)
s.str.lower()

# str.strip(self, to_strip=None)
#     删除开头和结尾指定的字符集，默认删除空白字符
#     返回：Series
# str.lstrip(self, to_strip=None)
#     删除开头指定的字符集，默认删除空白字符
# str.lstrip(self, to_strip=None)
#     删除结尾指定的字符集，默认删除空白字符
s = pd.Series(['1. Ant.  ', '2. Bee!\n', '3. Cat?\t', np.nan], dtype='string')
print(s)
s.str.strip()
s.str.strip('123.!?\n\t ')

# str.startswith(self, pat, na=nan)
#     测试每个字符串元素的开头是否与模式匹配，不接受正则表达式
#     参数：
#         pat: str
#             字符序列
#         na: default NaN
#             填充缺失值
#     返回：Series
# str.endswith(self, pat, na=nan)
#     测试每个字符串元素的结尾是否与模式匹配，不接受正则表达式
s = pd.Series(['bat', 'Bear', 'cat', np.nan], dtype='string')
s.str.startswith('b')
s.str.endswith('t')
z = s.str.startswith('b', na=False)
print(z, s[z], sep='\n')
z = s.str.endswith('t', na=False)
print(z, s[z], sep='\n')

# str.contains(self, pat, case=True, flags=0, na=nan, regex=True)
#     检测正则表达式对象是否包含在 Series 的字符串中，依赖于re.search
#     参数：
#         pat: str
#             字符序列或正则表达式
#         case: bool, default True
#             默认区分大小写
#         flags: int, default 0 (no flags)
#             传递到re模块的标志
#         na: default NaN
#             填充缺失值
#         regex: bool, default True
#             默认假定pat是一个正则表达式，如果为False，则将pat视为文本字符串
#     返回：
#         out: <class 'pandas.Series'>
s = pd.Series(['Mouse', 'dog', 'house and parrot', '23', np.NaN],
              dtype='string')
s.str.contains('og', regex=False)
z = s.str.contains('og', regex=False, na=True)
print(z, s[z], sep='\n')

z = s.str.contains('house|dog', regex=True, na=False)
print(z, s[z], sep='\n')

z = s.str.contains('PARROT', flags=re.IGNORECASE, regex=True, na=False)
print(z, s[z], sep='\n')

z = s.str.contains('\d', regex=True, na=False)
print(z, s[z], sep='\n')

s = pd.Series(['40', '40.0', '41', '41.0', '35'], dtype='string')
z = s.str.contains('.0', regex=True)
print(z, s[z], sep='\n')

# str.match(self, pat, case=True, flags=0, na=nan)
#     确定每个字符串是否与正则表达式完全匹配，依赖于re.match
#     参数：
#         pat: str
#             字符序列或正则表达式
#         case: bool, default True
#             默认区分大小写
#         flags: int, default 0 (no flags)
#             传递到re模块的标志
#         na: default NaN
#             填充缺失值
#     返回：Series
s = pd.Series(['Mouse', 'dog', 'house and parrot', '23', np.NaN],
              dtype='string')
s.str.match('dog')
z = s.str.match('dog', na=True)
print(z, s[z], sep='\n')

z = s.str.match('house|dog', na=False)
print(z, s[z], sep='\n')

z = s.str.match('HOU', flags=re.IGNORECASE, na=False)
print(z, s[z], sep='\n')

z = s.str.match('\d', na=False)
print(z, s[z], sep='\n')

s = pd.Series(['40', '40.0', '41', '41.0', '35'], dtype='string')
z = s.str.match('.0')
print(z, s[z], sep='\n')

# str.replace(self, pat, repl, n=-1, case=None, flags=0, regex=True)
#     用其他字符串替换 Series/Index 中出现的匹配对象
#     参数：
#         pat: str or compiled regex
#             字符序列或正则表达式
#         repl: str or callable
#             替换字符串
#         n: int
#             从开始更换的数量
#         case: bool
#             默认区分大小写
#         flags: int, default 0 (no flags)
#             传递到re模块的标志
#         regex: bool
#             默认传入是一个正则表达式
#     返回：Series
s = pd.Series(['f.o', 'fuzfo', np.nan], dtype='string')
print(s)
s.str.replace('f.', 'ba', regex=True)
s.str.replace('f.', 'ba', regex=False)

# repl可调用
s = pd.Series(['foo 123', 'bar baz', np.nan], dtype='string')
print(s)
s.str.findall(r'[a-z]+')
s.str.replace(r'[a-z]+', lambda m: m.group(0)[::-1])

pat = r"(?P<one>\w+) (?P<two>\w+) (?P<three>\w+)"
s = pd.Series(['One Two Three', 'Foo Bar Baz'])
s.str.findall(pat)
s.str.replace(pat, lambda m: m.group('two').swapcase())

# str.extract(self, pat, flags=0, expand=True)
#     提取正则表达式中匹配的组合作为DataFrame中的列,对于 Series 中的每个字符串，
#     仅从首次匹配中提取组，组名用于列名;无组名将使用捕获组号
#     参数：
#         pat: str
#             含有组合的正则表达式
#         flags: int, default 0 (no flags)
#             传递到re模块的标志
#         expand: bool
#             为False且正则表达式对象中只有一个组合，返回一个Series，否则返回DataFrame
#     返回：
#         out: 取决于 expand 参数
# str.extractall(self, pat, flags=0)
#     提取正则表达式中匹配的组合作为DataFrame中的列,对于 Series 中的每个字符串，
#     从所有匹配中提取组，组名用于列名;无组名将使用捕获组号
s = pd.Series(['a1a2', 'b2', 'c3'], index=["A", "B", "C"], dtype='string')
print(s)
s.str.extract(r'([ab])(\d)')
s.str.extractall(r'([ab])(\d)')
s.str.extract(r'([ab])?(\d)')

s.str.extract(r'[ab](\d)')
s.str.extract(r'[ab](\d)', expand=False)

s.str.extract(r'(?P<letter>[ab])(?P<digit>\d)')

# str.extractall(self, pat, flags=0)
#     提取正则表达式中匹配的组合作为DataFrame中的列,对于 Series 中的每个字符串，
#     从所有匹配中提取组，组名用于列名;无组名将使用捕获组号；索引是一个由 Series
#     自带索引以及被命名为“match”的匹配索引组成的多层索引
#     参数：
#         pat: str
#             含有组合的正则表达式
#         flags: int, default 0 (no flags)
#             传递到re模块的标志
#     返回：DataFrame
s = pd.Series(['a1a2', 'b2', 'c3'], index=["A", "B", "C"], dtype='string')
s.str.extractall(r'([ab])(\d)')
s.str.extractall(r'([ab])?(\d)')
s.str.extractall(r'[ab](\d)')
s.str.extractall(r'(?P<letter>[ab])(?P<digit>\d)')

# str.findall(self, pat, flags=0, **kwargs)
#     找出 Series/Index 中出现的所有非重叠匹配对象
#     参数：
#         pat: str
#             含有组合的正则表达式
#         flags: int, default 0 (no flags)
#             传递到re模块的标志
#     返回：Series/Index
s = pd.Series(['Lion', 'Monkey', 'Rabbit'], dtype='string')
s.str.findall('MONKEY')
s.str.findall('MONKEY', flags=re.IGNORECASE)
s.str.findall('on')
s.str.findall('on$')
s.str.findall('b')

s1 = pd.Series(['A(w)-10', 'B(w)-10', 'C(w)-20', 'A(w)-20'], dtype='string')
s2 = s1.str.findall(r'[A-B]\(.*\)')
print(s2, s2.to_numpy(), sep='\n')
se = set([match[0] for match in s2.to_numpy() if match != []])
print(se, sorted(se), sep='\n')

# str.split(self, pat=None, n=-1, expand=False)
#     从左边开始在给定的pat周围拆分字符串
#         pat: str
#             字符串或正则表达式，没有指定，则在空格上分割
#         n: int
#             限制输出中分割的数量，None、0和-1将被解释为返回所有分割
#         expand: bool
#             拆分后的字符串默认不展开为单独的列
#     返回：Series, Index, DataFrame or MultiIndex
# str.rsplit(self, pat=None, n=-1, expand=False)
#     从右边开始在给定的分隔符周围拆分字符串
s = pd.Series(['a b c', 'c_d_e', np.nan, 'f/g/h'], dtype="string")
print(s)

s.str.split()
s.str.split(expand=True)
s.str.split(pat="/")
s.str.split(pat="/", n=1, expand=True)
s.str.split(r' |_|/')
s.str.split(r' |_|/', expand=True)
s.str.split(r' |_|/').get(1)

s = pd.Series(["1+1=2"], dtype='string')
print(s)
s.str.split(r"\+|=", expand=True)

# str.join(self, sep)
#     用分隔符连接列表中的元素，如果列表中存在非字符串对象，结果为NaN
#     返回：Series/Index: object
s = pd.Series([['lion', 'elephant', 'zebra'],
               [1.1, 2.2, 3.3],
               ['cat', np.nan, 'dog'],
               ['cow', 4.5, 'goat'],
               ['duck', ['swan', 'fish'], 'guppy']])
print(s)
s.str.join('-')

# str.cat(self, others=None, sep=None, na_rep=None, join='left')
#     如果指定了others，则连接两者相同索引对应的元素。如果未传递others，则连接
#     Series/Index 中的所有元素
#     参数：
#         sep: str
#             默认使用空字符串
#         na_rep: str or None
#             缺失值表示，未指定时如果others=None，则舍弃缺失值
#         join: {‘left’, ‘right’, ‘outer’, ‘inner’}
#             定义连接的方式：
#                 left: 使用self的索引/列
#                 right: 使用other的索引
#                 outer: 使用self的索引/列和other的索引的并集，并对其进行排序
#                 inner: 使用self的索引/列和other的索引的交集，并对其进行排序
#     返回：str, Series/Index
s = pd.Series(['a', 'b', np.nan, 'd'], dtype='string')
print(s)
s.str.cat()
s.str.cat(sep=' ')
s.str.cat(na_rep='?')

other = pd.DataFrame([['d', 'a'], ['a', 'b'], ['e', 'c'], ['c', 'd']],
                     index=[3, 0, 4, 2], dtype='string')
print(s, other, sep='\n')
s.str.cat(other, sep=',')
s.str.cat(other, na_rep='-')
s.str.cat(other, join='outer', na_rep='-')
s.str.cat(other, join='inner', na_rep='-')
s.str.cat(other, join='right', na_rep='-')
s.str.cat(other.to_numpy(), na_rep='-')

# str.get(self, i)
#     从每部分的指定位置提取元素
s = pd.Series(["String", (1, 2, 3), ["a", "b", "c"], 123, -456,
               {1: "Hello", "2": "World"}])
print(s)
print(s.str[1])
s.str.get(1)
s.str.get(-1)

# str.get_dummies(self, sep='|')
#     通过sep分割Series 的每个字符串,返回一个虚拟指标化的 DataFrame
s = pd.Series(['a|b', np.nan, 'a|c'], dtype='string')
print(s)
s.str.get_dummies()

# str.count(self, pat, flags=0, **kwargs)
#     计算正则表达式匹配对象在 Series/Index 的每个字符串中出现的次数
#     参数：
#         pat: str
#             正则表达式
#         flags: int, default 0 (no flags)
#             传递到re模块的标志
#     返回：Series
s = pd.Series(['$', None, 'Aab$', '$$ca', 'C$B$', 'cat'], dtype='string')
print(s)
s.str.count('a')
s.str.count('\$')
s.dropna().str.count("a")

# str.isalpha(self)
#     检查每个字符串中的所有字符是否均为字母
#     返回：
#         out: <class 'pandas.Series'>
# str.isnumeric(self)
#     检查每个字符串中的所有字符是否均为数字
# str.isalnum(self)
#     检查每个字符串中的所有字符是否均为字母数字
# str.isdigit(self)
#     检查每个字符串中的所有字符是否均为数字
# str.isdecimal(self)
#     检查每个字符串中的所有字符是否均为小数
# str.isspace(self)
#     检查每个字符串中的所有字符是否均为空白
# str.islower(self)
#     检查每个字符串中的所有字符是否均为小写字母
# str.isupper(self)
#     检查每个字符串中的所有字符是否均为大写字母
# str.istitle(self)
#     检查每个字符串中的所有字符是否均为主题字符
s = pd.Series(['one', 'one1', '1', ''], dtype='string')
s.str.isalpha()

# %% Series方法：Categorical访问器
# Series.cat 属性:
#     categories    分类,分配到类别是一个 inplace 操作
#     ordered       类别是否具有有序关系
#     codes         返回标签化Series
df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6],
                   'col2': ['a', 'b', 'b', 'a', 'a', 'e']})
df['col2'] = df['col2'].astype('category')
print(df)

print(df['col2'].cat.categories,
      df['col2'].cat.ordered,
      df['col2'].cat.codes,
      sep='\n')

df['col2'].cat.categories = ['high', 'medium', 'low']
print(df)

# Series.cat 方法
# set_categories(new_categories, ordered=False, rename=False, inplace=False)
#     将类别设置为指定值
#     参数：
#         new_categories: Index-like
#             按新顺序分类
#         ordered: bool
#             是否被视为有序范畴, 默认不改变排序
#         rename: bool
#             默认new_categories被视为重新排序的类别,new_categories可以包含新的
#             类别(这将导致未使用的类别)或删除旧的类别(这将导致设置为NaN的值)
#              rename=True,new_categories被视为旧类别的重命名
#     返回：Categorical
# remove_unused_categories()
#     删除不使用的类别
#     返回：Categorical
df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6],
                   'col2': ['a', 'b', 'b', 'a', 'a', 'e']})
df['col2'] = df['col2'].astype('category')
print(df, df['col2'], sep='\n')

df['col2'].cat.set_categories(['a', 'low', 'medium', 'b'], inplace=True)
print(df, df['col2'], sep='\n')

df['col2'].cat.set_categories(['x', 'd', 'f', 'y'], rename=True, inplace=True)
print(df, df['col2'], sep='\n')

# 排序是按类别中的顺序进行，而非词法顺序
df.sort_values(by='col2')

df['col2'].cat.set_categories(['y', 'f', 'd', 'x'], ordered=True, inplace=True)
print(df, df['col2'], sep='\n')
df.sort_values(by='col2')

df.groupby('col2').size()

df['col2'].cat.remove_unused_categories(inplace=True)
print(df['col2'])

# rename_categories(new_categories, inplace=False)
#     重命名类别
#     参数：
#         new_categories: list-like, dict-like or callable
#             按新顺序分类
#     返回：Categorical
df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6]})
df['col2'] = pd.Categorical(['a', 'b', 'b', 'a', 'a', 'e'])
print(df, df['col2'], sep='\n')

df['col2'].cat.rename_categories([1, 2, 3], inplace=True)
print(df)

df['col2'].cat.rename_categories({'a': 'A', 'c': 'C'}, inplace=True)
print(df)

df['col2'].cat.rename_categories(lambda x: x.upper(), inplace=True)
print(df)

# add_categories(new_categories, inplace=False)
#     在类别的最后添加新类别
#     参数：
#         new_categories: category or list-like of category
#             新类别
#     返回：Categorical
df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6]})
df['col2'] = pd.Categorical(['a', 'b', 'b', 'a', 'a', 'e'])
print(df, df['col2'], sep='\n')

df['col2'].cat.add_categories('c', inplace=True)
print(df['col2'])

# reorder_categories(new_categories, inplace=False)
#     按照new_categories中指定的方式重新排序类别
#     new_categories需要包含所有旧的类别，并且不包含新类别项
#     参数：
#         new_categories: Index-like
#             新类别
#         ordered: bool
#             是否被视为有序范畴, 默认不改变排序
#     返回：Categorical
df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6]})
df['col2'] = pd.Categorical(['a', 'b', 'b', 'a', 'a', 'e'])
print(df, df['col2'], sep='\n')

df['col2'].cat.reorder_categories(['e', 'a', 'b'], inplace=True, ordered=True)
print(df['col2'])

print(df.sort_values('col2'))

# remove_categories(removals, inplace=False)
#     删除指定的类别
#     返回：Categorical
df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6]})
df['col2'] = pd.Categorical(['a', 'b', 'b', 'a', 'a', 'e'])
print(df, df['col2'], sep='\n')

df['col2'].cat.remove_categories('a', inplace=True)
print(df['col2'])

# %% 附录一: Pandas 扩展数据类型
# 数据类型                 字符串别名
# DatetimeTZDtype         'datetime64[ns, <tz>]'
# CategoricalDtype        'category'
# PeriodDtype             'period[<freq>]', 'Period[<freq>]'
# SparseDtype             'Sparse', 'Sparse[int]', 'Sparse[float]'
# IntervalDtype           'interval', 'Interval', 'Interval[<numpy_dtype>]',
#                         'Interval[datetime64[ns, <tz>]]',
#                         'Interval[timedelta64[<freq>]]'
# Int64Dtype[...]           'Int8', 'Int16', 'Int32', 'Int64',
#                         'UInt8', 'UInt16', 'UInt32', 'UInt64
# StringDtype             'string'
# BooleanDtype            'boolean' 布尔

# %% 附录二：正则表达式中的特殊字符和序列：
# 常用特殊字符:
# "."      匹配除了换行的任意字符
# "^"      匹配字符串的开头
# "$"      匹配字符串尾或者换行符的前一个字符
# "*"      对它前一个正则式匹配0到任意次重复，ab* 会匹配 'a'， 'ab'， 或者 'a'
#          后面跟随任意个 'b'
# "+"      对它前一个正则式匹配1到任意次重复，ab+ 会匹配'a'后面跟随1到任意个'b'，
#          它不会匹配 'a'
# "?"      对它前一个正则式匹配0或1次重复，ab? 会匹配 'a' 或 'ab'
# *?,+?,?? 前三个特殊字符的非贪婪版本，尽量少的字符将会被匹配
# {m,n}    对正则式进行 m 到 n 次匹配，在 m 和 n 之间取尽量多
# {m,n}?   前一个修饰符的非贪婪模式，只匹配尽量少的字符次数
# "\\"     转义特殊字符，或者表示一个特殊序列
# []       用于表示一个字符集合，[amk]匹配'a'，'m'，或者'k'
# "|"      匹配 A 或者 B
# (...)    匹配括号内的任意正则表达式，并标识出组合的开始和结尾。匹配完成后，组合的
#          内容可以被获取，并可以在之后用 \number 转义序列进行再次匹配
# (?aiLmsux) 为 RE 设置  A, I, L, M, S, U, or X 标志
# (?:...)  匹配在括号内的任何正则表达式，但该分组所匹配的子字符串不能在执行匹配后
#          被获取或是之后在模式中被引用。
# (?P<name>...) 命名组合
# (?P=name)    反向引用一个命名组合
# (?#...)  注释里面的内容会被忽略
# (?=...)  Isaac(?=Asimov)匹配 'Isaac'只有在后面是 'Asimov'的时候
# (?!...)  Isaac(?!Asimov)只有后面不是'Asimov' 的时候才匹配'Isaac'
# (?<=...)  (?<=abc)def 会在 'abcdef' 中找到一个匹配
# (?<!...) (?!=abc)def 会在非 'abcdef' 中找到一个匹配
# (?(id/name)yes|no) 如果给定的 id 或 name 存在，将会尝试匹配 yes-pattern

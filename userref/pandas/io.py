# -*- coding: utf-8 -*-
"""
Created on 2020/4/3 14:04 by PyCharm

@author: xumiz
"""
# %% 模块导入
import pandas as pd
import numpy as np
from pathlib import Path
import stat
import shutil
import csv

# %% CSV 写入器
# DataFrame.to_csv 方法
# to_csv(self, path_or_buf=None, sep=',', na_rep='',float_format=None,
#        columns= None, header=True, index=True, index_label= None, mode='w',
#        encoding=None, compression='infer', quoting=None, quotechar='"',
#        line_terminator=None, chunksize=None, date_format=None,
#        doublequote=True, escapechar=None, decimal='.')
#     将对象写入逗号分隔值(csv)文件
#     参数：
#         path_or_buf: str or file handle
#             文件路径或对象，如果没有提供，则以字符串形式返回结果
#         sep: str
#             控制输出文件的字段分隔符
#         na_rep: str
#             缺失的数据表示
#         float_format: str
#             格式化浮点数的字符串
#         date_format: str
#             格式化datetime对象的字符串
#         quoting: csv模块常量，默认为 csv.QUOTE_MINIMAL
#             如果设置了float_format，浮点数将转换为字符串从而转换为csv。
#             设置csv.QUOTE_NONNUMERIC将把它们视为非数字
#         columns: sequence
#             指定要写入的列
#         header: bool or list of str
#             是否写入列名
#         index: bool
#             是否写入行名(索引)
#         index_label: str or sequence, or False
#             为索引列添加列标签
#         mode: str
#             ‘w’: 写入时，将创建一个新文件(具有相同名称的现有文件将被删除)
#             ‘a’: 将打开现有文件进行追加读写，如果该文件不存在，则创建该文件
#             ‘r+’: 将打开现有文件进行覆盖读写，文件必须已经存在
#         chunksize: int or None
#             指定每批写入的行数以提高效率，结果都是将目标数据全部写入
#         compression: str or dict
#             压缩模式
#     返回:
#         out: <class 'NoneType'> or <class 'str'>
datapath = Path('../data')
datapath.mkdir()

outpath = Path('../data/test.csv')

# 创建、写入和保存csv
d = [['A', 'B', 'C'], ['D', 'E', 'F']]
df = pd.DataFrame(data=d, columns=['col1', 'col2', 'col3'], copy=True)
df.to_csv()

# 更改分隔符
d = [['A', 'B', 'C'], ['D', 'E', 'F']]
df = pd.DataFrame(data=d, columns=['col1', 'col2', 'col3'], copy=True)
df.to_csv(outpath, index=False, sep='_')

# 缺失值表示
d = [['A', np.nan, 'C'], ['D', 'E', np.nan]]
df = pd.DataFrame(data=d, columns=['col1', 'col2', 'col3'], copy=True)
print(df)
df.to_csv(outpath, index=False)
df.to_csv(outpath, index=False, na_rep='缺失值')

# 控制浮点值的精度
d = [[1.1236, 4.3698, 6.354], [7.658, 3.2686, 3.6945]]
df = pd.DataFrame(data=d, columns=['col1', 'col2', 'col3'], copy=True)
print(df)
df.to_csv(outpath, index=False, float_format="%.2f")
df.to_csv(outpath, index=False, float_format="%.2f",
          quoting=csv.QUOTE_NONNUMERIC)

# 设置日期格式
d = [['KORD', pd.Timestamp('20190101'), 0.01],
     ['KORD', pd.Timestamp('20190102'), -0.59],
     ['KORD', pd.Timestamp('20190103'), -0.99]]
df = pd.DataFrame(data=d, copy=True)
print(df)
df.to_csv(outpath, index=False, date_format='%Y/%m/%d')

# 行/列名写入
d = [['A', 'B', 'C'], ['D', 'E', 'F']]
df = pd.DataFrame(data=d, columns=['col1', 'col2', 'col3'], copy=True)
df.to_csv(outpath, header=False)
df.to_csv(outpath, index=False)
df.to_csv(outpath, index=False, columns=['col1', 'col3'])

# 设置索引列列标签
d = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]
df = pd.DataFrame(data=d, columns=['col1', 'col2', 'col3'], copy=True)
df_screening = df.iloc[lambda x: x.index % 2 == 0]
print(df_screening)
df_screening.to_csv(outpath, index_label='Id')

# 文件写入模式
d1 = [['A', 'B', 'C'], ['D', 'E', 'F']]
d2 = [['a', 'b']]
d3 = [['a', 'b', 'c'], ['d', 'e', 'f']]
df1 = pd.DataFrame(data=d1, columns=['col0', 'col1', 'col2'], copy=True)
df2 = pd.DataFrame(data=d2, columns=['col0', 'col1'], copy=True)
df3 = pd.DataFrame(data=d3, columns=['col0', 'col1', 'col2'], copy=True)
df1.to_csv(outpath, index=False)
df2.to_csv(outpath, mode='r+', index=False)
df3.to_csv(outpath, mode='a', index=False)
outpath.unlink()

# 设置每批次写入行数
d = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]
df = pd.DataFrame(data=d, columns=['col1', 'col2', 'col3'], copy=True)
df.to_csv(outpath, index=False, chunksize=1)

# 控制浮点值的精度
d = [[136, 48, 6.354], [7.658, 3.2686, 3.6945]]
df = pd.DataFrame(data=d, columns=['col1', 'col2', 'col3'], copy=True)
print(df)
df.to_csv(outpath, index=False, float_format="%.2f")
df.to_csv(outpath, index=False, float_format="%.2f",
          quoting=csv.QUOTE_NONNUMERIC)

# 创建压缩文件
d = [['A', 'B', 'C'], ['D', 'E', 'F']]
df = pd.DataFrame(data=d, columns=['col1', 'col2', 'col3'], copy=True)
outpath = Path('../data/test.zip')
opts = dict(method='zip', archive_name='out.csv')
df.to_csv(outpath, index=False, compression=opts)


# 清空数据文件夹
def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal"""
    Path(path).chmod(stat.S_IWRITE)
    func(path)


shutil.rmtree(datapath, onerror=remove_readonly)

# %% CSV 读取器章节的数据准备
datapath = Path('../data')
datapath.mkdir()

csv_test1 = Path('../data/test1.csv')
with open(csv_test1, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['A', 'B', 'C', 'D', 'E', 'F'])
    writer.writerow([1, '2020-01-01', 1.0, 1, 'test', 1])
    writer.writerow([2, '2020-01-02', 1.0, 1, 'train', 3])
    writer.writerow([3, '2020-01-03', 1.0, 1, 'test', 2])
    writer.writerow([4, '2020-01-04', 1.0, 1, 'train', 1])

csv_test2 = Path('../data/test2.csv')
with open(csv_test2, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['a', 'b', 'c'])
    writer.writerow([4, 'apple', 'bat', ''])
    writer.writerow([8, 'orange', 'cow', ''])

csv_test3 = Path('../data/test3.csv')
with open(csv_test3, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow('')
    writer.writerow(['a', 'b', 'c'])
    writer.writerow(' ')
    writer.writerow(['# commented line'])
    writer.writerow([1, 2, 3])
    writer.writerow('')
    writer.writerow([4, 5, 6])

csv_test4 = Path('../data/test4.csv')
with open(csv_test4, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['AAA', '20180102', '9:00:00', '9:56:00', '0.0900'])
    writer.writerow(['BBB', '2019/01/03', '15:00:00', '15:56:00', '0.1500'])
    writer.writerow(['CCC', '01/04/2020', '20:00:00', '20:56:00', '-0.2000'])

csv_test5 = Path('../data/test5.csv')
with open(csv_test5, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID|level|category'])
    writer.writerow(['Patient1|123', '000|x'])
    writer.writerow(['Patient2|23', '000|y'])
    writer.writerow(['Patient3|1', '234', '018|z'])

csv_test6 = Path('../data/test6.csv')
with open(csv_test6, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['test', 'N/A', 'n/a'])
    writer.writerow(['NA', '<NA>', '#NA'])
    writer.writerow(['NULL', 5.0, 'NaN'])
    writer.writerow(['nan', 5, ''])
    writer.writerow(['Yes', 'test', 'No'])

csv_test7 = Path('../data/test7.csv')
with open(csv_test7, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Yes', 'b', 'test'])
    writer.writerow([1, 'Yes', 'NAN'])
    writer.writerow(['No', 'No', 'No'])

csv_test8 = Path('../data/test8.csv')
with open(csv_test8, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['level'])
    writer.writerow(['Patient1', 123])
    writer.writerow(['Patient2', 23])
    writer.writerow(['Patient3', 1234])

csv_test9 = Path('../data/test9.csv')
with open(csv_test9, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, quotechar="'")
    writer.writerow(['label1', 'label2', 'label3'])
    writer.writerow(['index1', '"a', 'c', 'e'])
    writer.writerow(['index2', 'b', 'd', 'f'])

csv_test10 = Path('../data/test10.csv')
with open(csv_test10, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['a', 'b', 'c~1', '2', '3~4', '5', '6'])

csv_test11 = Path('../data/test11.csv')
with open(csv_test11, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['a', ' b', ' c'])
    writer.writerow(['1', ' 2', ' 3'])
    writer.writerow(['4', ' 5', ' 6'])

csv_test12 = Path('../data/test12.csv')
with open(csv_test12, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, escapechar='\\', quoting=csv.QUOTE_NONE)
    writer.writerow(['a', 'b'])
    writer.writerow(["hello, \"Bob\", nice to see you", 5])

# %% CSV 读取器
# pandas.read_csv()
#     读取逗号分隔值(csv)文件，返回类型 DataFrame
#         位置参数
#             filepath_or_buffer: str, path object or file-like object 路径
#         关键字参数
#             sep: str, defaults to ','
#                 分隔符
#             dtype: type, default None
#                 指定数据或列的数据类型
#             converters: dict, default None
#                 用于转换列中值的函数字典
#             header: int, list of int, default ‘infer’
#                 用作DataFrame列标签的行（忽略注释/空行）
#             index_col: int, str, or False, default None
#                 用作行标签的列
#                 如果数据列比列名称多出N列，默认多出的前N列将用作DataFrame的行标签
#                 如果只多处一列，可使用index_col=False显式禁用索引列推断，并放弃
#                 最后一列
#             names: array-like, default None 要使用的列名
#                 默认行为是推断列名:如果没有传递任何名称，则该行为与header=0相同，
#                 显式传递列名，则该行为与header=None相同
#             usecols: list-like or callable, default None
#                 返回列的子集
#             skiprows: list-like or integer, default None
#                 要跳过的行（包含注释/空行在内）
#             comment: str, default None
#                 如果注释在一行的开头，则完全舍去本行；如果注释在行内，则舍去注释及
#                 以其后的数据
#             skip_blank_lines: boolean, default True
#                 如果为真，则跳过空白行
#             parse_dates: boolean or list or dict, default False
#                 将数据解析为日期:
#                     True: 解析索引
#                     [1, 2, 3]: 将第1、2、3列解析为单独的日期列
#                     [[1, 3]]: 合并第1列和第3列，并将其解析为单个日期列,解析器
#                         默认会删除组件的日期列
#                     {'foo': [1, 3]}: 合并第1列和第3列，并将其解析为单个日期列并
#                         设置列名，默认会删除组件的日期列
#             keep_date_col: boolean, default False
#                 为真保留日期合并的原始列
#             infer_datetime_format: bool, default False
#                 为真提高日期推断速度
#             dayfirst: boolean, default False
#                 形如“01/12/2011" 按DD/MM格式日期(国际和欧洲格式)解析，
#                 默认则按照 MM/DD 解析
#             thousands: str, default None
#                 千分隔符
#             error_bad_lines: boolean, default True
#                 为假，则“坏行”将删除
#             na_values: scalar, str, list or dict, default None
#                 如果指定一个数字，相应的等效值同样视为缺失值
#                 默认的缺失值：['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN',
#                 '#N/A N/A', '#N/A', 'N/A', 'n/a', 'NA', '<NA>', '#NA',
#                 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']
#             keep_default_na: boolean, default True
#                 控制解析器默认的缺失值是否生效
#             squeeze: boolean, default False
#                 解析后数据只包含一列，则返回Series
#             true_values: list, default None
#             false_values: list, default None
#                 通过将字符串识别为布尔值，转换列类型为布尔型
#             dialect: str or csv.Dialect instance, default None
#                 方言，默认情况下，使用Excel方言
#             chunksize: int, default None
#                 返回用于迭代的TextFileReader对象
#             iterator: boolean, default False
#                 返回用于迭代的TextFileReader对象
#     返回：
#         out: <class 'pandas.core.frame.DataFrame'> or TextParser
# 指定特定列数据类型
df = pd.read_csv(csv_test1)
print(df, df.dtypes, sep='\n')

df = pd.read_csv(csv_test1, dtype='category')
print(df, df.dtypes, df['E'], sep='\n')

df = pd.read_csv(csv_test1, dtype={'F': 'category'})
print(df, df.dtypes, df['F'], sep='\n')

df['F'].cat.categories = pd.to_numeric(df['F'].cat.categories)
print(df['F'])

dtype = pd.CategoricalDtype([4, 1, 2], ordered=True)
df = pd.read_csv(csv_test1, dtype={'F': dtype})
print(df, df.dtypes, df['F'], sep='\n')

df = pd.read_csv(csv_test1, converters={'F': list})
df['F'].apply(type).value_counts()
print(df)

# 过滤列和跳行
pd.read_csv(csv_test1, usecols=['B', 'E'])
pd.read_csv(csv_test1, usecols=lambda x: x not in ['B', 'E'])
pd.read_csv(csv_test1, usecols=[1, 4])

pd.read_csv(csv_test1, skiprows=2)
pd.read_csv(csv_test1, skiprows=lambda x: x % 2 != 0)

# 设置行列标签
pd.read_csv(csv_test1, index_col=0)
pd.read_csv(csv_test1, index_col=[0, 1])
pd.read_csv(csv_test2)
pd.read_csv(csv_test2, index_col=False)

pd.read_csv(csv_test2, usecols=['b', 'c'])
pd.read_csv(csv_test2, usecols=['b', 'c'], index_col=False)

pd.read_csv(csv_test1)
pd.read_csv(csv_test1, names=['F', 'E', 'D', 'C', 'B', 'A'], header=0)
pd.read_csv(csv_test1, header=None)
pd.read_csv(csv_test1, header=[1, 2])
pd.read_csv(csv_test1, header=1)

# 注释,空行
pd.read_csv(csv_test3)
pd.read_csv(csv_test3, comment='#')
pd.read_csv(csv_test3, comment='#', header=1)
pd.read_csv(csv_test3, comment='#', skiprows=lambda x: x % 2 != 0)

pd.read_csv(csv_test3, skip_blank_lines=False)
pd.read_csv(csv_test3, skip_blank_lines=False, header=1)

# 解析日期列
pd.read_csv(csv_test4)
pd.read_csv(csv_test4, header=None, index_col=1, parse_dates=True)
pd.read_csv(csv_test4, header=None, parse_dates=[1])
pd.read_csv(csv_test4, header=None, parse_dates=[[1, 2], [1, 3]])
pd.read_csv(csv_test4, header=None,
            parse_dates={'nominal': [1, 2], 'actual': [1, 3]})
pd.read_csv(csv_test4, header=None, parse_dates={'nominal': [1, 2, 3]},
            index_col=0)

pd.read_csv(csv_test4, header=None, parse_dates=[1],
            infer_datetime_format=True)
pd.read_csv(csv_test4, header=None, parse_dates=[1], dayfirst=True)

# 分隔符与千分隔符
pd.read_csv(csv_test5, sep='|')
pd.read_csv(csv_test5, sep='|', thousands=',')

# 处理“坏”行
pd.read_csv(csv_test5, error_bad_lines=False)

# 缺失值处理
pd.read_csv(csv_test6, header=None)
pd.read_csv(csv_test6, header=None, na_values=[5, 'Nope'])
pd.read_csv(csv_test6, header=None, keep_default_na=False, na_values=["NA"])

# 布尔值处理
pd.read_csv(csv_test7, true_values=['Yes'], false_values=['No'], header=None)
pd.read_csv(csv_test7, true_values=['Yes', '1'], false_values=['No'],
            header=None)
pd.read_csv(csv_test7, true_values=['Yes'], false_values=['No'], header=0)
pd.read_csv(csv_test7, true_values=['NAN'], false_values=['No'], header=0)

# 返回 Series
pd.read_csv(csv_test8)
pd.read_csv(csv_test8, squeeze=True)

# 方言
dia = csv.excel()
dia.quoting = csv.QUOTE_NONE
pd.read_csv(csv_test9, dialect=dia)
pd.read_csv(csv_test9, quoting=csv.QUOTE_NONE)

pd.read_csv(csv_test10)
pd.read_csv(csv_test10, lineterminator='~')

pd.read_csv(csv_test11)
pd.read_csv(csv_test11, skipinitialspace=True)

pd.read_csv(csv_test12, escapechar='\\')

# 块遍历文件
pd.read_csv(csv_test1)
reader = pd.read_csv(csv_test1, chunksize=2)
for chunk in reader:
    print(chunk)
reader = pd.read_csv(csv_test1, iterator=True)
reader.get_chunk(3)
reader.close()

shutil.rmtree(datapath, onerror=remove_readonly)

# %% Excel 读取器章节的数据准备
# DataFrame.to_excel 方法
# to_excel(self, excel_writer, sheet_name='Sheet1', na_rep='',
#          float_format=None, columns=None, header=True, index=True,
#          index_label=None, startrow=0, startcol=0, engine=None,
#          merge_cells=True, encoding=None, inf_rep='inf', verbose=True,
#          freeze_panes=None)
#     将对象写入Excel工作表
#     使用已存在的文件名创建ExcelWriter对象将导致已存在文件的内容被擦除
#     参数：
#         excel_writer: str or ExcelWriter object
#             文件路径或现有的ExcelWriter
#         sheet_name: str
#             工作表名称
#         na_rep: str
#             缺失的数据表示
#         inf_rep: str
#             无穷大的表示
#         float_format: str
#             格式化浮点数的字符串
#         columns: sequence or list of str
#             指定要写入的列
#         header: bool or list of str
#             是否写入列名
#         index: bool
#             是否写入行名(索引)
#         index_label: str or sequence, or False
#             为索引列添加列标签
#         compression: str or dict
#             指定要冻结的基于1的最底行和最右列
#     返回:
#         out: <class 'NoneType'>
# class pandas.ExcelWriter(path, engine=None, date_format=None,
#                          datetime_format=None, mode='w', **engine_kwargs)
#     创建 ExcelWriter 对象
#     为了与CSV写入器兼容，ExcelWriter在写入之前将列表和字典序列化为字符串
#     参数:
#         path: str
#             路径
#         datetime_format: str
#             设置日期时间格式
#         mode: {‘w’, ‘a’}
#             文件模式(写入或追加工作表)
#     返回：
#         xlsx 格式对应  'pandas._OpenpyxlWriter'
#         xls 格式对应 class 'pandas.XlwtWriter'
datapath = Path('../data')
datapath.mkdir()

# 在工作簿中写入多个工作表，以及添加到现有的Excel文件
excel_path1 = Path('../data/test1.xlsx')
df1 = pd.DataFrame([['a', 'b', 'c'], ['d', 'e', 'f']],
                   index=['row1', 'row2'], columns=['col1', 'col2', 'col3'])
df2 = df1.transpose(copy=True)
print(df1, df2, sep='\n')
with pd.ExcelWriter(excel_path1) as writer:
    df1.to_excel(writer, sheet_name='表一')
    df2.to_excel(writer, sheet_name='表二')

with pd.ExcelWriter(excel_path1, mode='a') as writer:
    df1.to_excel(writer, sheet_name='表三')

# 将含有多层索引的DataFrame写入Excel
excel_path2 = Path('../data/test2.xlsx')
index = pd.MultiIndex.from_product([['a', 'b'], ['c', 'd']],
                                   names=['lvl1', 'lvl2'])
col = pd.MultiIndex.from_product([['col'], ['col1', 'col2', 'col3']],
                                 names=['lvl1', 'lvl2'])
df = pd.DataFrame([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]],
                  index=index, columns=col)
with pd.ExcelWriter(excel_path2) as writer:
    df.to_excel(writer, sheet_name='MultiIndex')

# 时间类型字符串写入工作表
excel_path3 = Path('../data/test3.xlsx')
df = pd.DataFrame([['AAA', '20180102', '9:00:00', '9:56:00', '0.0900'],
                   ['BBB', '2019/01/03', '15:00:00', '15:56:00', '0.1500'],
                   ['CCC', '01/04/2020', '20:00:00', '20:56:00', '-0.2000']],
                  index=['row1', 'row2', 'row3'],
                  columns=['col1', 'col2', 'col3', 'col4', 'col5'])
print(df)
with pd.ExcelWriter(excel_path3) as writer:
    df.to_excel(writer, sheet_name='表一')

# %% Excel 读取器
# pandas.read_excel()
#     读取Excel文件，依赖项 xlrd，返回类型 DataFrame
#     相似参数参见CSV
#         位置参数
#             io: str, bytes, ExcelFile, xlrd.Book, path, or file-like
#                 路径
#             sheet_name: str, int, list, or None, default 0  表名
#                 • 0: 第一个工作表
#                 • "Sheet1": 指定的工作表
#                 • [1, "Sheet5"] 用于表名的列表
#                 • None: 所有工作表
#             header: int, list of int, default 0
#                 用作DataFrame列标签的行
#             index_col: int, list of int, default None
#                 用作DataFrame行标签的列
#             usecols: int, str, list-like, or callable
#                 返回列的一个子集
#             dtype: type, default None
#                 数据或列的数据类型
#             na_values: scalar, str, list, or dict, default None
#                 转换字符串为无效值 NaN
#             comment: str, default None
#                 注释字符串和当前行末尾之间的数据
#             parse_dates: boolean or list or dict, default False
#                 将数据解析为日期:
#                     True: 解析索引
#                     [1, 2, 3]: 将第1、2、3列解析为单独的日期列
#                     [[1, 3]]: 合并第1列和第3列，并将其解析为单个日期列,解析器
#                         默认会删除组件的日期列
#                     {'foo': [1, 3]}: 合并第1列和第3列，并将其解析为单个日期列并
#                         设置列名，默认会删除组件的日期列
# class pandas.ExcelFile(io, engine=None)
#     创建 ExcelFile 对象
#     参数:
#         io: str, path, file-like, xlrd workbook or openpypl workbook
#             路径
#     返回：
#         xlsx 格式对应  'pandas._OpenpyxlWriter'
#         xls 格式对应 class 'pandas.XlwtWriter'
with pd.ExcelFile(excel_path1) as xlsx:
    print(xlsx.sheet_names)
    df1 = pd.read_excel(xlsx, '表一', index_col=0, usecols=[0, 1])
    df2 = pd.read_excel(xlsx, '表二', index_col=0)
    df3 = pd.read_excel(xlsx, '表三')
print(df1, df2, df3, sep='\n')

with pd.ExcelFile(excel_path1) as xlsx:
    df = pd.read_excel(xlsx, sheet_name=None, index_col=0)
print(df)

with pd.ExcelFile(excel_path2) as xlsx:
    df = pd.read_excel(xlsx, index_col=[0, 1], header=[0, 1])
print(df)

with pd.ExcelFile(excel_path3) as xlsx:
    df1 = pd.read_excel(xlsx, index_col=2, parse_dates=True)
    df2 = pd.read_excel(xlsx, index_col=0, parse_dates=['col2'])
    df3 = pd.read_excel(xlsx, parse_dates=[[2, 3], [2, 4]])
print(df1, df2, df3, sep='\n')

shutil.rmtree(datapath, onerror=remove_readonly)

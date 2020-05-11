# -*- coding: utf-8 -*-
"""
Created on 2020/4/23 15:38 by PyCharm

@author: xumiz
"""
# %% 模块导入
import csv
from pathlib import Path

# %% 模块常量
# csv.QUOTE_ALL         引用所有字段
# csv.QUOTE_MINIMAL     仅引用包含特殊字符（如delimiter、quotechar字符）的字段
# csv.QUOTE_NONNUMERIC  仅引用所有非数值字段
# csv.QUOTE_NONE        不引用字段，当此设置引发歧义（如字段中包含delimiter）,可
#                       通过设置escapechar，使它出现在delimiter之前避免歧义
print(csv.QUOTE_ALL, csv.QUOTE_MINIMAL, csv.QUOTE_NONNUMERIC, csv.QUOTE_NONE)

# %% csv 写入器
# csv.writer(csvfile, dialect='excel', **fmtparams)
#     将数据写入csv,返回一个写入器对象
#     参数：
#         csvfile: file-like object
#             传入file对象，用newline= "打开
#         **fmtparams
#             格式化参数:
#                 Dialect.delimiter      设置分隔符标志，默认为 ','
#                 Dialect.quotechar      设置引用标志，默认为 '"'
#                 Dialect.quoting        设置引用方式，默认为 QUOTE_MINIMAL
#                 Dialect.escapechar     设置转义字符，默认为 None（禁用转义）
#                 Dialect.lineterminator 终止行的字符串，默认为'\r\n'
#                 Dialect.skipinitialspace
#                     True: 忽略紧跟在delimiter后面的空白，默认为False
#                 Dialect.doublequote
#                     True：如字段中包含quotechar将此quotechar双倍化呈现(默认)
#                     False： 如字段中包含quotechar将escapechar作quotechar的前缀
#     返回：<class '_csv.writer'>
# DictWriter与_csv.writer的公共方法
# writerow(self, rowdict)
#     将rowdict参数根据当前方言格式化写入file对象
#     参数：
#         rowdict: iterable of strings or numbers
# writerows(self, rowdicts)
#     将rowdicts参数根据当前方言格式化写入file对象
Path('../data').mkdir()
csv_test = Path('../data/test.csv')
with open(csv_test, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['Spam', 2, 7, 'Spam', 5, 'Spam', 'Baked Beans'])

with open(csv_test, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['Spam', 2, 7, 'Spam', 5, 'Spam', 'Baked,Beans'])

with open(csv_test, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, quotechar="'")
    spamwriter.writerow(['Spam', 2, 7, 'Spam', 5, 'Spam', 'Baked,Beans'])

with open(csv_test, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, doublequote=True)
    spamwriter.writerow(['Spam', 2, 7, 'Spam', 5, 'Spam', 'Baked""Beans'])

with open(csv_test, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, doublequote=False, escapechar='\\')
    spamwriter.writerow(['Spam', 2, 7, 'Spam', 5, 'Spam', 'Baked"Beans'])

with open(csv_test, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar='\\')
    spamwriter.writerow(['Spam', 2, 7, 'Spam', 5, 'Spam', 'Baked,Beans'])

with open(csv_test, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerows([['Spam', 'Baked Beans'],
                          ['Spam', 'Lovely Spam', 'Wonderful Spam']])
csv_test.unlink()

# %% DictWriter 字典写入器
# class csv.DictWriter
# DictWriter(f, fieldnames, restval='', extrasaction='raise', dialect='excel',
#            *args, **kwds)
#     将字典映射到输出行写入csv
#     参数：
#         fieldnames: a sequence of keys
#             用于确定将字典中值写入文件的顺序
#         restval：str
#             指定当字典在fieldnames中缺少一个键时要写入的值
#         extrasaction:['raise', 'ignore']
#             控制字典包含fieldnames中没有找到的键时的操作
#                 'raise': 引发ValueError
#                 'ignore': 忽略字典中的额外值
#     返回：<class 'csv.DictWriter'>
# DictWriter.writeheader()
#     将fieldnames写入file对象
csv_test = Path('../data/test.csv')
with open(csv_test, 'w', newline='') as csvfile:
    fieldnames = ['first_name', 'last_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
    writer.writerows(({'first_name': 'Lovely', 'last_name': 'Spam'},
                      {'first_name': 'Wonderful', 'last_name': 'Spam'}))
csv_test.unlink()

# %% csv.excel 对象
# class excel(Dialect)
# excel()
#     描述Excel生成的CSV文件的一般属性
dia = csv.excel()
print(dia.quoting)
print(dia.doublequote)
print(dia.escapechar)
print(dia.quotechar)
print(dia.delimiter)
repr(dia.lineterminator)
print(dia.skipinitialspace)

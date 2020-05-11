# -*- coding: utf-8 -*-
"""
Created on 2020/4/8 16:47 by PyCharm

@author: xumiz
"""
# %% zip 对象
# class zip(object)
# zip(*iterables)
#     创建一个迭代器，它聚合来自每个迭代器的元素
#     参数：
#         iterables: iterable
#             可迭代对象
#     返回：
#         <class 'zip'>  zip迭代器
x = [1, 2, 3]
y = [4, 5, 6]
zipped = zip(x, y)
for i in zipped:
    print(i)

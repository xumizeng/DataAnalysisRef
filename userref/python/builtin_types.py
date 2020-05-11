# -*- coding: utf-8 -*-
"""
Created on 2020/4/8 16:02 by PyCharm

@author: xumiz
"""
# %% 布尔类型(Boolean Types)
# class bool
# bool([x])
#     真值检测
#     返回：
#         <class 'bool'>
bool()
bool(None)
bool(0)

# %% 数字类型(Numeric Types)
# class int
# int([x])
#     返回一个由数字或字符串构造的整数对象
# int(x, base=10)
#     返回一个由字符串、字节或字节数组实例构造的整数对象
#     返回：
#         <class 'int'>
int(3.4)
int()
int('34')

int('110', base=2)

# class float([x])
# float([x])
#     返回由数字或字符串x构造的浮点数
#     返回：
#         <class 'float'>
float('   -12.34\n')
float('1e-003')
float()

# class complex
# complex([real[, imag]])
#     返回real+ imag*1j的复数
#     返回：
#         <class 'complex'>
complex(2, 4)
complex('1+2j')
complex()


# %% 迭代器(Iterator Types)
# iter(object[, sentinel])
#     返回一个迭代器对象
#     如果给出了sentinel，对象必须是一个可调用的对象,返回的值等于sentinel，
#     则将引发StopIteration，否则将返回该值
# next(iterator[, default])
#     从迭代器中检索下一项
#     default在迭代器耗尽时返回，未指定将引发StopIteration
def userzip(*iterables):
    # zip('ABCD', 'xy') --> Ax By
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)


zipped = userzip('ABCD', 'xy')
list(zipped)

# %% 生成器类型(Generator Types)
# class generator(object)
#     通过生成器表达式或Yield表达式创建
generator = (x * y for x in range(2) for y in range(x, x + 2))
for i in generator:
    print(i)


def echo(value=None):
    print("Execution starts when 'next()' is called for the first time.")
    try:
        while True:
            try:
                value = (yield value)
            except Exception as e:
                value = e
    finally:
        print("Don't forget to clean up when 'close()' is called.")


# generator方法
# generator.__next__()
#     启动生成器函数的执行，或在最后执行的yield表达式时继续执行
#     通常隐式调用，例如由for循环调用，或由next()函数调用
# generator.send(value)
#     继续执行并向生成器函数传递一个值，value成为当前yield表达式的结果
# generator.close()
#     在生成器函数暂停的地方引发GeneratorExit，关闭生成器
generator = echo(1)
print(next(generator))
print(next(generator))
print(generator.send(2))
print(next(generator))
generator.close()

# %% 序列类型(Sequence Types)
# class list(object)
# list(iterable=())
#     构建一个列表
#     返回：
#         <class 'list'> 可变序列
lst = [x * y for x in range(2) for y in range(x, x + 2)]
print(lst)
list(('a', None, 2))
list('abc')

# class tuple(object)
# tuple(iterable=())
#     构建一个元组
#     返回：
#         <class 'tuple'> 不可变序列
tuple(('a', None, 2))
tuple('abc')

# class range(object)
# class range(stop)
# class range(start, stop[, step])
#     返回一个整数序列
#     返回：
#         <class 'range'> 不可变序列
list(range(0, 10, 3))

# class str(object)
# str(object='')
#     从给定对象创建一个新的字符串对象
#     返回：
#         <class 'str'> 不可变序列
str("A")
str('A')
str("""A
    d,h""")
str('''A
    d,h''')

# %% 集合类型(Set Types)
# class set(object)
# set([iterable])
#     构建惟一元素的无序集合
#     返回：
#         <class 'set'>  可变序列
set('abcb')
set()

# class frozenset(object)
# frozenset([iterable])
#     构建一个不可变的、无序的唯一元素集合
#         <class 'frozenset'> 不可变序列
frozenset('abcb')
frozenset()

# %% 映射类型(Mapping Types)
# class dict(object)
# dict(**kwarg)
#     使用关键字参数列表中的 name=value 对初始化新字典
# dict(mapping)
#     通过映射对象 (key, value) 对初始化的新字典
# dict(iterable)
#     通过可迭代对象初始化的新字典
#     参数：
#         **kwarg: name=value
#             键值对
#         mapping: mapping object
#             映射对象
#         iterable: iterable
#             可迭代对象
#     返回：
#         <class 'dict'>  字典，可变容器，不可哈希(unhashable)
dic1 = dict()
dic2 = dict(one=1, two=2, three=3)
dic3 = dict([('two', 2), ('one', 1), ('three', 3)])
dic4 = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
print(dic1, dic2, dic3, dic4, sep='\n')

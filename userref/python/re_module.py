# -*- coding: utf-8 -*-
"""
Created on 2020/4/18 16:34 by PyCharm

@author: xumiz
"""
# %% 模块导入
import re

# %% Match对象构造器
# re.match(pattern, string, flags=0)
#     字符串开头的零个或多个字符与正则表达式模式匹配，则返回相应的匹配对象
#     如果字符串与模式不匹配，则返回None
#     返回：Match
re.match(pattern=r"(\w+) (\w+)", string="Isaac Newton, physicist")
re.match(pattern=r"(\w+) (\w+)", string="physicist, Isaac Newton")

# re.search(pattern, string, flags=0)
#     扫描字符串，寻找正则表达式模式产生匹配的第一个位置，并返回相应的匹配对象
#     如果字符串中没有与模式匹配的位置，则返回None
#     返回：Match
re.search(pattern=r"(\w+) (\w+)", string="Isaac Newton, physicist")
re.search(pattern=r"(\w+) (\w+)", string="physicist, Isaac Newton")

# %% Match 方法
# group([group1, ...])
#     返回匹配项的一个或多个组合，如果一个组匹配多次，只有最后一个匹配可单独访问的
# groupdict(self, default=None)
#     返回一个字典，其中包含匹配的所有命名子组
m = re.match(pattern=r"(\w+) (\w+)", string="Isaac Newton, physicist")
print(m)
m.group(0)
m.group(1)
m.group(2)
m.group(1, 2)
m.groupdict()

m = re.match(r"(?P<A>\w+) (?P<B>\w+)", "Isaac Newton, physicist")
print(m)
m.group(0)
m.group(1)
m.group(2)
m.group(1, 2)
m.group('A')
m.group('B')
m.group('A', 'B')
m.groupdict()

m = re.match(r"(..)+", "a1b2c3")
print(m)
m.group(0)
m.group(1)
m.groupdict()


# %% 其他顶级函数
# re.findall(pattern, string, flags=0)
#     返回字符串中所有非重叠匹配
#     返回：list
# re.sub(pattern, repl, string, count=0, flags=0)
#     使用repl替换字符串中非重叠匹配，如果没有找到，原样返回字符串
#     返回：str
def dash(mobj):
    if mobj.group(0) == '-':
        return ' '
    else:
        return '-'


re.findall('-{1,2}', 'pro----gram-files')
re.sub('-{1,2}', dash, 'pro----gram-files')

re.sub(r'\sAND\s', ' & ', 'Beans And Spam And Else', flags=re.IGNORECASE)
re.sub(r'\sAND\s', ' & ', 'Beans And And Spam', flags=re.IGNORECASE)
re.sub(r'\sAND\s', ' & ', 'Beans And  And Spam', flags=re.IGNORECASE)
re.sub(r'\sAND\s', ' & ', 'Beans or Spam', flags=re.IGNORECASE)

# %% 附录一： 模块常量
# 本模块中的一些函数将如下常用标志作为可选参数:
# A  ASCII       使 \w, \W, \b, \B, \d, \D, \s 和 \S 只匹配ASCII，而非Unicode
# I  IGNORECASE  忽略大小写匹配
# M  MULTILINE   "^" 匹配字符串的开始，以及换行后每一行的开始
#                "$" 匹配字符串的结尾，以及换行后前一行的末尾
# S  DOTALL      "." 匹配任何字符，包括换行符
# X  VERBOSE     忽略空白和注释

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
# {m,n}    对它之前的正则式进行 m 到 n 次匹配，在 m 和 n 之间取尽量多
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
#
# 特殊序列：
# \d       对于 Unicode (str) 样式：匹配任何Unicode十进制数包括了 [0-9] 以及很多
#          其他的数字字符
# \s       匹配Unicode空白字符，其中包括[ \t\n\r\f\v]
# \w       匹配Unicode单词字符

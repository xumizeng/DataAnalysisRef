# -*- coding: utf-8 -*-
"""
Created on 2020/4/25 11:08 by PyCharm

@author: xumiz
"""
# %% 模块导入
from pathlib import Path
import stat

# %% 具体路径(Concrete paths)
# class pathlib.Path
# Path(*pathsegments)
#     实例化具体路径
Path('../data')

# %% Concrete paths方法
# classmethod Path.cwd()
#     返回一个表示当前工作目录的路径对象
Path.cwd()

# resolve(self, strict=False)
#     返回绝对路径
Path().resolve()
Path('..').resolve()

# mkdir(self, mode=0o777, parents=False, exist_ok=False)
#     在给定的路径上创建一个新目录
#     参数：
#         parents: bool
#             缺少父类默认引发FileNotFoundError，否则使用默认权限（mode=0o777）
#             创建此路径中所有缺失的父类
#         exist_ok: bool
#             路径已存在默认引发FileExistsError，否则忽略此异常且操作无效
# rmdir(self)
#     删除空目录
Path('../date').mkdir(mode=stat.S_IREAD)
Path('../data/io/csv').mkdir(parents=True)

Path('../data/io/csv').rmdir()
Path('../data/io').rmdir()

# touch(self, mode=0o666, exist_ok=True)
#     在给定的路径上创建一个文件
#     参数：
#         mode: numeric
#             Windows只支持设置文件的只读标志（stat.S_IREAD，stat.S_IWRITE），
#             所有其他位都被忽略
#         exist_ok: bool
#             路径已存在默认忽略FileExistsError且操作无效，否则引发异常
# write_text(self, data, encoding=None, errors=None)
#     以文本模式打开文件向内写入数据，然后关闭文件
# read_text(self, encoding=None, errors=None)
#     以字符串的形式返回文件内容
# Path.stat(self)
#     返回此路径信息的stat_result对象
# Path.chmod(self, mode)
#     更改文件模式和权限
# unlink(self)
#     删除文件
p = Path('../data/test1.txt')
p.touch(exist_ok=False)
p.write_text('Text file contents')
p.read_text()
p.unlink()

p = Path('../data/test2.txt')
p.touch(mode=stat.S_IREAD, exist_ok=False)
p.read_text()
oct(p.stat().st_mode)
p.chmod(mode=stat.S_IWRITE)
oct(p.stat().st_mode)
p.unlink()

Path('../data').rmdir()
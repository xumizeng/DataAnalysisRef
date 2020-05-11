# -*- coding: utf-8 -*-
"""
Created on 2020/4/26 17:43 by PyCharm

@author: xumiz
"""
# %% 模块导入
import shutil
from pathlib import Path
import stat


# %% 目录和文件操作
# shutil.rmtree(path, ignore_errors=False, onerror=None)
#     删除整个目录树
#     参数：
#         ignore_errors: bool
#             False：清除失败导致错误发生时，通过调用onerror=func处理异常:
#             True：忽略清除失败所导致的错误
#         onerror: None or func
#             可调用的函数func接受三个参数(function, path, excinfo):
#                 引发异常的函数、路径和 sys.exc_info()返回的异常信息
# 清空数据文件夹
def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal"""
    Path(path).chmod(stat.S_IWRITE)
    func(path)


Path('../data/io/csv').mkdir(parents=True)
Path('../data/io/csv/test.txt').touch()
shutil.rmtree(Path('../data'), onerror=remove_readonly)

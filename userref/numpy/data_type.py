# -*- coding: utf-8 -*-
"""
Created on 2020/4/8 21:26 by PyCharm

@author: xumiz
"""
# %% 模块导入
import numpy as np

# %% dtype 对象
# class dtype(builtins.object)
# dtype(obj[, align, copy])
#     创建一个数据类型对象
#     参数：
#         obj
#             可转换对象，见附录
#     返回
#         out: <class 'numpy.dtype'>
# 直接使用 dtype 对象
np.dtype('int8')

# 使用 Array-scalar 类型
np.dtype(np.int8)
np.dtype((np.void, 10))

# 使用 Python的内置类型
np.dtype(int)

# 使用 One-character 字符串
np.dtype('<f')

# 使用 Array-protocol类型字符串
np.dtype('i4')

# 结构化类型：[(field_name, field_dtype, [field_shape]), ...]
np.dtype([('f1', np.int16)])
np.dtype([('f1', [('f1', np.int16)])])
np.dtype([('f1', np.uint64), ('f2', np.int32)])
np.dtype([('hello', (np.int64, 3)), ('world', np.void, 10)])
np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])

# 用逗号分隔字段的字符串
np.dtype("i4, (2,3)f8, 2f4")

# 使用元组 (fixed_dtype, shape)
np.dtype((np.int32, (2, 2)))
np.dtype(('i4, (2,3)f8, f4', (2, 3)))

# 使用字典 {'names':.., 'formats':..}
np.dtype({'names': ['gender', 'age'], 'formats': ['S1', np.uint8]})
np.dtype({'names': ['r', 'g', 'b', 'a'],
          'formats': ['uint8', 'uint8', 'uint8', 'uint8']})

# %% 附录:
# 可转换对象:
#     dtype object
#         'bool'
#         'int8', 'int16', 'int32', 'int64',
#         'uint8', 'uint16', 'uint32', 'uint64'
#         'float16', 'float32', 'float64', 'complex64', 'complex128'
#         'O', 'S', '<U', 'V', '<m8', '<M8'
#     None
#         默认数据类型为 float_
#     Array-scalar
#         24个内置的数组标量类型对象
#             布尔值: bool_ (bool8, '?')
#             数字：
#                 整型                      无符号整型
#                 byte(int8, 'b')           ubyte(uint8, 'B')
#                 short(int16, 'h')         ushort(uint16, 'H')
#                 intc(int32, 'i')          uintc(uint32, 'I')
#                 int_(int32, 'l')          uint(uint32, 'L')
#                 longlong(int64, 'q')      ulonglong(uint64, 'Q')
#                 intp(int64, 'p')          uintp(uint64, 'P')
#                 浮点数                     复数
#                 half(float16, 'e')        csingle(complex64, 'F')
#                 single(float32, 'f')      complex_(complex128, 'D')
#                 float_(float64, 'd')      clongfloat(complex128, 'G')
#                 longfloat(float64, 'g')
#             Python对象：
#                 object_('O')
#             灵活类型 flexible:
#                 bytes_('S#'), unicode_('U#'), void('V#')
#                 灵活的数据类型的默认项大小为0，需要显式给定的大小：
#                 (flexible_dtype, itemsize)
#     Python的内置类型
#         int, bool, float, complex, bytes, str, object
#     One-character 字符串
#         单字符指定数据的类型
#     Array-protocol类型字符串
#         第一个字符指定数据的类型，其余的字符指定每个项目的字节数
#     日期类型字符串
#         'datetime64', 'Datetime64', 'M8', 'timedelta64', 'Timedelta64', 'm8'
#     用逗号分隔字段的字符串
#         生成的数据类型字段名为'f0'、'f1'、…、'f<N-1>'，eg:
#             "i4, (2,3)f8, 2f4"
#         f0包含一个32位整数,f1包含一个2x3浮点数子数组,f2包含一个2x1浮点数子数组
#     元组：
#         (fixed_dtype, shape)
#     列表：
#         [(field_name, field_dtype, [field_shape]), ...]
#     字典：
#         {'names': ..., 'formats': ...,
#          ['offsets: ...'], ['titles': ...], ['itemsize: ...']}
#
# 备注：
#     **_   表示对相应的Python类型兼容
#     #     表示一个整数
#     '>'   大端字节顺序(big-endian)
#     '<'   小端字节顺序(little-endian)
#     '|'   无关字节顺序(not-relevant)
#     '='   默认字节顺序(hardware-native, the default)

# PythonDiary
 Python日记

python日记 1 | 环境搭建

平台：Windows

基础软件：

    Python发行版本： Miniconda3-latest-Windows-x86_64.exe
    软件项目的托管平台：Git-2.23.0-64-bit.exe
    IDE：pycharm-professional-2019.2.2.exe

详细步骤：

    1.Miniconda3
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    如需导入到Pycharm，安装过程中请勾选添加到环境变量选项

    下载安装完毕，打开：开始菜单-Anaconda3(64-bit)-Anaconda Prompt (miniconda3)，
    输入指令：conda update --all ，在命令行 Proceed ([y]/n)? 后输入 y ，等待更新

    卸载：

    安装 Anaconda-Clean 软件包：conda install anaconda-clean
    在同一窗口中，运行以下命令：anaconda-clean --yes
    在安装的根目录中删除 envs 和 pkgs 文件夹，之后常规卸载程序

    2.Git
    https://git-scm.com/download/win
    安装过程中，勾选自动更新选项

    下载安装完成后打开，设置GitHub账号。若没有此账号，登录官网按照提示进行注册：
    https://github.com/join?source=header-home
    注册完成后在网页 https://github.com/ 上重新登录。登录后点击右侧头像下拉菜单  
    Your repositories，可通过界面中的按钮 New 创建一个新仓库。
    如果Github网络端中Emai勾选了私人隐私保护，邮箱设置为被转换后的邮箱地址。
    git config --global user.name "xumi"
    git config --global user.email 53687238+xumizeng@users.noreply.github.com

    3.Pycharm
    http://www.jetbrains.com/pycharm/download/
    安装过程中勾选 64-bit launcher | .py | Add launchers dir to the PATH 三项
    安装完成，右键点击Pycharm快捷方式图标，属性中设置兼容性，勾选以管理员身份运行此程序
    重启，打开程序，导入备份设置（如果存在），安装完毕简单设置下：
    在欢迎界面，展开底边设置，检查更新后重启：
    国内的pip镜像：
          清华：https://pypi.tuna.tsinghua.edu.cn/simple
          阿里：http://mirrors.aliyun.com/pypi/simple/
          豆瓣：http://pypi.douban.com/simple/
          华中理工大学：http://pypi.hustunique.com/
          山东理工大学：http://pypi.sdutlinux.org/
          中国科学技术大学：http://pypi.mirrors.ustc.edu.cn/
	
------------------------------------------------------------------------------

python日记 2 | Python语言参考（词法分析）

定义编码

    Python 通过编码声明或默认为 UTF-8将程序文本读取为 Unicode 代码点。
    这里出现了两个关键词：UTF-8和Unicode，解释之前先解释另外一个关键词：ASCII

    标准ASCII：是基于拉丁字母的一套电脑编码系统，用于显示现代英语和其他西欧语言，
    到目前为止共定义了128个字符。
        0～31及127是控制字符或通信专用字符，共33个不可显示字符
        32～126(共95个)是可显示字符
        最高位用作奇偶校验位。用来检验数据传输是否出现错误，分奇校验和偶校验

    Unicode：统一码。在英语中，用ASCII码便可以表示所有，但表示其他语言这是不够的。
    为统一编码，Unicode应运而生。Unicode通常用两个字节表示一个字符，
    原有的ASCII码从单字节变成双字节。例如中文范围 4E00-9FA5

    UTF-8：Unicode字符集转换格式。由于对可以用ASCII表示的字符使用Unicode并不高效，
    为了解决这个问题，UTF应运而生。常见的UTF格式：UTF-8,UTF-16, UTF-32

行结构

    Python程序由解析器读取被划分为许多逻辑行。逻辑行通过显式或隐式行连接规则从一条
    或多条物理行构造。物理行是由行尾序列终止的字符序列。可以理解为我们输入的代码，
    以ASCII 序列 CR LF（回车加换行）终止。注释是被语法忽略的物理行，以字符#开头
    可能出现在行的开头或空格或代码之后，但不能出现在字符串文本中。

    显式行连接：当物理行非字符串文本或注释不以反斜杠结束，反斜杠将字符多条物理行
    连入逻辑行，其后不能带有注释。
    隐式行连接：括号、方括号或大括号中的表达式可以拆分到多个物理行上
    Python缩进：由于用于确定语句的分组，所以对物理行缩进的要求相当严格，如果源文件
    混合了制表符和空格，可能会引发TabError。

形符

    NEWLINE, INDENT,  DEDENT, 标识符, 关键字, 字面值, 运算符, 分隔符
    空白字符 (行终止符除外) 不属于形符，而是用来分隔形符

 ----------------------------------------------------------------------------

python日记 3 | Python语言参考（数据模型一）

标准类型层级结构
    None
        表示空值，例如未显式指明返回值的函数将返回 None。逻辑值为假
    NotImplemented
        数值方法和比较方法如未实现指定运算符表示的运算则应返回此值。
        逻辑值为真
    Ellipsis
        通过字面值 ... 或内置名称 Ellipsis 访问。逻辑值为真
    numbers.Number
        numbers.Integral(int, bool)
            此类对象表示数学中整数集合的成员 (包括正数和负数)。

        numbers.Real (float)
            此类对象表示双精度浮点数
        numbers.Complex (complex)
            此类对象以一对双精度浮点数来表示复数值
    序列
        此类对象表示以非负整数作为索引的有限有序集。len() 可返回一个序列的条目数量
        当序列长度为n时，索引集包含0, 1, ..., n-1。序列a的条目i可通过 a[i] 选择。
        支持切片a[i:j:k] 选择a中索引号为x的所有条目，x=i+n*k,n>=0且i<=x<j
        不可变序列：
            字符串：Unicode码位值（U+0000-U+10FFFF）组成的序列。Python没有char型
            元组：元组中的条目可以是任意 Python 对象
            字节串：数组条目都是一个 8 位字节，以取值范围 0 <= x < 256 的整型表示
        可变序列
            列表：条目可以是任意 Python 对象
            字节数组：可变的 (因而也是不可哈希的)
    集合
        集合, 冻结集合：由不重复不可变对象组成的无序且有限的集合。不能通过下标索引
    映射
        字典：由几乎任意值作为索引的有限个对象的集合
    可调用类型
        用户定义函数
            特殊属性:
            __doc__         该函数的文档字符串，没有为None；不会被子类继承。可写
            __name__        该函数的名称。可写
            __qualname__    该函数的 qualified name。可写
            __module__      该函数所属模块的名称，没有则为 None。可写
            __defaults__    由默认参数值组成的元组，默认为None。可写
            __code__        表示编译后的函数体的代码对象。可写
            __globals__     对存放该函数中全局变量的字典的引用，函数所属模块的
                            全局命名空间。 只读
            __dict__        命名空间支持的函数属性。可写
            __closure__     None 或包含该函数可用变量的绑定的单元的元组。只读
            __annotations__ 包含参数标注的字典。字典的键是参数名，如存在返回标注
                            则为 'return'。可写
            __kwdefaults__  仅包含关键字参数默认值的字典。可写
        实例方法
            实例方法用于结合类、类实例和任何可调用对象 (通常为用户定义函数)。
            特殊属性：
            __self__        类实例对象本身
            __func__        函数对象
            __doc__         方法的文档
            __name__        方法名称
            __module__      方法所属模块的名称，没有则为 None。
        生成器函数
            一个使用 yield 语句的函数或方法
        协程函数
            使用 async def 来定义的函数或方法
        异步生成器函数
            使用 async def 来定义并包含 yield 语句的函数或方法
        内置函数
            特殊属性：
             __doc__        函数的文档字符串，如果没有则为 None
             __name__       函数的名称
             __self__       设定为 None
             __module__     函数所属模块的名称，如果没有则为 None。
        内置方法
            内置函数的另一种形式
        类
            特殊属性:
            __name__            类的名称
            __module__          类所在模块的名称
            __dict__            包含类命名空间的字典
            __bases__           包含基类的元组，按其在基类列表中的出现顺序排列
            __doc__             类的文档字符串，如果没有则为 None
            __annotations__     为一个包含变量标注的字典，在类体执行时获取。(可选)
        类实例
            任意类的实例通过在所属类中定义 __call__() 方法即能成为可调用的对象。
            特殊属性:
            __dict__ 为属性字典
            __class__ 为实例对应的类
    模块
        可由 import 语句发起调用，模块对象具有由字典对象实现的命名空间。
    自定义类
    I/O 对象
        多种创建文件对象的快捷方式
        内置函数open(), os.popen(), os.fdopen(),socket 对象的 makefile() 方法

    内部类型
        代码对象
            表示 编译为字节的 可执行 Python 代码
        帧对象
            表示执行帧。它们可能出现在回溯对象中，还会被传递给注册跟踪函数
        回溯对象
            表示一个异常的栈跟踪记录。当异常发生时会隐式地创建一个回溯对象
            也可通过调用 types.TracebackType 显式地创建。
        切片对象
            表示 __getitem__() 方法得到的切片。可使用内置的 slice() 函数来创建。
        静态方法
            提供了一种避免上文所述将函数对象转换为方法对象的方式
        类方法
            从类或类实例获取该对象

------------------------------------------------------------------------------

python日记 4 | Python语言参考（数据模型二）

特殊方法名称（基本定制）

    object.__new__(cls[,...])
        创建一个 cls 类新实例时被调用。典型实现使用 super().__new__(cls[, ...])
        来创建一个类的新实例，然后根据需要修改新创建的实例再将其返回。
        如果 __new__() 未返回一个cls的实例，则新实例的 __init__() 方法不会被执行
    object.__init__(self[, ...])
        实例(通过 __new__()) 被创建之后调用。基类和派生的类如果都有__init__() 方法
        必须显式地调用它以确保实例基类正确初始化，如: super().__init__([args...])
    object.__del__(self)
        在实例将被销毁时调用。一个基类以及其派生的类如果都有 __del__() 方法，必须
        显式地调用它以确保实例基类部分的正确清除。del x 并不直接调用 x.__del__()
        前者会将 x 的引用计数减一，而后者仅会在 x 的引用计数变为零时被调用。
    object.__repr__(self)
        由 repr() 内置函数调用以输出一个对象的规范字符串表示        
    object.__str__(self)
        通过 str(object) 以及内置函数 format() 和 print() 调用以生成一个对象的
        “非正式”或格式良好的字符串表示
    object.__bytes__(self)
        通过 bytes 调用以生成一个对象的字节串表示
    object.__format__(self, format_spec)
        通过 format() 内置函数扩展格式化字符串字面值的求值以及 str.format() 方法
        调用以生成一个对象的“格式化”字符串表示。
    object.__hash__(self)
        通过内置函数 hash() 调用以对哈希集的成员进行操作
        属于哈希集的类型包括 set、frozenset 以及 dict
    object.__bool__(self)
        调用此方法以实现真值检测以及内置的 bool() 操作；应该返回 False 或 True。        
    object.__lt__(self, other)  “富比较”方法，x<y 调用
    object.__le__(self, other)  “富比较”方法，x<=y  调用
    object.__eq__(self, other)  “富比较”方法，x==y 调用
    object.__ne__(self, other)  “富比较”方法，x!=y 调用
    object.__gt__(self, other)  “富比较”方法，x>y 调用
    object.__ge__(self, other)  “富比较”方法，x>=y 调用

特殊方法名称（自定义属性访问）

    自定义类实例属性访问
    object.__getattr__(self, name)
        当默认属性访问引发 AttributeError时被调用,返回（找到的）属性值或引发一个
        AttributeError 异常。引发 AttributeError原因可能是调用__getattribute__()
        时由于 name 不是一个实例属性或不是self 的类关系树中的属性而引发，也可能是
        对 name 特性属性调用 __get__() 时引发
    object.__getattribute__(self, name)
        访问类实例属性时调用，此方法返回（找到的）属性值或引发一个 AttributeError
    object.__setattr__(self, name, value)
        一个属性被尝试赋值时被调用
    object.__delattr__(self, name)
        一个属性被尝试删除时被调用
    object.__dir__(self)
        对相应对象调用 dir() 时被调用。返回一个属性序列。

------------------------------------------------------------------------------

python日记 5 | Python语言参考（语句一）

简单语句

    pass 语句
        空操作，被执行时什么都不发生
    del 语句
        目标列表的删除将从左至右递归地删除每一个目标。
        名称的删除将从局部或全局命名空间中移除该名称的绑定
        属性引用、抽取和切片的删除会被传递给相应的原型对象
    return 语句
        return 会离开当前函数调用，并以表达式列表 (或 None) 作为返回值
    yield 语句
        仅在定义 generator 函数或是 asynchronous generator 时使用
        当一个生成器函数被调用的时候，返回一个用来控制生成器函数的执行的生成器。
        当此生成器的某个方法被调用，生成器函数开始执行到第一个 yield 表达式，
        在此执行再次被挂起，给生成器的调用者返回 expression_list 的值。挂起后，
        通过调用生成器的某一个方法，生成器函数继续执行。恢复后 yield 表达式的值
        取决于调用的哪个方法来恢复执行， 生成器-迭代器的方法如下：
        generator.__next__()
            开始一个生成器函数的执行或是从上次执行的 yield 表达式位置恢复执行
            此方法通常是隐式地调用，例如通过 for 循环或是内置的 next() 函数。
            如果生成器没有产生下一个值就退出，则将引发 StopIteration 异常。
        generator.send(value)
            恢复执行并向生成器函数“发送”一个值。 value即当前 yield 表达式的结果
            如果生成器没有产生下一个值就退出则会引发 StopIteration。
        generator.throw(type[, value[, traceback]])
            在生成器暂停的位置引发type类型的异常，返回该生成器函数所产生的下一个值
            如果生成器没有产生下一个值就退出，则将引发 StopIteration 异常。
            如果生成器函数未捕获异常或引发了另一个异常，该异常会被传递给调用者。
        generator.close()
            在生成器函数暂停的位置引发 GeneratorExit。
            如果之后生成器函数正常退出、关闭或引发 GeneratorExit则关闭并返回调用者
            如果生成器产生了一个值，关闭会引发 RuntimeError。
            如果生成器引发任何其他异常，它会被传播给调用者。
            如果生成器已经由于异常或正常退出则 close() 不会做任何事。
    raise 语句
        raise后无表达式
            重新引发当前作用域内最后一个激活的异常。没有激活的异常引发RuntimeError
        raise后带表达式
            将第一个表达式求值为BaseException 的子类或实例，from 子句用于异常串连
    break 语句
        终结最近的外层循环，如果循环有可选的 else 子句，也会跳过该子句。
    continue 语句
        继续最近的外层循环的下一个循环，跳过子句体中的剩余部分并返回检验表达式，
    import 语句
        不带 from 子句的执行过程:
            查找一个模块，如果有必要还会加载并初始化模块。
            在局部命名空间中为 import 语句发生位置所处的作用域定义一个或多个名称。
        带有 from 子句的执行过程：
            查找 from 子句中指定的模块，如有必要还会加载并初始化模块；
            对于 import 子句中指定的每个标识符，检查被导入模块是否有该名称的属性
                如果有，将该名称存入局部命名空间（如果有as子句则使用其指定的名称）
                如果没有，尝试导入具有该名称的子模块，然后再次检查被导入模块
                仍未找到该属性，则引发 ImportError
    global 语句
        作用于整个当前代码块的声明，意味着所列出的标识符将被解读为全局变量。
    nonlocal 语句
        使得所列出的名称指向之前在最近的包含作用域中绑定的除全局变量以外的变量

--------------------------------------------------------------------------

python日记 6 | Python语言参考（语句二）

复合语句

    if 语句
        用于有条件的执行，表达式为真执行该子句体；表达式均为假值，执行else 子句
    while 语句
        在表达式保持为真的情况下重复地执行，表达式值为假则终止循环执行else 子句
    for 语句
        表达式列表会被求值一次产生一个可迭代对象。 系统将为该对象创建一个迭代器，
        然后将为迭代器的每一项执行一次子句体。当所有项被耗尽时执行可选的else 子句
        的子句体终止循环。表达式是一个可变序列，应对整个序列使用切片来创建一个
        临时副本（浅拷贝），避免当序列在循环中被修改
    try 语句
        为一组语句指定异常处理器或清理代码。当 try 子句中发生异常时，将对except
        子句中异常处理器依次搜索。 如果最后一个except 子句无表达式将匹配任何异常。
        搜索到一个匹配的 except 子句时，执行该 except 子句体。当使用 as 将目标赋值
        为一个异常时，它将在 except 子句结束时被清除。
        在一个 except 子句体被执行之前，有关异常的详细信息存放在sys模块中，可通过
        sys.exc_info() 来访问，它返回一个 3 元组，由异常类、异常实例和回溯对象组成
        finally 子句可被用来指定清理代码，无论之前的代码是否发生异常都会被执行。
        当在try语句中执行return，break或continue语句时，finally子句也会执行。
        
------------------------------------------------------------------------------

附录

SQLite

SQLite 命令
    与关系数据库进行交互的标准 SQLite 命令类似于 SQL。基于操作性质可分为以下几种：
DDL - 数据定义语言

    命令     描述
    CREATE  创建一个新的表，一个表的视图，或者数据库中的其他对象。
    ALTER   修改数据库中的某个已有的数据库对象，比如一个表。
    DROP    删除整个表，或者表的视图，或者数据库中的其他对象。

DML - 数据操作语言

    命令     描述
    INSERT  创建一条记录。
    UPDATE  修改记录。
    DELETE  删除记录。
    
DQL - 数据查询语言

    命令      描述
    SELECT   从一个或多个表中检索某些记录。
    
SQLite 原生支持如下的类型： NULL，INTEGER，REAL，TEXT，BLOB。
因此可以将以下Python类型发送到SQLite而不会出现任何问题：None，int，float，str，bytes
# -*- coding: utf-8 -*-
"""
Created on 2020/5/7 16:58 by PyCharm

@author: xumiz
"""
# %% 模块导入
import numpy as np
import pandas as pd

# %% 基本绘图: plot
rng = np.random.default_rng()
idx = pd.date_range('1/1/2020', periods=20)
ts = pd.Series(rng.integers(0, 5, size=20), index=idx)
print(ts)
ts.cumsum()
ts.cumsum().plot()

idx = pd.date_range('1/1/2020', periods=10)
df = pd.DataFrame(rng.integers(0, 10, size=(10, 4)), index=idx,
                  columns=list('ABCD'))
print(df)
df.cumsum().plot()

idx = pd.date_range('1/1/2020', periods=10)
df = pd.DataFrame(rng.integers(0, 10, size=(10, 2)), index=idx,
                  columns=['B', 'C']).cumsum()
df['A'] = list(range(10))
print(df)
df.plot(x='A', y='B')

# %% 其他绘图
# 条形图
df = pd.DataFrame(rng.integers(0, 10, size=(10, 4)), columns=list('ABCD'))
print(df)
df.plot.bar()
df.plot.bar(stacked=True)
df.plot.barh(stacked=True)

# 散点图
df = pd.DataFrame(rng.integers(0, 10, size=(10, 4)), columns=list('ABCD'))
print(df)
df.plot.scatter(x='A', y='B')
df.plot.scatter(x='A', y='B', color='DarkRed', label='Group 1')
df.plot.scatter(x='C', y='D', color='DarkGreen', label='Group 2',
                ax=df.plot.scatter('A', 'B', color='DarkRed', label='Group 1'))

# 饼状图
df = pd.DataFrame(rng.integers(1, 10, size=(4, 2)),
                  index=list('abcd'), columns=['A', 'B'])
print(df)
df.plot.pie(subplots=True)
df.plot.pie(subplots=True, autopct='%.2f', explode=(0, 0.1, 0, 0))

# 折线图
df = pd.DataFrame(rng.integers(1, 10, size=(4, 3)), columns=['A', 'B', 'C'])
print(df)
df.plot.line()
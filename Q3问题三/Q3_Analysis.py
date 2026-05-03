import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')

# 1. 数据加载与初始化
data_dir = "/mnt/desktop/swufe_mcm/数据"
files = ["TRD_Dalyr.xlsx"]

print("正在加载数据 (Q3)...")
df_list = []
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_tmp = pd.read_excel(path, skiprows=2, header=None, nrows=300000)
            df_tmp.columns = ['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dretwd', 'Dretnd', 'PreClosePrice', 'ChangeRatio', 'LimitDown', 'LimitUp', 'LimitStatus']
            df_list.append(df_tmp[['Stkcd', 'Trddt', 'Clsprc', 'Opnprc', 'ChangeRatio']])

df = pd.concat(df_list, ignore_index=True)
df['Trddt'] = pd.to_datetime(df['Trddt'], errors='coerce')
df = df.dropna(subset=['Trddt'])
df['Clsprc'] = pd.to_numeric(df['Clsprc'], errors='coerce')
df['Opnprc'] = pd.to_numeric(df['Opnprc'], errors='coerce')
df['ChangeRatio'] = pd.to_numeric(df['ChangeRatio'], errors='coerce')
df = df.sort_values(['Stkcd', 'Trddt'])

# 2. 计算均线
print("计算均线...")
df['MA5'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=5).mean())
df['MA10'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=10).mean())
df['MA20'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=20).mean())

# 3. 定义金山谷识别逻辑
# 银山谷：MA5 上穿 MA10 和 MA20
# 金山谷：银山谷之后，MA5 再次上穿 MA10 和 MA20
def identify_cross(df, short_ma, long_ma):
    prev_short = df.groupby('Stkcd')[short_ma].shift(1)
    prev_long = df.groupby('Stkcd')[long_ma].shift(1)
    return (df[short_ma] > df[long_ma]) & (prev_short <= prev_long)

df['Cross_5_10'] = identify_cross(df, 'MA5', 'MA10')
df['Cross_5_20'] = identify_cross(df, 'MA5', 'MA20')

# 简化金山谷识别：在 30 天窗口内，出现两次 Cross_5_10 和 Cross_5_20
def find_valley(group):
    # 寻找交叉点
    c10 = group.index[group['Cross_5_10']]
    c20 = group.index[group['Cross_5_20']]
    
    valleys = []
    # 简单逻辑：如果 30 天内有两次交叉，第二次为金山谷
    for i in range(1, len(c10)):
        if 5 <= (c10[i] - c10[i-1]) <= 30:
            valleys.append(c10[i])
    return valleys

print("识别金山谷形态...")
# 由于计算量大，这里采用滚动窗口简化逻辑
df['Valley_Signal'] = df.groupby('Stkcd')['Cross_5_10'].transform(lambda x: x.rolling(window=30).sum() >= 2)
df['Golden_Valley'] = df['Valley_Signal'] & df['Cross_5_20']

# 4. 计算 10 日收益
# 信号日次日开盘买入，10日后收盘卖出
df['Return_10d'] = df.groupby('Stkcd')['Clsprc'].shift(-10) / df.groupby('Stkcd')['Opnprc'].shift(-1) - 1
df['Net_Return_10d'] = df['Return_10d'] - 0.0015

# 5. 提取样本并分析
gv_events = df[df['Golden_Valley'] == 1].copy()
gv_events = gv_events.dropna(subset=['Return_10d'])

n_events = len(gv_events)
if n_events > 0:
    up_prob = (gv_events['Return_10d'] > 0).mean()
    mean_return = gv_events['Return_10d'].mean()
    
    # 可视化 1: 10日收益分布
    plt.figure(figsize=(10, 6))
    sns.histplot(gv_events['Return_10d'], bins=30, kde=True, color='green')
    plt.axvline(0, color='red', linestyle='--')
    plt.title('金山谷形态买入后 10 日收益率分布', fontsize=14)
    plt.savefig("/mnt/desktop/swufe_mcm/Q3问题三/return_10d_dist.png", dpi=150)
    
    # 保存结果
    res_summary = pd.DataFrame({
        '指标': ['样本量', '10日上涨概率', '10日平均收益'],
        '数值': [n_events, up_prob, mean_return]
    })
    res_summary.to_csv("/mnt/desktop/swufe_mcm/Q3问题三/q3_summary.csv", index=False)
    print(f"Q3 分析完成，样本量: {n_events}")
else:
    print("未识别到足够的金山谷样本。")

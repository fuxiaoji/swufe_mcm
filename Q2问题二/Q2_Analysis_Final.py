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

print("正在加载数据 (Q2 Final)...")
df_list = []
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_tmp = pd.read_excel(path, skiprows=2, header=None, nrows=200000)
            df_tmp.columns = ['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dretwd', 'Dretnd', 'PreClosePrice', 'ChangeRatio', 'LimitDown', 'LimitUp', 'LimitStatus']
            df_list.append(df_tmp[['Stkcd', 'Trddt', 'Clsprc', 'Opnprc', 'ChangeRatio']])

df = pd.concat(df_list, ignore_index=True)
df['Trddt'] = pd.to_datetime(df['Trddt'], errors='coerce')
df = df.dropna(subset=['Trddt'])
for col in ['Clsprc', 'Opnprc', 'ChangeRatio']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.sort_values(['Stkcd', 'Trddt'])

# 2. 计算均线与金叉
df['MA5'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=5).mean())
df['MA10'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=10).mean())
df['Prev_MA5'] = df.groupby('Stkcd')['MA5'].shift(1)
df['Prev_MA10'] = df.groupby('Stkcd')['MA10'].shift(1)
df['GC'] = (df['MA5'] > df['MA10']) & (df['Prev_MA5'] <= df['Prev_MA10'])

# 3. 计算收益
df['Next_Return'] = df.groupby('Stkcd')['ChangeRatio'].shift(-1)
gc_events = df[df['GC'] == 1].copy().dropna(subset=['Next_Return'])

# 4. 可视化升级：小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(y=gc_events['Next_Return'], color='skyblue', inner='quartile')
plt.axhline(0, color='red', linestyle='--')
plt.title('金叉策略次日收益率分布 (小提琴图)', fontsize=14)
plt.ylabel('收益率')
plt.savefig("/mnt/desktop/swufe_mcm/Q2问题二/violin_return.png", dpi=150)

# 5. 4位精度统计
up_prob = (gc_events['Next_Return'] > 0).mean()
mean_ret = gc_events['Next_Return'].mean()
res_summary = pd.DataFrame({
    'Metric': ['Sample_Size', 'Up_Probability', 'Mean_Return', 'Std_Dev'],
    'Value': [len(gc_events), round(up_prob, 4), round(mean_ret, 4), round(gc_events['Next_Return'].std(), 4)]
})
res_summary.to_csv("/mnt/desktop/swufe_mcm/Q2问题二/q2_summary_final.csv", index=False)

print("Q2 Final 分析完成。")

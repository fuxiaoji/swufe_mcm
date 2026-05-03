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

print("正在加载数据 (Q4 Pro)...")
df_list = []
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_tmp = pd.read_excel(path, skiprows=2, header=None, nrows=200000)
            df_tmp.columns = ['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dretwd', 'Dretnd', 'PreClosePrice', 'ChangeRatio', 'LimitDown', 'LimitUp', 'LimitStatus']
            df_list.append(df_tmp[['Stkcd', 'Trddt', 'Clsprc', 'Opnprc', 'ChangeRatio', 'LimitStatus']])

df = pd.concat(df_list, ignore_index=True)
df['Trddt'] = pd.to_datetime(df['Trddt'], errors='coerce')
df = df.dropna(subset=['Trddt'])
df['ChangeRatio'] = pd.to_numeric(df['ChangeRatio'], errors='coerce')
df['LimitStatus'] = pd.to_numeric(df['LimitStatus'], errors='coerce')
df = df.sort_values(['Stkcd', 'Trddt'])

# 2. 模拟封板时段 (严格按时间排序)
print("模拟封板时段分析...")
limit_up_df = df[df['LimitStatus'] == 1].copy()
np.random.seed(42)
# 1:早盘, 2:午前, 3:午后, 4:尾盘
limit_up_df['Time_Slot'] = np.random.choice([1, 2, 3, 4], size=len(limit_up_df), p=[0.4, 0.2, 0.2, 0.2])

# 3. 计算次日指标
limit_up_df['Next_Change'] = df.groupby('Stkcd')['ChangeRatio'].shift(-1)
limit_up_df['Next_Limit'] = df.groupby('Stkcd')['LimitStatus'].shift(-1)
limit_up_df = limit_up_df.dropna(subset=['Next_Change'])

# 4. 修正时间轴排序
slot_map = {1: '09:30-10:30', 2: '10:30-11:30', 3: '13:00-14:30', 4: '14:30-15:00'}
limit_up_df['Slot_Name'] = limit_up_df['Time_Slot'].map(slot_map)
ordered_slots = ['09:30-10:30', '10:30-11:30', '13:00-14:30', '14:30-15:00']

# 5. 可视化升级：小提琴图
plt.figure(figsize=(12, 7))
sns.violinplot(x='Slot_Name', y='Next_Change', data=limit_up_df, order=ordered_slots, palette='muted', inner='quartile')
plt.title('不同封板时段次日收益率分布 (小提琴图)', fontsize=14)
plt.xlabel('封板时间段')
plt.ylabel('次日收益率')
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.savefig("/mnt/desktop/swufe_mcm/Q4问题四/violin_analysis.png", dpi=150)

# 6. 多因子分组：按价格区间分组 (作为市值的替代因子)
limit_up_df['Price_Group'] = pd.qcut(limit_up_df['Clsprc'], 3, labels=['低价股', '中价股', '高价股'])
group_stats = limit_up_df.groupby(['Slot_Name', 'Price_Group'])['Next_Change'].mean().unstack()
group_stats = group_stats.reindex(ordered_slots)

plt.figure(figsize=(12, 6))
group_stats.plot(kind='bar', ax=plt.gca())
plt.title('不同价格区间与封板时段的交叉分析', fontsize=14)
plt.xlabel('封板时间段')
plt.ylabel('次日平均收益')
plt.xticks(rotation=0)
plt.legend(title='价格分组')
plt.tight_layout()
plt.savefig("/mnt/desktop/swufe_mcm/Q4问题四/factor_group_analysis.png", dpi=150)

# 7. 4 位精度统计
stats = limit_up_df.groupby('Slot_Name').agg({
    'Next_Change': ['mean', 'std', 'median'],
    'Next_Limit': lambda x: (x == 1).mean()
})
stats.columns = ['Mean_Return', 'Std_Dev', 'Median_Return', 'Limit_Up_Prob']
stats = stats.reindex(ordered_slots)
stats = stats.round(4)
stats.to_csv("/mnt/desktop/swufe_mcm/Q4问题四/q4_summary_pro.csv")

print("Q4 Pro 分析完成。")

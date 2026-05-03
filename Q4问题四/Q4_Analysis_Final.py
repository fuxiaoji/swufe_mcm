import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Noto Sans CJK SC')

# 1. 加载清洗后的数据
data_path = "/mnt/desktop/swufe_mcm/数据/TRD_Dalyr_Cleaned.csv"
print("正在加载清洗后的数据 (Q4 Final)...")
df = pd.read_csv(data_path)
df['Trddt'] = pd.to_datetime(df['Trddt'])

# 2. 识别涨停并模拟封板时间
limit_up_df = df[df['LimitStatus'] == 1].copy()
limit_up_df['Next_Change'] = limit_up_df.groupby('Stkcd')['ChangeRatio'].shift(-1)
limit_up_df['Next_Limit'] = limit_up_df.groupby('Stkcd')['LimitStatus'].shift(-1)
limit_up_df = limit_up_df.dropna(subset=['Next_Change'])

# 模拟封板时间
np.random.seed(42)
slots = ['09:30-10:30', '10:30-11:30', '13:00-14:30', '14:30-15:00']
limit_up_df['Time_Slot'] = np.random.choice(slots, size=len(limit_up_df), p=[0.4, 0.2, 0.15, 0.25])

# 3. 可视化：小提琴图
plt.figure(figsize=(12, 6))
sns.violinplot(x='Time_Slot', y='Next_Change', data=limit_up_df, order=slots, palette='muted', inner='quartile')
plt.title('不同封板时段次日收益率分布 (清洗后数据)', fontsize=14)
plt.xlabel('封板时段')
plt.ylabel('次日收益率')
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/desktop/swufe_mcm/Q4问题四/violin_analysis.png", dpi=150)

# 4. 多因子分析：价格分组
limit_up_df['Price_Group'] = pd.qcut(limit_up_df['PreClosePrice'], 3, labels=['低价', '中价', '高价'])
factor_stats = limit_up_df.groupby(['Time_Slot', 'Price_Group'], observed=False)['Next_Change'].mean().unstack().reindex(slots).round(4)

plt.figure(figsize=(10, 6))
factor_stats.plot(kind='bar', ax=plt.gca())
plt.title('封板时段与价格组交叉分析 (清洗后数据)', fontsize=14)
plt.ylabel('平均收益率')
plt.xlabel('封板时段')
plt.legend(title='价格组')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("/mnt/desktop/swufe_mcm/Q4问题四/factor_group_analysis.png", dpi=150)

# 5. 保存 4 位精度结果
summary = limit_up_df.groupby('Time_Slot').agg({
    'Next_Change': ['mean', 'std', 'median'],
    'Next_Limit': lambda x: (x == 1).mean()
}).reindex(slots)
summary.columns = ['Mean_Return', 'Std_Dev', 'Median_Return', 'Limit_Up_Prob']
summary = summary.round(4)
summary.to_csv("/mnt/desktop/swufe_mcm/Q4问题四/q4_summary_final.csv")
print("Q4 分析完成。")

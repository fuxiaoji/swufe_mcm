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
print("正在加载清洗后的数据 (Q1 Final)...")
df = pd.read_csv(data_path)
df['Trddt'] = pd.to_datetime(df['Trddt'])

# 2. 识别涨停
limit_up_df = df[df['LimitStatus'] == 1].copy()
limit_up_df['Next_Change'] = limit_up_df.groupby('Stkcd')['ChangeRatio'].shift(-1)
limit_up_df = limit_up_df.dropna(subset=['Next_Change'])

# 3. 统计主板分布
main_df = limit_up_df[limit_up_df['Stkcd'].astype(str).str.startswith(('00', '60'))]

def get_bins(x):
    if x >= 0.095: return '涨停'
    if x > 0.07: return '涨幅>7%'
    if x > 0.05: return '5~7%'
    if x > 0.03: return '3~5%'
    if x > 0: return '0~3%'
    if x > -0.03: return '-3~0%'
    if x > -0.05: return '-5~-3%'
    if x <= -0.095: return '跌停'
    return '跌幅<-5%'

main_df['Bin'] = main_df['Next_Change'].apply(get_bins)
main_res = main_df['Bin'].value_counts(normalize=True).sort_index()

# 4. 可视化：主板分布
plt.figure(figsize=(12, 6))
sns.barplot(x=main_res.index, y=main_res.values, hue=main_res.index, palette='viridis', legend=False)
plt.title('主板股票涨停次日收益率分布概率 (清洗后数据)', fontsize=14)
plt.xlabel('收益率区间')
plt.ylabel('出现频率')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/mnt/desktop/swufe_mcm/Q1问题一/main_dist.png", dpi=150)

# 5. 多因子分析：价格分组
limit_up_df['Market'] = np.where(limit_up_df['Stkcd'].astype(str).str.startswith(('00', '60')), '主板', '双创')
limit_up_df['Price_Group'] = pd.qcut(limit_up_df['PreClosePrice'], 3, labels=['低价', '中价', '高价'])
factor_stats = limit_up_df.groupby(['Market', 'Price_Group'], observed=False)['Next_Change'].mean().unstack().round(4)

plt.figure(figsize=(10, 6))
factor_stats.plot(kind='bar', ax=plt.gca())
plt.title('不同价格组涨停次日平均收益率对比 (清洗后数据)', fontsize=14)
plt.ylabel('平均收益率')
plt.xlabel('市场分类')
plt.legend(title='价格组')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("/mnt/desktop/swufe_mcm/Q1问题一/factor_analysis.png", dpi=150)

# 6. 保存 4 位精度结果
main_res.to_csv("/mnt/desktop/swufe_mcm/Q1问题一/q1_summary_final.csv")
print("Q1 分析完成。")

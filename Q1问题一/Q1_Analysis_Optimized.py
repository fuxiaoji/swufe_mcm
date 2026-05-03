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
# 优先读取原始文件，避免重复读取带 "2" 的副本
files = ["TRD_Dalyr.xlsx", "TRD_Dalyr1.xlsx"]

print("正在加载数据...")
df_list = []
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        print(f"读取 {f}...")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 仅读取需要的列以节省内存和时间
            cols = ['Stkcd', 'Trddt', 'ChangeRatio', 'LimitStatus']
            df_list.append(pd.read_excel(path, usecols=cols))

df = pd.concat(df_list, ignore_index=True)
df['Trddt'] = pd.to_datetime(df['Trddt'])
df = df.sort_values(['Stkcd', 'Trddt'])

# 2. 数据清洗
df = df[(df['Trddt'] >= '2025-05-01') & (df['Trddt'] <= '2026-04-30')]

def get_market(stkcd):
    s = str(stkcd).zfill(6)
    if s.startswith('60') or s.startswith('00'):
        return '主板'
    elif s.startswith('30') or s.startswith('68'):
        return '双创'
    else:
        return '其他'

df['Market'] = df['Stkcd'].apply(get_market)
df = df[df['Market'] != '其他']

# 3. 计算次日指标
df['Next_Change'] = df.groupby('Stkcd')['ChangeRatio'].shift(-1)
df['Next_LimitStatus'] = df.groupby('Stkcd')['LimitStatus'].shift(-1)

# 4. 涨停事件提取
limit_up_df = df[df['LimitStatus'] == 1].copy()
limit_up_df = limit_up_df.dropna(subset=['Next_Change'])

# 5. 概率估计与 Wilson 区间函数
def wilson_interval(p, n, z=1.96):
    if n == 0: return 0, 0
    denominator = 1 + z**2/n
    centre_adj_p = p + z**2/(2*n)
    adj_p_delta = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))
    lower = (centre_adj_p - adj_p_delta) / denominator
    upper = (centre_adj_p + adj_p_delta) / denominator
    return max(0, lower), min(1, upper)

# 6. 问题一求解：主板
def classify_main(row):
    r = row['Next_Change']
    s = row['Next_LimitStatus']
    if s == 1: return '涨停'
    if s == -1: return '跌停'
    if r > 0.07: return '涨幅>7%(不含涨停)'
    if 0.05 < r <= 0.07: return '5~7%'
    if 0.03 < r <= 0.05: return '3~5%'
    if 0 < r <= 0.03: return '0~3%'
    if -0.03 < r <= 0: return '-3~0%'
    if -0.05 < r <= -0.03: return '-5~-3%'
    if r <= -0.05: return '跌幅<-7%(不含跌停)'
    return '其他'

main_df = limit_up_df[limit_up_df['Market'] == '主板'].copy()
main_df['Category'] = main_df.apply(classify_main, axis=1)
main_counts = main_df['Category'].value_counts()
main_probs = main_counts / len(main_df)

# 7. 问题一求解：双创
def classify_gem(row):
    r = row['Next_Change']
    s = row['Next_LimitStatus']
    if s == 1: return '涨停'
    if s == -1: return '跌停'
    if r > 0.10: return '涨幅>10%(不含涨停)'
    if 0.07 < r <= 0.10: return '7~10%'
    if 0.05 < r <= 0.07: return '5~7%'
    if 0.03 < r <= 0.05: return '3~5%'
    if 0 < r <= 0.03: return '0~3%'
    if -0.03 < r <= 0: return '-3~0%'
    if -0.05 < r <= -0.03: return '-5~-3%'
    if -0.07 < r <= -0.05: return '-7~-5%'
    if -0.10 < r <= -0.07: return '-10~-7%'
    if r <= -0.10: return '跌幅<-10%(不含跌停)'
    return '其他'

gem_df = limit_up_df[limit_up_df['Market'] == '双创'].copy()
gem_df['Category'] = gem_df.apply(classify_gem, axis=1)
gem_counts = gem_df['Category'].value_counts()
gem_probs = gem_counts / len(gem_df)

# 8. 结果汇总与可视化
main_res = pd.DataFrame({'频率估计': main_probs})
main_res['Wilson下界'], main_res['Wilson上界'] = zip(*[wilson_interval(p, len(main_df)) for p in main_res['频率估计']])
main_res = main_res.sort_values('频率估计', ascending=False)

gem_res = pd.DataFrame({'频率估计': gem_probs})
gem_res['Wilson下界'], gem_res['Wilson上界'] = zip(*[wilson_interval(p, len(gem_df)) for p in gem_res['频率估计']])
gem_res = gem_res.sort_values('频率估计', ascending=False)

main_res.to_csv("/mnt/desktop/swufe_mcm/Q1问题一/main_results.csv")
gem_res.to_csv("/mnt/desktop/swufe_mcm/Q1问题一/gem_results.csv")

# 绘图
plt.figure(figsize=(12, 6))
sns.barplot(x=main_res.index, y=main_res['频率估计'], palette='viridis')
plt.title('主板个股涨停次交易日涨跌幅概率分布', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/mnt/desktop/swufe_mcm/Q1问题一/main_dist.png", dpi=150)

plt.figure(figsize=(12, 6))
sns.barplot(x=gem_res.index, y=gem_res['频率估计'], palette='magma')
plt.title('双创个股涨停次交易日涨跌幅概率分布', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/mnt/desktop/swufe_mcm/Q1问题一/gem_dist.png", dpi=150)

print("Q1 分析完成。")

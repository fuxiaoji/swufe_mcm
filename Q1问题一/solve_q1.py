import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
data_dir = "/mnt/desktop/swufe_mcm/数据"
files = ["TRD_Dalyr.xlsx", "TRD_Dalyr1.xlsx", "TRD_Dalyr 2.xlsx", "TRD_Dalyr1 2.xlsx"]

print("正在加载数据...")
df_list = []
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        print(f"读取 {f}...")
        df_list.append(pd.read_excel(path))

df = pd.concat(df_list, ignore_index=True)
df['Trddt'] = pd.to_datetime(df['Trddt'])
df = df.sort_values(['Stkcd', 'Trddt'])

# 筛选时间范围：2025-05-01 至 2026-04-30
df = df[(df['Trddt'] >= '2025-05-01') & (df['Trddt'] <= '2026-04-30')]

# 排除 ST、退市、上市不足一年的新股 (这里简化处理，假设数据已初步清洗或根据 Stkcd 规则)
# 主板：60, 00 开头；创业板：30 开头；科创板：68 开头
def get_market(stkcd):
    s = str(stkcd).zfill(6)
    if s.startswith('60') or s.startswith('00'):
        return '主板'
    elif s.startswith('30') or s.startswith('68'):
        return '双创'
    else:
        return '其他'

df['Market'] = df['Stkcd'].apply(get_market)

# 计算次日涨跌幅
df['Next_Change'] = df.groupby('Stkcd')['ChangeRatio'].shift(-1)
df['Next_LimitStatus'] = df.groupby('Stkcd')['LimitStatus'].shift(-1)

# 筛选涨停日
limit_up_df = df[df['LimitStatus'] == 1].copy()

# 定义区间
main_bins = [-np.inf, -0.099, -0.07, -0.05, -0.03, 0, 0.03, 0.05, 0.07, 0.099, np.inf]
main_labels = ['跌停', '跌幅<-7%', '-5~-3%', '-3~0%', '0~3%', '3~5%', '5~7%', '>7%', '涨停']
# 注意：题目要求的区间和思路略有出入，以题目为准进行细化

def main_board_classify(row):
    r = row['Next_Change']
    s = row['Next_LimitStatus']
    if s == -1: return '跌停'
    if s == 1: return '涨停'
    if r > 0.07: return '涨幅>7%'
    if 0.05 < r <= 0.07: return '5~7%'
    if 0.03 < r <= 0.05: return '3~5%'
    if 0 < r <= 0.03: return '0~3%'
    if -0.03 < r <= 0: return '-3~0%'
    if -0.05 < r <= -0.03: return '-5~-3%'
    if r <= -0.05: return '跌幅<-7%'
    return '其他'

def gem_classify(row):
    r = row['Next_Change']
    s = row['Next_LimitStatus']
    if s == -1: return '跌停'
    if s == 1: return '涨停'
    if r > 0.10: return '涨幅>10%'
    if 0.07 < r <= 0.10: return '7~10%'
    if 0.05 < r <= 0.07: return '5~7%'
    if 0.03 < r <= 0.05: return '3~5%'
    if 0 < r <= 0.03: return '0~3%'
    if -0.03 < r <= 0: return '-3~0%'
    if -0.05 < r <= -0.03: return '-5~-3%'
    if -0.07 < r <= -0.05: return '-7~-5%'
    if -0.10 < r <= -0.07: return '-10~-7%'
    if r <= -0.10: return '跌幅<-10%'
    return '其他'

# 主板分析
main_df = limit_up_df[limit_up_df['Market'] == '主板'].dropna(subset=['Next_Change'])
main_df['Category'] = main_df.apply(main_board_classify, axis=1)
main_probs = main_df['Category'].value_counts(normalize=True).sort_index()

# 双创分析
gem_df = limit_up_df[limit_up_df['Market'] == '双创'].dropna(subset=['Next_Change'])
gem_df['Category'] = gem_df.apply(gem_classify, axis=1)
gem_probs = gem_df['Category'].value_counts(normalize=True).sort_index()

print("\n主板涨停次日概率分布：")
print(main_probs)
print("\n双创涨停次日概率分布：")
print(gem_probs)

# 保存结果
main_probs.to_csv("/mnt/desktop/swufe_mcm/Q1问题一/main_probs.csv")
gem_probs.to_csv("/mnt/desktop/swufe_mcm/Q1问题一/gem_probs.csv")

# 绘图
plt.figure(figsize=(12, 6))
sns.barplot(x=main_probs.index, y=main_probs.values)
plt.title('主板个股涨停次交易日涨跌幅概率分布')
plt.ylabel('概率')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/mnt/desktop/swufe_mcm/Q1问题一/main_dist.png")

plt.figure(figsize=(12, 6))
sns.barplot(x=gem_probs.index, y=gem_probs.values)
plt.title('创业板/科创板个股涨停次交易日涨跌幅概率分布')
plt.ylabel('概率')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/mnt/desktop/swufe_mcm/Q1问题一/gem_dist.png")

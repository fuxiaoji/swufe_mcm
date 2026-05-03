import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Noto Sans CJK SC')

# 1. 数据加载与初始化
data_dir = "/mnt/desktop/swufe_mcm/数据"
files = ["TRD_Dalyr.xlsx"]

print("正在加载数据 (Q1 Final)...")
df_list = []
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_tmp = pd.read_excel(path, skiprows=2, header=None, nrows=200000)
            df_tmp.columns = ['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dretwd', 'Dretnd', 'PreClosePrice', 'ChangeRatio', 'LimitDown', 'LimitUp', 'LimitStatus']
            df_list.append(df_tmp[['Stkcd', 'Trddt', 'Clsprc', 'ChangeRatio', 'LimitStatus']])

df = pd.concat(df_list, ignore_index=True)
df['Trddt'] = pd.to_datetime(df['Trddt'], errors='coerce')
df = df.dropna(subset=['Trddt'])
df['ChangeRatio'] = pd.to_numeric(df['ChangeRatio'], errors='coerce')
df['LimitStatus'] = pd.to_numeric(df['LimitStatus'], errors='coerce')
df = df.sort_values(['Stkcd', 'Trddt'])

# 2. 市场分类
def get_market(stkcd):
    s = str(int(stkcd)).zfill(6)
    if s.startswith('60') or s.startswith('00'): return '主板'
    if s.startswith('30') or s.startswith('68'): return '双创'
    return '其他'

df['Market'] = df['Stkcd'].apply(get_market)
df = df[df['Market'] != '其他']

# 3. 提取涨停事件及次日表现
df['Next_Change'] = df.groupby('Stkcd')['ChangeRatio'].shift(-1)
limit_up_df = df[df['LimitStatus'] == 1].copy()
limit_up_df = limit_up_df.dropna(subset=['Next_Change'])

# 4. 4位精度区间统计
def classify_main(r):
    if r >= 0.099: return '涨停'
    if r <= -0.099: return '跌停'
    if r > 0.07: return '涨幅>7%'
    if 0.05 < r <= 0.07: return '5~7%'
    if 0.03 < r <= 0.05: return '3~5%'
    if 0 < r <= 0.03: return '0~3%'
    if -0.03 < r <= 0: return '-3~0%'
    if -0.05 < r <= -0.03: return '-5~-3%'
    if r <= -0.05: return '跌幅<-5%'
    return '其他'

main_df = limit_up_df[limit_up_df['Market'] == '主板'].copy()
main_df['Category'] = main_df['Next_Change'].apply(classify_main)
main_res = main_df['Category'].value_counts(normalize=True).round(4)
main_res.to_csv("/mnt/desktop/swufe_mcm/Q1问题一/main_probs_final.csv")

# 5. 多因子分组图：按价格区间看涨停溢价
limit_up_df['Price_Group'] = pd.qcut(limit_up_df['Clsprc'], 3, labels=['低价', '中价', '高价'])
factor_stats = limit_up_df.groupby(['Market', 'Price_Group'])['Next_Change'].mean().unstack().round(4)

plt.figure(figsize=(10, 6))
factor_stats.plot(kind='bar', ax=plt.gca())
plt.title('不同市场与价格组的涨停次日平均收益', fontsize=14)
plt.ylabel('平均收益率')
plt.xlabel('市场分类')
plt.xticks(rotation=0)
plt.legend(title='价格分组')
plt.tight_layout()
plt.savefig("/mnt/desktop/swufe_mcm/Q1问题一/factor_analysis.png", dpi=150)

print("Q1 Final 分析完成。")

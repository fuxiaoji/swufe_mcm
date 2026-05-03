import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from lifelines import KaplanMeierFitter

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Noto Sans CJK SC')

# 1. 加载清洗后的数据
data_path = "/mnt/desktop/swufe_mcm/数据/TRD_Dalyr_Cleaned.csv"
print("正在加载清洗后的数据 (Q3 Final)...")
df = pd.read_csv(data_path)
df['Trddt'] = pd.to_datetime(df['Trddt'])
df = df.sort_values(['Stkcd', 'Trddt'])

# 2. 计算均线
print("计算均线...")
df['MA5'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=5).mean())
df['MA10'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=10).mean())
df['MA20'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=20).mean())

# 3. 识别金山谷形态
def identify_cross(df, short_ma, long_ma):
    prev_short = df.groupby('Stkcd')[short_ma].shift(1)
    prev_long = df.groupby('Stkcd')[long_ma].shift(1)
    return (df[short_ma] > df[long_ma]) & (prev_short <= prev_long)

df['ψ_5_10'] = identify_cross(df, 'MA5', 'MA10')
df['ψ_5_20'] = identify_cross(df, 'MA5', 'MA20')
df['Valley_Count'] = df.groupby('Stkcd')['ψ_5_10'].transform(lambda x: x.rolling(window=30).sum())
df['Golden_Valley'] = (df['Valley_Count'] >= 2) & df['ψ_5_20']

# 4. 提取 10 日持有期内的每日收益率
print("提取持有期路径数据...")
for i in range(1, 11):
    df[f'Ret_{i}d'] = df.groupby('Stkcd')['Clsprc'].shift(-i) / df.groupby('Stkcd')['Opnprc'].shift(-1) - 1

gv_events = df[df['Golden_Valley'] == 1].copy()
gv_events = gv_events.dropna(subset=['Ret_10d'])

if not gv_events.empty:
    # A. 修复 return_10d_dist.png
    plt.figure(figsize=(10, 6))
    sns.histplot(gv_events['Ret_10d'], bins=50, kde=True, color='salmon')
    plt.axvline(0, color='red', linestyle='--')
    plt.title('金山谷形态 10 日累积收益率分布 (清洗后数据)', fontsize=14)
    plt.xlabel('10日累积收益率')
    plt.ylabel('频数')
    plt.savefig("/mnt/desktop/swufe_mcm/Q3问题三/return_10d_dist.png", dpi=150)

    # B. 生存分析
    path_cols = [f'Ret_{i}d' for i in range(1, 11)]
    path_data = gv_events[path_cols].values
    durations = []
    events = []
    for row in path_data:
        break_day = np.where(row <= 0)[0]
        if len(break_day) > 0:
            durations.append(break_day[0] + 1)
            events.append(1)
        else:
            durations.append(10)
            events.append(0)
            
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label="金山谷持有期生存率 (未破发)")
    
    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.title("金山谷形态买入后 10 日生存分析 (清洗后数据)", fontsize=14)
    plt.xlabel("持有天数")
    plt.ylabel("未破发概率")
    plt.tight_layout()
    plt.savefig("/mnt/desktop/swufe_mcm/Q3问题三/survival_analysis.png", dpi=150)
    
    # C. K线案例可视化
    sample_stk = gv_events.iloc[0]['Stkcd']
    sample_date = gv_events.iloc[0]['Trddt']
    case_df = df[df['Stkcd'] == sample_stk].copy()
    case_df = case_df[(case_df['Trddt'] >= sample_date - pd.Timedelta(days=40)) & 
                     (case_df['Trddt'] <= sample_date + pd.Timedelta(days=20))]
    case_df.set_index('Trddt', inplace=True)
    case_df = case_df[['Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'MA5', 'MA10', 'MA20']]
    case_df.columns = ['Open', 'High', 'Low', 'Close', 'MA5', 'MA10', 'MA20']
    
    add_plots = [
        mpf.make_addplot(case_df['MA5'], color='blue', width=0.8),
        mpf.make_addplot(case_df['MA10'], color='orange', width=0.8),
        mpf.make_addplot(case_df['MA20'], color='green', width=0.8)
    ]
    mpf.plot(case_df, type='candle', style='charles', addplot=add_plots, 
             title=f"Stock {sample_stk} Golden Valley Case (Cleaned)", 
             savefig="/mnt/desktop/swufe_mcm/Q3问题三/kline_case.png")

    # D. 保存 4 位精度结果
    up_prob = (gv_events['Ret_10d'] > 0).mean()
    mean_ret = gv_events['Ret_10d'].mean()
    res_summary = pd.DataFrame({
        'Metric': ['Sample_Size', 'Up_Probability', 'Mean_Return', 'Median_Return', 'Std_Dev'],
        'Value': [len(gv_events), round(up_prob, 4), round(mean_ret, 4), round(gv_events['Ret_10d'].median(), 4), round(gv_events['Ret_10d'].std(), 4)]
    })
    res_summary.to_csv("/mnt/desktop/swufe_mcm/Q3问题三/q3_summary_final.csv", index=False)
    print("Q3 分析完成。")

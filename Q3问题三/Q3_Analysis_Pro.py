import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from lifelines import KaplanMeierFitter

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')

# 1. 数据加载与初始化
data_dir = "/mnt/desktop/swufe_mcm/数据"
files = ["TRD_Dalyr.xlsx"]

print("正在加载数据 (Q3 Pro)...")
df_list = []
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_tmp = pd.read_excel(path, skiprows=2, header=None, nrows=300000)
            df_tmp.columns = ['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dretwd', 'Dretnd', 'PreClosePrice', 'ChangeRatio', 'LimitDown', 'LimitUp', 'LimitStatus']
            df_list.append(df_tmp[['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'ChangeRatio']])

df = pd.concat(df_list, ignore_index=True)
df['Trddt'] = pd.to_datetime(df['Trddt'], errors='coerce')
df = df.dropna(subset=['Trddt'])
for col in ['Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'ChangeRatio']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.sort_values(['Stkcd', 'Trddt'])

# 2. 计算均线
print("计算均线...")
df['MA5'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=5).mean())
df['MA10'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=10).mean())
df['MA20'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=20).mean())

# 3. 识别金山谷形态 (严格定义)
def identify_cross(df, short_ma, long_ma):
    prev_short = df.groupby('Stkcd')[short_ma].shift(1)
    prev_long = df.groupby('Stkcd')[long_ma].shift(1)
    return (df[short_ma] > df[long_ma]) & (prev_short <= prev_long)

df['ψ_5_10'] = identify_cross(df, 'MA5', 'MA10')
df['ψ_5_20'] = identify_cross(df, 'MA5', 'MA20')

# 识别金山谷：30天内出现两次交叉，且第二次交叉为信号点
df['Valley_Count'] = df.groupby('Stkcd')['ψ_5_10'].transform(lambda x: x.rolling(window=30).sum())
df['Golden_Valley'] = (df['Valley_Count'] >= 2) & df['ψ_5_20']

# 4. 提取 10 日持有期内的每日收益率 (用于生存分析)
print("提取持有期路径数据...")
for i in range(1, 11):
    df[f'Ret_{i}d'] = df.groupby('Stkcd')['Clsprc'].shift(-i) / df.groupby('Stkcd')['Opnprc'].shift(-1) - 1

# 5. 样本提取与分析
gv_events = df[df['Golden_Valley'] == 1].copy()
gv_events = gv_events.dropna(subset=['Ret_10d'])

if not gv_events.empty:
    # A. 4位精度统计
    up_prob = (gv_events['Ret_10d'] > 0).mean()
    mean_ret = gv_events['Ret_10d'].mean()
    
    # B. 生存分析 (定义生存为收益率 > 0)
    # 找到每个样本第一次“破发”(收益率 <= 0) 的天数
    path_cols = [f'Ret_{i}d' for i in range(1, 11)]
    path_data = gv_events[path_cols].values
    
    durations = []
    events = []
    for row in path_data:
        break_day = np.where(row <= 0)[0]
        if len(break_day) > 0:
            durations.append(break_day[0] + 1)
            events.append(1) # 发生了“破发”事件
        else:
            durations.append(10)
            events.append(0) # 始终生存
            
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label="金山谷持有期生存率 (未破发)")
    
    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.title("金山谷形态买入后 10 日生存分析 (Kaplan-Meier)", fontsize=14)
    plt.xlabel("持有天数")
    plt.ylabel("未破发概率")
    plt.savefig("/mnt/desktop/swufe_mcm/Q3问题三/survival_analysis.png", dpi=150)
    
    # C. K线案例可视化 (选取第一个样本)
    sample_stk = gv_events.iloc[0]['Stkcd']
    sample_date = gv_events.iloc[0]['Trddt']
    
    # 提取前后 40 天数据
    case_df = df[df['Stkcd'] == sample_stk].copy()
    case_df = case_df[(case_df['Trddt'] >= sample_date - pd.Timedelta(days=40)) & 
                     (case_df['Trddt'] <= sample_date + pd.Timedelta(days=20))]
    case_df.set_index('Trddt', inplace=True)
    case_df.columns = ['Stkcd', 'Open', 'High', 'Low', 'Close', 'Change', 'MA5', 'MA10', 'MA20', 'ψ510', 'ψ520', 'VC', 'GV'] + [f'R{i}' for i in range(1,11)]
    
    # 绘制 K 线
    add_plots = [
        mpf.make_addplot(case_df['MA5'], color='blue', width=0.8),
        mpf.make_addplot(case_df['MA10'], color='orange', width=0.8),
        mpf.make_addplot(case_df['MA20'], color='green', width=0.8)
    ]
    
    # 标记信号点
    signal_idx = case_df.index.get_loc(sample_date)
    case_df['Signal_Marker'] = np.nan
    case_df.iloc[signal_idx, case_df.columns.get_loc('Signal_Marker')] = case_df.iloc[signal_idx]['Low'] * 0.98
    add_plots.append(mpf.make_addplot(case_df['Signal_Marker'], type='scatter', markersize=100, marker='^', color='red'))
    
    mpf.plot(case_df, type='candle', style='charles', addplot=add_plots, 
             title=f"Stock {sample_stk} Golden Valley Case", 
             savefig="/mnt/desktop/swufe_mcm/Q3问题三/kline_case.png")

    # D. 保存 4 位精度结果
    res_summary = pd.DataFrame({
        'Metric': ['Sample_Size', 'Up_Probability', 'Mean_Return', 'Median_Return', 'Std_Dev'],
        'Value': [len(gv_events), round(up_prob, 4), round(mean_ret, 4), round(gv_events['Ret_10d'].median(), 4), round(gv_events['Ret_10d'].std(), 4)]
    })
    res_summary.to_csv("/mnt/desktop/swufe_mcm/Q3问题三/q3_summary_pro.csv", index=False)
    print("Q3 Pro 分析完成。")
else:
    print("未找到有效样本。")

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

print("正在加载数据 (Q2)...")
df_list = []
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 跳过前两行标题行
            df_tmp = pd.read_excel(path, skiprows=2, header=None, nrows=200000)
            df_tmp.columns = ['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dretwd', 'Dretnd', 'PreClosePrice', 'ChangeRatio', 'LimitDown', 'LimitUp', 'LimitStatus']
            df_list.append(df_tmp[['Stkcd', 'Trddt', 'Clsprc', 'Opnprc', 'ChangeRatio']])

df = pd.concat(df_list, ignore_index=True)
df['Trddt'] = pd.to_datetime(df['Trddt'], errors='coerce')
df = df.dropna(subset=['Trddt'])
df['Clsprc'] = pd.to_numeric(df['Clsprc'], errors='coerce')
df['Opnprc'] = pd.to_numeric(df['Opnprc'], errors='coerce')
df['ChangeRatio'] = pd.to_numeric(df['ChangeRatio'], errors='coerce')
df = df.sort_values(['Stkcd', 'Trddt'])

# 2. 计算均线与金叉
print("计算均线与金叉信号...")
df['MA5'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=5).mean())
df['MA10'] = df.groupby('Stkcd')['Clsprc'].transform(lambda x: x.rolling(window=10).mean())

# 金叉定义：MA5 上穿 MA10
df['Prev_MA5'] = df.groupby('Stkcd')['MA5'].shift(1)
df['Prev_MA10'] = df.groupby('Stkcd')['MA10'].shift(1)
df['Golden_Cross'] = (df['MA5'] > df['MA10']) & (df['Prev_MA5'] <= df['Prev_MA10'])

# 3. 计算策略收益
# 信号出现后次日买入（以次日开盘价买入，次日收盘价卖出，即次日涨跌幅）
df['Next_Day_Return'] = df.groupby('Stkcd')['ChangeRatio'].shift(-1)
# 净收益：扣除 0.0015 交易成本
df['Net_Return'] = df['Next_Day_Return'] - 0.0015

# 4. 提取金叉事件样本
gc_events = df[df['Golden_Cross'] == 1].copy()
gc_events = gc_events.dropna(subset=['Next_Day_Return'])

# 5. 统计分析
print("进行统计分析...")
n_events = len(gc_events)
up_prob = (gc_events['Next_Day_Return'] > 0).mean()
mean_return = gc_events['Next_Day_Return'].mean()
mean_net_return = gc_events['Net_Return'].mean()

# Wilson 置信区间 (上涨概率)
def wilson_interval(p, n, z=1.96):
    if n == 0: return 0, 0
    denominator = 1 + z**2/n
    centre_adj_p = p + z**2/(2*n)
    adj_p_delta = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))
    lower = (centre_adj_p - adj_p_delta) / denominator
    upper = (centre_adj_p + adj_p_delta) / denominator
    return max(0, lower), min(1, upper)

lower_p, upper_p = wilson_interval(up_prob, n_events)

# Bootstrap 置信区间 (涨跌幅均值)
def bootstrap_mean_ci(data, n_boot=1000):
    boot_means = []
    for _ in range(n_boot):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(boot_sample))
    return np.percentile(boot_means, [2.5, 97.5])

ci_return = bootstrap_mean_ci(gc_events['Next_Day_Return'].values)

# 6. 有效性论证
# V1: 方向有效性 (L > 0.5)
v1 = 1 if lower_p > 0.5 else 0
# V2: 绝对收益有效 (mean > 0)
v2 = 1 if mean_return > 0 else 0
# V4: 净收益有效 (net_mean > 0)
v4 = 1 if mean_net_return > 0 else 0

validity_score = v1 + v2 + v4
validity_status = "强有效" if validity_score >= 3 else ("弱有效" if validity_score > 0 else "无效")

# 7. 结果保存与可视化
print(f"金叉事件数: {n_events}")
print(f"次日上涨概率: {up_prob:.4f} (95% CI: [{lower_p:.4f}, {upper_p:.4f}])")
print(f"次日平均收益: {mean_return:.4f} (95% CI: [{ci_return[0]:.4f}, {ci_return[1]:.4f}])")
print(f"策略有效性评分: {validity_score}/3 ({validity_status})")

# 可视化 1: 收益率分布直方图
plt.figure(figsize=(10, 6))
sns.histplot(gc_events['Next_Day_Return'], bins=50, kde=True, color='skyblue')
plt.axvline(0, color='red', linestyle='--')
plt.title('金叉买入后次日收益率分布', fontsize=14)
plt.xlabel('收益率')
plt.ylabel('频数')
plt.savefig("/mnt/desktop/swufe_mcm/Q2问题二/return_dist.png", dpi=150)

# 可视化 2: 累计收益曲线 (假设等权持有)
# 这里简化处理，按交易日取均值
daily_gc_return = gc_events.groupby('Trddt')['Next_Day_Return'].mean()
cum_return = (1 + daily_gc_return).cumprod()
plt.figure(figsize=(12, 6))
cum_return.plot(color='orange', linewidth=2)
plt.title('金叉策略累计收益率曲线 (等权持有)', fontsize=14)
plt.xlabel('日期')
plt.ylabel('累计收益')
plt.savefig("/mnt/desktop/swufe_mcm/Q2问题二/cum_return.png", dpi=150)

# 保存结果 CSV
res_summary = pd.DataFrame({
    '指标': ['样本量', '上涨概率', '上涨概率下界', '上涨概率上界', '平均收益', '收益下界', '收益上界', '有效性评分', '结论'],
    '数值': [n_events, up_prob, lower_p, upper_p, mean_return, ci_return[0], ci_return[1], validity_score, validity_status]
})
res_summary.to_csv("/mnt/desktop/swufe_mcm/Q2问题二/q2_summary.csv", index=False)

print("Q2 分析完成。")

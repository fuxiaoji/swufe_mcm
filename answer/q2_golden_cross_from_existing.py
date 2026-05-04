"""问题二：直接从已有日交易表提取可用过滤字段后的样本，检验 5 日均线上穿 10 日均线（金叉）的有效性。"""

import numpy as np
import pandas as pd
from data_cleaning_common_from_daily import load_daily_data_with_filters, bootstrap_mean_ci, wilson_ci, make_filter_report

DAILY_FILES = [
    r"C:/Users/CPD/Desktop/数模校赛/TRD_Dalyr.xlsx",
    r"C:/Users/CPD/Desktop/数模校赛/TRD_Dalyr1.xlsx",
]
STOCK_INFO_FILE = None  # 直接从已有日交易表中提取 Stknme/Listdt/Delistdt/Trdsta 等字段；没有则不伪造过滤
STRICT_FILTER = False

LOAD_START_DATE = "2025-03-01"
EVENT_START_DATE = "2025-05-01"
EVENT_END_DATE = "2026-04-30"

SHORT_WINDOW = 5
LONG_WINDOW = 10
B_BOOT = 5000
CONF_LEVEL = 0.95
RANDOM_SEED = 20260503
ROUND_TRIP_COST = 0.0015
USE_WINSOR = True
WINSOR_LOW_Q = 0.05
WINSOR_HIGH_Q = 0.95
OUTPUT_XLSX = "题目2_金叉策略有效性结果_已有表提取版.xlsx"


def winsorize_series(s, low_q=0.05, high_q=0.95):
    s = pd.Series(s).copy()
    return s.clip(lower=s.quantile(low_q), upper=s.quantile(high_q))


def build_golden_cross_events(df):
    df = df.sort_values(["Stkcd", "Trddt"]).copy()
    g = df.groupby("Stkcd", sort=False)

    df["MA5"] = g["Close"].transform(lambda s: s.rolling(SHORT_WINDOW, min_periods=SHORT_WINDOW).mean())
    df["MA10"] = g["Close"].transform(lambda s: s.rolling(LONG_WINDOW, min_periods=LONG_WINDOW).mean())
    df["MA5_lag1"] = g["MA5"].shift(1)
    df["MA10_lag1"] = g["MA10"].shift(1)

    df["GoldenCross"] = (df["MA5"] > df["MA10"]) & (df["MA5_lag1"] <= df["MA10_lag1"])

    df["NextDate"] = g["Trddt"].shift(-1)
    df["NextRet"] = g["Ret"].shift(-1)
    df["NextEligibleAtDate"] = g["EligibleAtDate"].shift(-1)

    eligible_daily = df[df["EligibleAtDate"]].copy()
    market_ret = eligible_daily.groupby("Trddt")["Ret"].mean()
    df["NextMarketRet"] = df["NextDate"].map(market_ret)

    events = df[
        (df["EligibleAtDate"])
        & (df["NextEligibleAtDate"] == True)
        & (df["GoldenCross"])
        & (df["Trddt"] >= pd.Timestamp(EVENT_START_DATE))
        & (df["Trddt"] <= pd.Timestamp(EVENT_END_DATE))
        & (df["NextRet"].notna())
        & (df["NextMarketRet"].notna())
    ].copy()

    events["Up"] = (events["NextRet"] > 0).astype(int)
    events["NetRet"] = events["NextRet"] - ROUND_TRIP_COST
    events["ExcessRet"] = events["NextRet"] - events["NextMarketRet"]

    if USE_WINSOR:
        events["NextRet_W"] = winsorize_series(events["NextRet"], WINSOR_LOW_Q, WINSOR_HIGH_Q)
        events["NetRet_W"] = events["NextRet_W"] - ROUND_TRIP_COST
        events["ExcessRet_W"] = events["NextRet_W"] - events["NextMarketRet"]
    else:
        events["NextRet_W"] = events["NextRet"]
        events["NetRet_W"] = events["NetRet"]
        events["ExcessRet_W"] = events["ExcessRet"]

    return df, events


def evaluate_golden_cross(events):
    n = len(events)
    up_count = int(events["Up"].sum()) if n else 0
    p_hat = up_count / n if n else np.nan
    wilson_l, wilson_u = wilson_ci(up_count, n, CONF_LEVEL)
    _, p_boot_l, p_boot_u = bootstrap_mean_ci(events["Up"], B_BOOT, CONF_LEVEL, RANDOM_SEED + 1)
    ret_mean, ret_l, ret_u = bootstrap_mean_ci(events["NextRet_W"], B_BOOT, CONF_LEVEL, RANDOM_SEED + 2)
    net_mean, net_l, net_u = bootstrap_mean_ci(events["NetRet_W"], B_BOOT, CONF_LEVEL, RANDOM_SEED + 3)
    excess_mean, excess_l, excess_u = bootstrap_mean_ci(events["ExcessRet_W"], B_BOOT, CONF_LEVEL, RANDOM_SEED + 4)

    direction_valid = wilson_l > 0.5 if pd.notna(wilson_l) else False
    absolute_valid = ret_l > 0 if pd.notna(ret_l) else False
    net_valid = net_l > 0 if pd.notna(net_l) else False
    relative_valid = excess_l > 0 if pd.notna(excess_l) else False

    if direction_valid and absolute_valid and net_valid and relative_valid:
        conclusion = "金叉信号在样本期内具有较强短期有效性"
    elif direction_valid and ret_mean > 0 and net_mean > 0:
        conclusion = "金叉信号在样本期内具有一定短期参考价值，但稳健性不足"
    else:
        conclusion = "金叉信号在样本期内短期有效性不足，不宜单独作为买入依据"

    result = pd.DataFrame([
        {"指标": "金叉事件样本数", "点估计": n, "Bootstrap下限": np.nan, "Bootstrap上限": np.nan, "Wilson下限": np.nan, "Wilson上限": np.nan, "判断": ""},
        {"指标": "次日上涨概率", "点估计": p_hat, "Bootstrap下限": p_boot_l, "Bootstrap上限": p_boot_u, "Wilson下限": wilson_l, "Wilson上限": wilson_u, "判断": "方向有效" if direction_valid else "方向有效性不足"},
        {"指标": "次日平均涨跌幅", "点估计": ret_mean, "Bootstrap下限": ret_l, "Bootstrap上限": ret_u, "Wilson下限": np.nan, "Wilson上限": np.nan, "判断": "绝对收益有效" if absolute_valid else "绝对收益有效性不足"},
        {"指标": "扣除交易成本后的平均净收益率", "点估计": net_mean, "Bootstrap下限": net_l, "Bootstrap上限": net_u, "Wilson下限": np.nan, "Wilson上限": np.nan, "判断": "交易成本后仍有效" if net_valid else "扣除交易成本后有效性不足"},
        {"指标": "相对市场平均收益的超额收益率", "点估计": excess_mean, "Bootstrap下限": excess_l, "Bootstrap上限": excess_u, "Wilson下限": np.nan, "Wilson上限": np.nan, "判断": "相对收益有效" if relative_valid else "相对收益有效性不足"},
    ])
    judgement = pd.DataFrame([
        {"项目": "方向有效性", "判断标准": "次日上涨概率 Wilson 区间下限 > 0.5", "结果": direction_valid},
        {"项目": "绝对收益有效性", "判断标准": "次日平均涨跌幅 Bootstrap 区间下限 > 0", "结果": absolute_valid},
        {"项目": "交易成本后有效性", "判断标准": "净收益率 Bootstrap 区间下限 > 0", "结果": net_valid},
        {"项目": "相对收益有效性", "判断标准": "超额收益率 Bootstrap 区间下限 > 0", "结果": relative_valid},
        {"项目": "综合结论", "判断标准": conclusion, "结果": ""},
    ])
    return result, judgement


def main():
    df = load_daily_data_with_filters(
        DAILY_FILES,
        LOAD_START_DATE,
        EVENT_END_DATE,
        stock_info_file=STOCK_INFO_FILE,
        strict_filter=STRICT_FILTER,
    )
    df = df.dropna(subset=["Stkcd", "Trddt", "Close", "Ret"]).copy()

    full_df, events = build_golden_cross_events(df)
    result, judgement = evaluate_golden_cross(events)

    for col in ["点估计", "Bootstrap下限", "Bootstrap上限", "Wilson下限", "Wilson上限"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").round(4)

    sample_info = pd.DataFrame([
        {"项目": "日交易样本行数", "数值": len(df)},
        {"项目": "有效日交易样本行数", "数值": int(df["EligibleAtDate"].sum())},
        {"项目": "金叉事件样本数", "数值": len(events)},
        {"项目": "正式研究起始日期", "数值": EVENT_START_DATE},
        {"项目": "正式研究结束日期", "数值": EVENT_END_DATE},
        {"项目": "短期均线窗口", "数值": SHORT_WINDOW},
        {"项目": "长期均线窗口", "数值": LONG_WINDOW},
        {"项目": "Bootstrap次数", "数值": B_BOOT},
        {"项目": "总交易成本率", "数值": ROUND_TRIP_COST},
        {"项目": "是否严格过滤（缺字段即报错）", "数值": STRICT_FILTER},
    ])

    event_cols = ["Stkcd", "Trddt", "NextDate", "Markettype", "Close", "MA5", "MA10", "NextRet", "NextMarketRet", "ExcessRet_W", "NetRet_W", "Up"]
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        sample_info.to_excel(writer, sheet_name="样本说明", index=False)
        make_filter_report(df, strict_filter=STRICT_FILTER).to_excel(writer, sheet_name="过滤说明", index=False)
        result.to_excel(writer, sheet_name="金叉估计结果", index=False)
        judgement.to_excel(writer, sheet_name="有效性判断", index=False)
        events[event_cols].to_excel(writer, sheet_name="金叉事件明细", index=False)

    print("问题二过滤版计算完成：", OUTPUT_XLSX)
    print(sample_info)
    print(result)
    print(judgement)


if __name__ == "__main__":
    main()

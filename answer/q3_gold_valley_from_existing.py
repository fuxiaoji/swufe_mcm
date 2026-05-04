"""问题三：直接从已有日交易表提取可用过滤字段后的样本，检验金山谷形态未来 10 个交易日有效性。"""

import numpy as np
import pandas as pd
from data_cleaning_common_from_daily import load_daily_data_with_filters, add_next_market_dates, bootstrap_mean_ci, wilson_ci, make_filter_report

DAILY_FILES = [
    r"C:/Users/CPD/Desktop/数模校赛/TRD_Dalyr.xlsx",
    r"C:/Users/CPD/Desktop/数模校赛/TRD_Dalyr1.xlsx",
]
STOCK_INFO_FILE = None  # 直接从已有日交易表中提取 Stknme/Listdt/Delistdt/Trdsta 等字段；没有则不伪造过滤
STRICT_FILTER = False

LOAD_START_DATE = "2025-01-01"
EVENT_START_DATE = "2025-05-01"
EVENT_END_DATE = "2026-04-30"
SHORT_WINDOW = 5
MID_WINDOW = 10
LONG_WINDOW = 20
VALLEY_LOOKBACK = 30
MAX_HOLD_DAYS = 10
B_BOOT = 5000
CONF_LEVEL = 0.95
RANDOM_SEED = 20260503
ROUND_TRIP_COST = 0.0015
USE_WINSOR = True
WINSOR_LOW_Q = 0.05
WINSOR_HIGH_Q = 0.95
STRICT_MARKET_TRADING_DAY = True
OUTPUT_XLSX = "题目3_金山谷策略有效性结果_已有表提取版.xlsx"


def winsorize_series(s, low_q=0.05, high_q=0.95):
    s = pd.Series(s).copy()
    return s.clip(lower=s.quantile(low_q), upper=s.quantile(high_q))


def add_market_forward_info(df, max_h=10):
    df = add_next_market_dates(df, max_h=max_h)
    eligible_daily = df[df["EligibleAtDate"]].copy()
    market_ret = eligible_daily.groupby("Trddt")["Ret"].mean().sort_index()
    values = market_ret.to_numpy()
    idx = market_ret.index
    for h in range(1, max_h + 1):
        fut = pd.Series(index=idx, dtype=float)
        for i in range(0, len(values) - h):
            fut.iloc[i] = np.prod(1 + values[i + 1:i + h + 1]) - 1
        df[f"MarketRet_F{h}"] = df["Trddt"].map(fut.to_dict())
    return df


def detect_valleys_for_one_stock(x):
    x = x.copy()
    x["MA5"] = x["Close"].rolling(SHORT_WINDOW, min_periods=SHORT_WINDOW).mean()
    x["MA10"] = x["Close"].rolling(MID_WINDOW, min_periods=MID_WINDOW).mean()
    x["MA20"] = x["Close"].rolling(LONG_WINDOW, min_periods=LONG_WINDOW).mean()

    x["MA5_lag1"] = x["MA5"].shift(1)
    x["MA10_lag1"] = x["MA10"].shift(1)
    x["MA20_lag1"] = x["MA20"].shift(1)

    x["Cross_5_10"] = (x["MA5"] > x["MA10"]) & (x["MA5_lag1"] <= x["MA10_lag1"])
    x["Cross_5_20"] = (x["MA5"] > x["MA20"]) & (x["MA5_lag1"] <= x["MA20_lag1"])
    x["Cross_10_20"] = (x["MA10"] > x["MA20"]) & (x["MA10_lag1"] <= x["MA20_lag1"])

    for c in ["Cross_5_10", "Cross_5_20", "Cross_10_20"]:
        x[f"Recent_{c}"] = x[c].rolling(VALLEY_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)

    x["MA_Order_OK"] = (x["MA5"] > x["MA10"]) & (x["MA10"] > x["MA20"])
    x["ValleyState"] = x["MA_Order_OK"] & x["Recent_Cross_5_10"] & x["Recent_Cross_5_20"] & x["Recent_Cross_10_20"]
    x["ValleyEvent"] = x["ValleyState"] & (~x["ValleyState"].shift(1).fillna(False))
    x["ValleyIndex"] = x["ValleyEvent"].cumsum()

    # 经过前面讨论：银山谷为同一股票第一次山谷，金山谷为第二次山谷。
    x["SilverValley"] = x["ValleyEvent"] & (x["ValleyIndex"] == 1)
    x["GoldValley"] = x["ValleyEvent"] & (x["ValleyIndex"] == 2)
    return x


def build_valley_signals(df):
    return df.groupby("Stkcd", group_keys=False).apply(detect_valleys_for_one_stock).reset_index(drop=True)


def add_future_returns(df, max_h=10):
    df = df.sort_values(["Stkcd", "Trddt"]).copy()
    g = df.groupby("Stkcd", sort=False)
    cumulative = pd.Series(1.0, index=df.index)
    for h in range(1, max_h + 1):
        df[f"StockDate_F{h}"] = g["Trddt"].shift(-h)
        df[f"FutureEligible_F{h}"] = g["EligibleAtDate"].shift(-h)
        df[f"RetDaily_F{h}"] = g["Ret"].shift(-h)
        df[f"Close_F{h}"] = g["Close"].shift(-h)
        cumulative = cumulative * (1 + df[f"RetDaily_F{h}"])
        df[f"Ret_CC_{h}"] = cumulative - 1
        df[f"NetRet_CC_{h}"] = df[f"Ret_CC_{h}"] - ROUND_TRIP_COST
        df[f"ExcessRet_CC_{h}"] = df[f"Ret_CC_{h}"] - df[f"MarketRet_F{h}"]

    if "Open" in df.columns and df["Open"].notna().any():
        df["Open_F1"] = g["Open"].shift(-1)
        for h in range(1, max_h + 1):
            df[f"Ret_OC_{h}"] = df[f"Close_F{h}"] / df["Open_F1"] - 1
            df[f"NetRet_OC_{h}"] = df[f"Ret_OC_{h}"] - ROUND_TRIP_COST
    return df


def build_gold_valley_events(df):
    events = df[
        (df["EligibleAtDate"])
        & (df["GoldValley"])
        & (df["Trddt"] >= pd.Timestamp(EVENT_START_DATE))
        & (df["Trddt"] <= pd.Timestamp(EVENT_END_DATE))
        & (df[f"Ret_CC_{MAX_HOLD_DAYS}"].notna())
    ].copy()

    if STRICT_MARKET_TRADING_DAY:
        ok = pd.Series(True, index=events.index)
        for h in range(1, MAX_HOLD_DAYS + 1):
            ok &= events[f"StockDate_F{h}"].eq(events[f"MarketDate_F{h}"])
            ok &= events[f"FutureEligible_F{h}"] == True
        events = events[ok].copy()

    future_ret_cols = [f"Ret_CC_{h}" for h in range(1, MAX_HOLD_DAYS + 1)]
    events["MaxRet_10"] = events[future_ret_cols].max(axis=1)
    events["Hit_10"] = (events["MaxRet_10"] > 0).astype(int)
    events["Up_10"] = (events[f"Ret_CC_{MAX_HOLD_DAYS}"] > 0).astype(int)

    for h in range(1, MAX_HOLD_DAYS + 1):
        for col in [f"Ret_CC_{h}", f"NetRet_CC_{h}", f"ExcessRet_CC_{h}", f"Ret_OC_{h}", f"NetRet_OC_{h}"]:
            if col in events.columns:
                events[f"{col}_W"] = winsorize_series(events[col], WINSOR_LOW_Q, WINSOR_HIGH_Q) if USE_WINSOR else events[col]
    return events


def summarize_probability(events):
    rows = []
    n = len(events)
    for var, name in [("Hit_10", "未来10个交易日内至少一次上涨概率"), ("Up_10", "未来第10个交易日累计收益为正的概率")]:
        success = int(events[var].sum()) if n else 0
        p_hat = success / n if n else np.nan
        w_l, w_u = wilson_ci(success, n, CONF_LEVEL)
        _, b_l, b_u = bootstrap_mean_ci(events[var], B_BOOT, CONF_LEVEL, RANDOM_SEED + len(name))
        rows.append({"指标": name, "成功样本数": success, "总样本数": n, "点估计": p_hat, "Wilson下限": w_l, "Wilson上限": w_u, "Bootstrap下限": b_l, "Bootstrap上限": b_u})
    return pd.DataFrame(rows)


def summarize_horizon_returns(events):
    rows = []
    for h in range(1, MAX_HOLD_DAYS + 1):
        metrics = [
            (f"Ret_CC_{h}_W", "收盘到收盘累计收益率"),
            (f"NetRet_CC_{h}_W", "扣除交易成本后的累计净收益率"),
            (f"ExcessRet_CC_{h}_W", "相对市场平均收益的超额收益率"),
        ]
        if f"Ret_OC_{h}_W" in events.columns:
            metrics += [(f"Ret_OC_{h}_W", "次日开盘买入的可交易收益率"), (f"NetRet_OC_{h}_W", "扣除交易成本后的可交易净收益率")]
        for j, (col, name) in enumerate(metrics):
            mean, low, high = bootstrap_mean_ci(events[col], B_BOOT, CONF_LEVEL, RANDOM_SEED + h * 100 + j)
            rows.append({"持有期h": h, "收益率指标": name, "点估计": mean, "Bootstrap下限": low, "Bootstrap上限": high, "样本数": events[col].dropna().shape[0]})
    return pd.DataFrame(rows)


def evaluate_effectiveness(prob_table, horizon_table):
    hit = prob_table[prob_table["指标"] == "未来10个交易日内至少一次上涨概率"].iloc[0]
    ret10 = horizon_table[(horizon_table["持有期h"] == MAX_HOLD_DAYS) & (horizon_table["收益率指标"] == "收盘到收盘累计收益率")].iloc[0]
    net10 = horizon_table[(horizon_table["持有期h"] == MAX_HOLD_DAYS) & (horizon_table["收益率指标"] == "扣除交易成本后的累计净收益率")].iloc[0]
    excess10 = horizon_table[(horizon_table["持有期h"] == MAX_HOLD_DAYS) & (horizon_table["收益率指标"] == "相对市场平均收益的超额收益率")].iloc[0]

    direction_valid = hit["Wilson下限"] > 0.5
    absolute_valid = ret10["Bootstrap下限"] > 0
    net_valid = net10["Bootstrap下限"] > 0
    relative_valid = excess10["Bootstrap下限"] > 0

    if direction_valid and absolute_valid and net_valid and relative_valid:
        conclusion = "金山谷形态在样本期内具有较强短期有效性"
    elif direction_valid and ret10["点估计"] > 0 and net10["点估计"] > 0:
        conclusion = "金山谷形态具有一定短期参考价值，但稳健性不足"
    else:
        conclusion = "金山谷形态短期有效性不足，不宜单独作为买入依据"

    return pd.DataFrame([
        {"项目": "方向有效性", "判断标准": "未来10日内至少一次上涨概率 Wilson 下限 > 0.5", "结果": direction_valid},
        {"项目": "绝对收益有效性", "判断标准": "第10日累计收益率 Bootstrap 下限 > 0", "结果": absolute_valid},
        {"项目": "交易成本后有效性", "判断标准": "第10日净收益率 Bootstrap 下限 > 0", "结果": net_valid},
        {"项目": "相对收益有效性", "判断标准": "第10日超额收益率 Bootstrap 下限 > 0", "结果": relative_valid},
        {"项目": "综合结论", "判断标准": conclusion, "结果": ""},
    ])


def main():
    df = load_daily_data_with_filters(
        DAILY_FILES,
        LOAD_START_DATE,
        EVENT_END_DATE,
        stock_info_file=STOCK_INFO_FILE,
        strict_filter=STRICT_FILTER,
    )
    df = df.dropna(subset=["Stkcd", "Trddt", "Close", "Ret"]).copy()
    df = add_market_forward_info(df, MAX_HOLD_DAYS)
    df = build_valley_signals(df)
    df = add_future_returns(df, MAX_HOLD_DAYS)
    events = build_gold_valley_events(df)

    prob_table = summarize_probability(events)
    horizon_table = summarize_horizon_returns(events)
    judgement = evaluate_effectiveness(prob_table, horizon_table)

    sample_info = pd.DataFrame([
        {"项目": "日交易样本行数", "数值": len(df)},
        {"项目": "有效日交易样本行数", "数值": int(df["EligibleAtDate"].sum())},
        {"项目": "银山谷事件数", "数值": int(df["SilverValley"].sum())},
        {"项目": "金山谷事件数_全样本", "数值": int(df["GoldValley"].sum())},
        {"项目": "金山谷事件数_过滤后研究区间", "数值": len(events)},
        {"项目": "正式研究起始日期", "数值": EVENT_START_DATE},
        {"项目": "正式研究结束日期", "数值": EVENT_END_DATE},
        {"项目": "未来收益观察窗口", "数值": MAX_HOLD_DAYS},
        {"项目": "Bootstrap次数", "数值": B_BOOT},
        {"项目": "总交易成本率", "数值": ROUND_TRIP_COST},
        {"项目": "是否严格过滤（缺字段即报错）", "数值": STRICT_FILTER},
    ])

    for table in [prob_table, horizon_table]:
        for c in table.columns:
            if pd.api.types.is_numeric_dtype(table[c]):
                table[c] = table[c].round(4)

    event_cols = ["Stkcd", "Trddt", "Markettype", "Close", "MA5", "MA10", "MA20", "ValleyIndex", "SilverValley", "GoldValley", "MaxRet_10", "Hit_10", "Up_10"]
    for h in range(1, MAX_HOLD_DAYS + 1):
        for c in [f"StockDate_F{h}", f"Ret_CC_{h}", f"NetRet_CC_{h}", f"ExcessRet_CC_{h}", f"Ret_OC_{h}", f"NetRet_OC_{h}"]:
            if c in events.columns:
                event_cols.append(c)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        sample_info.to_excel(writer, sheet_name="样本说明", index=False)
        make_filter_report(df, strict_filter=STRICT_FILTER).to_excel(writer, sheet_name="过滤说明", index=False)
        prob_table.to_excel(writer, sheet_name="上涨概率估计", index=False)
        horizon_table.to_excel(writer, sheet_name="未来1至10日收益估计", index=False)
        judgement.to_excel(writer, sheet_name="有效性判断", index=False)
        events[event_cols].to_excel(writer, sheet_name="金山谷事件明细", index=False)

    print("问题三过滤版计算完成：", OUTPUT_XLSX)
    print(sample_info)
    print(prob_table)
    print(horizon_table)
    print(judgement)


if __name__ == "__main__":
    main()

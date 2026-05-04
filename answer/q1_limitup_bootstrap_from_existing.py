"""问题一：直接从已有日交易表提取可用过滤字段后的样本，估计涨停后次日走势概率，并用 Bootstrap 构造置信区间。"""

import numpy as np
import pandas as pd
from data_cleaning_common_from_daily import (
    load_daily_data_with_filters,
    bootstrap_multinomial_ci,
    make_filter_report,
    MAIN_MARKET_TYPES,
    TECH_MARKET_TYPES,
)

DAILY_FILES = [
    r"C:/Users/CPD/Desktop/数模校赛/TRD_Dalyr.xlsx",
    r"C:/Users/CPD/Desktop/数模校赛/TRD_Dalyr1.xlsx",
]

# 需要上传/指定股票基本信息表，用于严格剔除 ST、退市、上市不足一年新股。
STOCK_INFO_FILE = None  # 直接从已有日交易表中提取 Stknme/Listdt/Delistdt/Trdsta 等字段；没有则不伪造过滤
STRICT_FILTER = False

START_DATE = "2025-05-01"
END_DATE = "2026-04-30"
B_BOOT = 5000
CONF_LEVEL = 0.95
RANDOM_SEED = 20260503
OUTPUT_XLSX = "题目1_涨停次日概率_Bootstrap结果_已有表提取版.xlsx"


def classify_main_board(r, limit_status):
    if pd.isna(r) or pd.isna(limit_status):
        return np.nan
    if int(limit_status) == 1:
        return "涨停"
    if int(limit_status) == -1:
        return "跌停"
    if r > 0.07:
        return "涨幅>7%(不含涨停)"
    if 0.05 <= r <= 0.07:
        return "5%~7%"
    if 0.03 <= r < 0.05:
        return "3%~5%"
    if 0 <= r < 0.03:
        return "0%~3%"
    if -0.03 <= r < 0:
        return "-3%~0%"
    if -0.05 <= r < -0.03:
        return "-5%~-3%"
    if -0.07 <= r < -0.05:
        return "-7%~-5%"
    if r < -0.07:
        return "跌幅<-7%(不含跌停)"
    return "未分类"


def classify_tech_board(r, limit_status):
    if pd.isna(r) or pd.isna(limit_status):
        return np.nan
    if int(limit_status) == 1:
        return "涨停"
    if int(limit_status) == -1:
        return "跌停"
    if r > 0.10:
        return "涨幅>10%(不含涨停)"
    if 0.07 <= r <= 0.10:
        return "7%~10%"
    if 0.05 <= r < 0.07:
        return "5%~7%"
    if 0.03 <= r < 0.05:
        return "3%~5%"
    if 0 <= r < 0.03:
        return "0%~3%"
    if -0.03 <= r < 0:
        return "-3%~0%"
    if -0.05 <= r < -0.03:
        return "-5%~-3%"
    if -0.07 <= r < -0.05:
        return "-7%~-5%"
    if -0.10 <= r < -0.07:
        return "-10%~-7%"
    if r < -0.10:
        return "跌幅<-10%(不含跌停)"
    return "未分类"


def main():
    df = load_daily_data_with_filters(
        DAILY_FILES,
        START_DATE,
        END_DATE,
        stock_info_file=STOCK_INFO_FILE,
        strict_filter=STRICT_FILTER,
    )

    df = df.sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)
    g = df.groupby("Stkcd", sort=False)
    df["NextDate"] = g["Trddt"].shift(-1)
    df["NextChangeRatio"] = g["ChangeRatio"].shift(-1)
    df["NextLimitStatus"] = g["LimitStatus"].shift(-1)
    df["NextEligibleAtDate"] = g["EligibleAtDate"].shift(-1)

    limitup_events = df[
        (df["EligibleAtDate"])
        & (df["NextEligibleAtDate"] == True)
        & (df["LimitStatus"] == 1)
        & (df["NextChangeRatio"].notna())
        & (df["NextLimitStatus"].notna())
    ].copy()

    main_events = limitup_events[limitup_events["Markettype"].isin(MAIN_MARKET_TYPES)].copy()
    tech_events = limitup_events[limitup_events["Markettype"].isin(TECH_MARKET_TYPES)].copy()

    main_events["NextCategory"] = [
        classify_main_board(r, s) for r, s in zip(main_events["NextChangeRatio"], main_events["NextLimitStatus"])
    ]
    tech_events["NextCategory"] = [
        classify_tech_board(r, s) for r, s in zip(tech_events["NextChangeRatio"], tech_events["NextLimitStatus"])
    ]

    main_categories = [
        "涨停", "涨幅>7%(不含涨停)", "5%~7%", "3%~5%", "0%~3%",
        "-3%~0%", "-5%~-3%", "-7%~-5%", "跌幅<-7%(不含跌停)", "跌停",
    ]
    tech_categories = [
        "涨停", "涨幅>10%(不含涨停)", "7%~10%", "5%~7%", "3%~5%", "0%~3%",
        "-3%~0%", "-5%~-3%", "-7%~-5%", "-10%~-7%", "跌幅<-10%(不含跌停)", "跌停",
    ]

    result_main = bootstrap_multinomial_ci(main_events["NextCategory"], main_categories, B_BOOT, CONF_LEVEL, RANDOM_SEED)
    result_main.insert(0, "板块", "主板")
    result_tech = bootstrap_multinomial_ci(tech_events["NextCategory"], tech_categories, B_BOOT, CONF_LEVEL, RANDOM_SEED + 1)
    result_tech.insert(0, "板块", "创业板/科创板")
    result = pd.concat([result_main, result_tech], ignore_index=True)

    summary = pd.DataFrame([
        {"项目": "日交易样本行数", "数值": len(df)},
        {"项目": "有效日交易样本行数", "数值": int(df["EligibleAtDate"].sum())},
        {"项目": "涨停事件总数", "数值": len(limitup_events)},
        {"项目": "主板涨停事件数", "数值": len(main_events)},
        {"项目": "创业板/科创板涨停事件数", "数值": len(tech_events)},
        {"项目": "起始日期", "数值": START_DATE},
        {"项目": "结束日期", "数值": END_DATE},
        {"项目": "Bootstrap次数", "数值": B_BOOT},
        {"项目": "是否严格过滤（缺字段即报错）", "数值": STRICT_FILTER},
    ])

    for c in ["频率估计", "Bootstrap下限", "Bootstrap上限"]:
        result[c] = result[c].round(4)

    detail_cols = ["Stkcd", "Trddt", "NextDate", "Markettype", "ChangeRatio", "NextChangeRatio", "NextLimitStatus", "NextCategory"]
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="样本说明", index=False)
        make_filter_report(df, strict_filter=STRICT_FILTER).to_excel(writer, sheet_name="过滤说明", index=False)
        result.to_excel(writer, sheet_name="题目1结果", index=False)
        main_events[detail_cols].to_excel(writer, sheet_name="主板事件样本", index=False)
        tech_events[detail_cols].to_excel(writer, sheet_name="创业科创事件样本", index=False)

    print("问题一过滤版计算完成：", OUTPUT_XLSX)
    print(summary)
    print(result)


if __name__ == "__main__":
    main()

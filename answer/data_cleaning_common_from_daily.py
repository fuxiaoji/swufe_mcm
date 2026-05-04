"""
通用数据清洗模块：供问题1、问题2、问题3共用。

本版本优先“直接从已有日交易表中提取”过滤字段：
- 若日交易表包含 Stknme，则剔除 ST/退市名称股票；
- 若日交易表包含 Listdt，则剔除上市不足一年新股；
- 若日交易表包含 Delistdt 或 Trdsta，则进一步剔除退市/异常状态股票；
- 若这些字段在已有表中不存在，则不会伪造过滤结果，而是在输出的“过滤说明”中标记为未应用。

核心原则：
1. 不先删除无效记录再 shift(-1)，避免“下一交易日”错位；
2. 先在完整交易序列上计算技术指标和未来收益；
3. 再用 EligibleAtDate / FutureEligible 标记筛选事件样本。
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook

VALID_MARKET_TYPES = {1, 2, 4, 8, 16, 32}
MAIN_MARKET_TYPES = {1, 2, 4, 8}
TECH_MARKET_TYPES = {16, 32}


def normalize_stock_code(x) -> str | None:
    """证券代码统一为 6 位字符串。"""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(6)


def pick_col(header: list[str], candidates: list[str], required: bool = False) -> str | None:
    """从表头中自动匹配字段名。"""
    header_set = set(header)
    for c in candidates:
        if c in header_set:
            return c
    if required:
        raise ValueError(f"缺少必要字段，候选字段为：{candidates}")
    return None


def normalize_return_unit(s: pd.Series) -> pd.Series:
    """
    统一收益率单位。
    若原始数据像 3.5、-2.1，视为百分数并转为 0.035、-0.021；
    若已经是 0.035，则保持不变。
    """
    s = pd.to_numeric(s, errors="coerce")
    q = s.abs().quantile(0.95)
    if pd.notna(q) and q > 1:
        s = s / 100
    return s


def read_csmar_daily_xlsx(path: str | Path) -> pd.DataFrame:
    """
    读取 CSMAR TRD_Dalyr 日交易数据。
    兼容 CSMAR 常见三行表头：
    第 1 行：英文字段名；第 2 行：中文说明；第 3 行：单位。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到日交易数据文件：{path}")

    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    ws.reset_dimensions()
    rows = ws.iter_rows(values_only=True)

    header = [str(v).strip() if v is not None else "" for v in next(rows)]

    col_map: dict[str, str | None] = {
        "Stkcd": pick_col(header, ["Stkcd", "Symbol", "证券代码"], required=True),
        "Trddt": pick_col(header, ["Trddt", "TradingDate", "交易日期"], required=True),
        "Markettype": pick_col(header, ["Markettype", "MarketType", "市场类型"]),
        "LimitStatus": pick_col(header, ["LimitStatus", "涨跌停状态"]),
        "Open": pick_col(header, ["Opnprc", "OpenPrice", "Open", "开盘价"]),
        "High": pick_col(header, ["Hiprc", "HighPrice", "High", "最高价"]),
        "Low": pick_col(header, ["Loprc", "LowPrice", "Low", "最低价"]),
        "Close": pick_col(header, ["Clsprc", "ClosePrice", "Close", "收盘价"]),
        "PreClose": pick_col(header, ["PreClosePrice", "PreClose", "昨收盘(交易所)", "昨收盘价"]),
        "ChangeRatio": pick_col(header, ["ChangeRatio", "涨跌幅"]),
        "Dretnd": pick_col(header, ["Dretnd", "不考虑现金红利的日个股回报率", "不考虑现金红利再投资的日个股回报率"]),
        "Dretwd": pick_col(header, ["Dretwd", "考虑现金红利再投资的日个股回报率"]),
        "Stknme": pick_col(header, ["Stknme", "ShortName", "SecurityAbbr", "股票简称"]),
        "Listdt": pick_col(header, ["Listdt", "ListDate", "上市日期"]),
        "Delistdt": pick_col(header, ["Delistdt", "DelistDate", "退市日期"]),
        "Trdsta": pick_col(header, ["Trdsta", "TradingStatus", "交易状态"]),
    }

    use_cols = [v for v in col_map.values() if v is not None]
    use_cols = list(dict.fromkeys(use_cols))
    col_idx = {c: header.index(c) for c in use_cols}

    # 跳过中文说明行、单位行
    next(rows, None)
    next(rows, None)

    records = []
    for row in rows:
        if row is None:
            continue
        rec = {}
        for std_name, raw_name in col_map.items():
            if raw_name is None:
                continue
            j = col_idx[raw_name]
            rec[std_name] = row[j] if j < len(row) else None
        if rec.get("Stkcd") is None or rec.get("Trddt") is None:
            continue
        records.append(rec)

    wb.close()
    df = pd.DataFrame(records)

    df["Stkcd"] = df["Stkcd"].map(normalize_stock_code)
    df = df[df["Stkcd"].astype(str).str.fullmatch(r"\d{6}", na=False)].copy()
    df["Trddt"] = pd.to_datetime(df["Trddt"], errors="coerce")

    for col in ["Markettype", "LimitStatus", "Open", "High", "Low", "Close", "PreClose"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["ChangeRatio", "Dretnd", "Dretwd"]:
        if col in df.columns:
            df[col] = normalize_return_unit(df[col])

    for col in ["Listdt", "Delistdt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 统一收益率字段。短期交易评价优先使用不考虑现金红利的 Dretnd。
    if "Dretnd" in df.columns:
        df["Ret"] = df["Dretnd"]
    elif "ChangeRatio" in df.columns:
        df["Ret"] = df["ChangeRatio"]
    elif "Dretwd" in df.columns:
        df["Ret"] = df["Dretwd"]
    else:
        df["Ret"] = np.nan

    return df


def _read_first_sheet_generic(path: str | Path) -> pd.DataFrame:
    """读取股票基本信息表，兼容普通一行表头或 CSMAR 三行表头。"""
    path = Path(path)
    raw = pd.read_excel(path, header=0)
    # 如果第一行看起来像中文说明/单位，而不是数据，则删除前两行。
    # 这里保守处理：只在 Stkcd 列的前两行不是 6 位代码时才跳过。
    if "Stkcd" in raw.columns and len(raw) >= 2:
        first_val = str(raw.iloc[0]["Stkcd"])
        if not first_val.zfill(6).isdigit():
            raw = pd.read_excel(path, header=0, skiprows=[1, 2])
    return raw


def read_stock_info(path: str | Path | None, strict_filter: bool = True) -> pd.DataFrame | None:
    """
    读取股票基本信息表。
    至少建议包含：Stkcd、Stknme、Listdt。
    可选字段：Delistdt、Trdsta。
    """
    if path is None or str(path).strip() == "":
        if strict_filter:
            raise FileNotFoundError(
                "未提供股票基本信息表路径。若要严格剔除 ST、退市股、上市不足一年新股，"
                "请提供包含 Stkcd、Stknme、Listdt 的股票基本信息表。"
            )
        return None

    path = Path(path)
    if not path.exists():
        if strict_filter:
            raise FileNotFoundError(
                f"未找到股票基本信息表：{path}。请上传或修改 STOCK_INFO_FILE 路径。"
            )
        return None

    info = _read_first_sheet_generic(path)
    header = [str(c).strip() for c in info.columns]
    info.columns = header

    stock_col = pick_col(header, ["Stkcd", "Symbol", "证券代码"], required=True)
    name_col = pick_col(header, ["Stknme", "ShortName", "SecurityAbbr", "股票简称"])
    list_col = pick_col(header, ["Listdt", "ListDate", "上市日期"])
    delist_col = pick_col(header, ["Delistdt", "DelistDate", "退市日期"])
    trdsta_col = pick_col(header, ["Trdsta", "TradingStatus", "交易状态"])

    rename_map = {stock_col: "Stkcd"}
    if name_col:
        rename_map[name_col] = "Stknme"
    if list_col:
        rename_map[list_col] = "Listdt"
    if delist_col:
        rename_map[delist_col] = "Delistdt"
    if trdsta_col:
        rename_map[trdsta_col] = "Trdsta"

    info = info.rename(columns=rename_map)
    keep_cols = [c for c in ["Stkcd", "Stknme", "Listdt", "Delistdt", "Trdsta"] if c in info.columns]
    info = info[keep_cols].copy()

    info["Stkcd"] = info["Stkcd"].map(normalize_stock_code)
    info = info[info["Stkcd"].astype(str).str.fullmatch(r"\d{6}", na=False)].copy()

    for col in ["Listdt", "Delistdt"]:
        if col in info.columns:
            info[col] = pd.to_datetime(info[col], errors="coerce")

    info = info.drop_duplicates(subset=["Stkcd"], keep="last")
    return info


def add_eligibility_flags(
    df: pd.DataFrame,
    stock_info: pd.DataFrame | None = None,
    strict_filter: bool = True,
) -> pd.DataFrame:
    """
    为每一条股票-日期记录生成 EligibleAtDate。
    EligibleAtDate=True 表示该记录满足：研究板块、非 ST、非退市、上市满一年。
    """
    df = df.copy()

    if stock_info is not None:
        df = df.merge(stock_info, on="Stkcd", how="left", suffixes=("", "_info"))
        for col in ["Stknme", "Listdt", "Delistdt", "Trdsta"]:
            info_col = f"{col}_info"
            if info_col in df.columns:
                if col in df.columns:
                    df[col] = df[col].combine_first(df[info_col])
                else:
                    df[col] = df[info_col]
                df = df.drop(columns=[info_col])

    missing = [c for c in ["Markettype", "Stknme", "Listdt"] if c not in df.columns]
    if strict_filter and missing:
        raise ValueError(
            f"无法严格过滤样本，缺少字段：{missing}。"
            "请在日交易数据或股票基本信息表中补充 Markettype、Stknme、Listdt。"
        )

    if "Markettype" in df.columns:
        df["Markettype"] = pd.to_numeric(df["Markettype"], errors="coerce")
        df["MarketOK"] = df["Markettype"].isin(VALID_MARKET_TYPES)
    else:
        df["MarketOK"] = True

    if "Stknme" in df.columns:
        name = df["Stknme"].astype(str)
        df["IsST"] = name.str.contains("ST", case=False, regex=True, na=False)
        df["IsDelistByName"] = name.str.contains("退", case=False, regex=True, na=False)
    else:
        df["IsST"] = False
        df["IsDelistByName"] = False

    if "Delistdt" in df.columns:
        df["Delistdt"] = pd.to_datetime(df["Delistdt"], errors="coerce")
        df["IsDelistedByDate"] = df["Delistdt"].notna() & (df["Delistdt"] <= df["Trddt"])
    else:
        df["IsDelistedByDate"] = False

    if "Trdsta" in df.columns:
        trdsta = df["Trdsta"].astype(str)
        df["IsDelistedByStatus"] = trdsta.str.contains("退|终止|暂停上市|终止上市", case=False, regex=True, na=False)
    else:
        df["IsDelistedByStatus"] = False

    if "Listdt" in df.columns:
        df["Listdt"] = pd.to_datetime(df["Listdt"], errors="coerce")
        df["AgeDays"] = (df["Trddt"] - df["Listdt"]).dt.days
        df["IsNewStock"] = df["AgeDays"].isna() | (df["AgeDays"] < 365)
    else:
        df["AgeDays"] = np.nan
        df["IsNewStock"] = False

    df["EligibleAtDate"] = (
        df["MarketOK"]
        & (~df["IsST"])
        & (~df["IsDelistByName"])
        & (~df["IsDelistedByDate"])
        & (~df["IsDelistedByStatus"])
        & (~df["IsNewStock"])
    )

    return df


def load_daily_data_with_filters(
    daily_files: list[str | Path],
    start_date: str,
    end_date: str,
    stock_info_file: str | Path | None = None,
    strict_filter: bool = True,
) -> pd.DataFrame:
    """
    读取日交易数据并生成 EligibleAtDate 标记。
    注意：返回的是完整交易序列 + 有效性标记，不会提前删除无效记录。
    """
    df = pd.concat([read_csmar_daily_xlsx(f) for f in daily_files], ignore_index=True)
    df = df.dropna(subset=["Stkcd", "Trddt"])
    df = df[(df["Trddt"] >= pd.Timestamp(start_date)) & (df["Trddt"] <= pd.Timestamp(end_date))].copy()
    df = df.drop_duplicates(subset=["Stkcd", "Trddt"], keep="last")
    df = df.sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)

    if df["Ret"].isna().all():
        if "Close" not in df.columns:
            raise ValueError("数据缺少收益率字段，也缺少收盘价，无法构造 Ret。")
        df["Ret"] = df.groupby("Stkcd")["Close"].pct_change()

    if "ChangeRatio" not in df.columns:
        df["ChangeRatio"] = df["Ret"]

    stock_info = read_stock_info(stock_info_file, strict_filter=strict_filter)
    df = add_eligibility_flags(df, stock_info=stock_info, strict_filter=strict_filter)
    return df


def add_next_market_dates(df: pd.DataFrame, max_h: int = 1) -> pd.DataFrame:
    """添加全市场未来第 h 个交易日日期。"""
    df = df.copy()
    dates = pd.Series(sorted(pd.to_datetime(df["Trddt"].dropna().unique()))).reset_index(drop=True)
    for h in range(1, max_h + 1):
        mapping = dict(zip(dates.iloc[:-h], dates.iloc[h:]))
        df[f"MarketDate_F{h}"] = df["Trddt"].map(mapping)
    return df


def bootstrap_mean_ci(values, b_boot: int = 5000, confidence: float = 0.95, seed: int = 20260503, chunk: int = 200):
    """均值 Bootstrap 区间。二元变量传入后，均值即概率。"""
    arr = pd.Series(values).dropna().to_numpy(dtype=float)
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = np.empty(b_boot)
    pos = 0
    while pos < b_boot:
        m = min(chunk, b_boot - pos)
        idx = rng.integers(0, n, size=(m, n))
        stats[pos:pos + m] = arr[idx].mean(axis=1)
        pos += m
    alpha = 1 - confidence
    return arr.mean(), np.quantile(stats, alpha / 2), np.quantile(stats, 1 - alpha / 2)


def bootstrap_multinomial_ci(labels: pd.Series, categories: list[str], b_boot: int = 5000, confidence: float = 0.95, seed: int = 20260503):
    """多类别频率的 Bootstrap 区间。"""
    labels = pd.Series(labels).dropna().astype(str)
    n = len(labels)
    if n == 0:
        return pd.DataFrame({
            "类别": categories,
            "样本数": 0,
            "频率估计": np.nan,
            "Bootstrap下限": np.nan,
            "Bootstrap上限": np.nan,
            "总事件数N": 0,
        })
    counts = labels.value_counts().reindex(categories, fill_value=0)
    p_hat = counts / n
    rng = np.random.default_rng(seed)
    boot_counts = rng.multinomial(n=n, pvals=p_hat.values, size=b_boot)
    boot_props = boot_counts / n
    alpha = 1 - confidence
    lower = np.quantile(boot_props, alpha / 2, axis=0)
    upper = np.quantile(boot_props, 1 - alpha / 2, axis=0)
    return pd.DataFrame({
        "类别": categories,
        "样本数": counts.values,
        "频率估计": p_hat.values,
        "Bootstrap下限": lower,
        "Bootstrap上限": upper,
        "总事件数N": n,
    })


def wilson_ci(success: int, n: int, confidence: float = 0.95):
    """Wilson 二项比例置信区间。"""
    if n == 0:
        return np.nan, np.nan
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = success / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half



def make_filter_report(df: pd.DataFrame, strict_filter: bool = False) -> pd.DataFrame:
    """生成数据过滤说明，写入结果 Excel，避免误以为不存在字段时仍完成了严格过滤。"""
    rows = []

    def status(applied: bool, applied_text: str, missing_text: str):
        return applied_text if applied else missing_text

    rows.append({
        "过滤项目": "板块范围",
        "是否应用": "是" if "Markettype" in df.columns else "否",
        "依据字段": "Markettype" if "Markettype" in df.columns else "无",
        "说明": "保留主板、创业板、科创板市场类型" if "Markettype" in df.columns else "已有日交易表未提供 Markettype，无法按板块过滤",
    })
    rows.append({
        "过滤项目": "ST 股票",
        "是否应用": "是" if "Stknme" in df.columns else "否",
        "依据字段": "Stknme" if "Stknme" in df.columns else "无",
        "说明": "股票简称含 ST 的样本被剔除" if "Stknme" in df.columns else "已有日交易表未提供 Stknme，无法直接剔除 ST；如需严格剔除，请补充股票基本信息表或包含 Stknme/Listdt 的日交易表",
    })
    delist_fields = [c for c in ["Stknme", "Delistdt", "Trdsta"] if c in df.columns]
    rows.append({
        "过滤项目": "退市股票",
        "是否应用": "是" if delist_fields else "否",
        "依据字段": ", ".join(delist_fields) if delist_fields else "无",
        "说明": "根据名称、退市日期或交易状态剔除退市/异常状态样本" if delist_fields else "已有日交易表未提供退市识别字段，无法直接剔除退市股",
    })
    rows.append({
        "过滤项目": "上市不足一年新股",
        "是否应用": "是" if "Listdt" in df.columns else "否",
        "依据字段": "Listdt" if "Listdt" in df.columns else "无",
        "说明": "事件日距上市日不足 365 天的样本被剔除" if "Listdt" in df.columns else "已有日交易表未提供 Listdt，无法直接剔除上市不足一年新股",
    })
    rows.append({
        "过滤项目": "运行模式",
        "是否应用": "严格" if strict_filter else "自动提取",
        "依据字段": "已有日交易表字段",
        "说明": "strict_filter=True 时缺少关键字段会报错；本自动版 strict_filter=False，缺失字段仅在本表说明，不伪造过滤结果。",
    })
    return pd.DataFrame(rows)

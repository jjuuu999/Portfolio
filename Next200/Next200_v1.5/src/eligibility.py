from __future__ import annotations

import pandas as pd


MANUAL_EXCLUDED_TICKERS = {
    "088980",  # 맥쿼리인프라: 사회기반시설투융자회사 성격의 명시 제외
}

MANUAL_EXCLUDED_NAME_KEYWORDS = (
    "스팩",
)


def _ticker_series(frame: pd.DataFrame) -> pd.Series:
    if "ticker" not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame["ticker"].astype(str).str.zfill(6)


def build_eligibility_mask(
    frame: pd.DataFrame,
    period_end_date: pd.Timestamp | None = None,
) -> pd.Series:
    if period_end_date is None:
        period_end_date = pd.Timestamp.now()

    float_rate = pd.to_numeric(frame.get("float_rate", pd.Series(index=frame.index)), errors="coerce")
    excluded_not_common = frame.get("is_not_common", pd.Series(0, index=frame.index)).fillna(0) == 1
    excluded_low_float = (float_rate < 0.10) & float_rate.notna()
    excluded_reits = frame.get("is_reits", pd.Series(0, index=frame.index)).fillna(0) == 1
    excluded_managed = frame.get("is_managed", pd.Series(0, index=frame.index)).fillna(0) == 1
    excluded_warning = frame.get("is_warning", pd.Series(0, index=frame.index)).fillna(0) == 1

    excluded_recent_listing = pd.Series(False, index=frame.index)
    if "list_date" in frame.columns:
        listing_date = pd.to_datetime(frame["list_date"], errors="coerce")
        months = (period_end_date.year - listing_date.dt.year) * 12 + (
            period_end_date.month - listing_date.dt.month
        )
        excluded_recent_listing = (months < 6) & months.notna()

    excluded_manual_ticker = _ticker_series(frame).isin(MANUAL_EXCLUDED_TICKERS)
    excluded_manual_name = pd.Series(False, index=frame.index)
    if "company" in frame.columns:
        company_series = frame["company"].fillna("").astype(str)
        for keyword in MANUAL_EXCLUDED_NAME_KEYWORDS:
            excluded_manual_name = excluded_manual_name | company_series.str.contains(keyword, regex=False)

    return (
        excluded_not_common
        | excluded_low_float
        | excluded_reits
        | excluded_managed
        | excluded_warning
        | excluded_recent_listing
        | excluded_manual_ticker
        | excluded_manual_name
    )


def apply_eligibility_filter(
    frame: pd.DataFrame,
    period_end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    return frame.loc[~build_eligibility_mask(frame, period_end_date=period_end_date)].copy()

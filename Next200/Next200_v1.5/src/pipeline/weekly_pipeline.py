from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd

from src.collectors import DartCollector, NaverCollector, YahooCollector
from src.config import load_config
from src.pipeline.feature_builder import build_weekly_features
from src.sql_dump import load_table


def get_default_tickers(project_root: Path) -> list[str]:
    sql_path = project_root / "data" / "raw" / "kospi_db_full_20260320.sql"
    feature = load_table(sql_path, "feature_krx")
    if feature.empty:
        return []
    feature["ticker"] = feature["ticker"].astype(str).str.zfill(6)
    feature["period"] = feature["period"].astype(str)
    latest_period = sorted(
        feature["period"].dropna().unique().tolist(),
        key=lambda value: (int(str(value).split("_")[0]), 1 if str(value).endswith("H1") else 2),
    )[-1]
    tickers = (
        feature.loc[feature["period"] == latest_period, "ticker"]
        .drop_duplicates()
        .tolist()
    )
    return tickers


def _apply_sql_fallback_meta(
    project_root: Path,
    tickers: list[str],
    naver_df: pd.DataFrame,
    previous_meta_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    sql_path = project_root / "data" / "raw" / "kospi_db_full_20260320.sql"
    latest_daily = load_table(sql_path, "kospi_friday_daily")
    feature = load_table(sql_path, "feature_krx")

    if latest_daily.empty or feature.empty:
        return naver_df

    latest_daily["ticker"] = latest_daily["ticker"].astype(str).str.zfill(6)
    latest_daily["date"] = pd.to_numeric(latest_daily["date"], errors="coerce")
    latest_daily = (
        latest_daily.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
    )

    feature["ticker"] = feature["ticker"].astype(str).str.zfill(6)
    feature["period"] = feature["period"].astype(str)
    latest_period = sorted(
        feature["period"].dropna().unique().tolist(),
        key=lambda value: (int(str(value).split("_")[0]), 1 if str(value).endswith("H1") else 2),
    )[-1]
    feature_latest = feature.loc[feature["period"] == latest_period].copy()
    feature_latest = feature_latest[["ticker", "float_ratio", "gics_sector", "krx_group"]]

    expected_columns = [
        "ticker",
        "company",
        "market",
        "sector",
        "industry",
        "shares_outstanding",
        "float_rate",
        "foreign_ratio",
        "major_holder_ratio",
        "treasury_ratio",
        "current_price",
        "current_volume",
        "source_main_url",
        "source_coinfo_url",
        "source_wisereport_url",
        "float_rate_source",
        "major_holder_source",
        "treasury_source",
    ]

    merged = pd.DataFrame({"ticker": tickers})
    if naver_df.empty:
        merged = merged.merge(pd.DataFrame(columns=expected_columns), on="ticker", how="left")
    else:
        for column in expected_columns:
            if column not in naver_df.columns:
                naver_df[column] = pd.NA
        merged = merged.merge(naver_df, on="ticker", how="left")

    previous_meta_df = previous_meta_df.copy() if previous_meta_df is not None and not previous_meta_df.empty else pd.DataFrame()
    if not previous_meta_df.empty:
        for column in expected_columns:
            if column not in previous_meta_df.columns:
                previous_meta_df[column] = pd.NA
        previous_meta_df["ticker"] = previous_meta_df["ticker"].astype(str).str.zfill(6)
        previous_meta_df = previous_meta_df.rename(
            columns={
                "float_rate": "prev_float_rate",
                "foreign_ratio": "prev_foreign_ratio",
                "major_holder_ratio": "prev_major_holder_ratio",
                "treasury_ratio": "prev_treasury_ratio",
            }
        )
        merged = merged.merge(
            previous_meta_df[
                [
                    "ticker",
                    "prev_float_rate",
                    "prev_foreign_ratio",
                    "prev_major_holder_ratio",
                    "prev_treasury_ratio",
                ]
            ],
            on="ticker",
            how="left",
        )
    else:
        merged["prev_float_rate"] = pd.NA
        merged["prev_foreign_ratio"] = pd.NA
        merged["prev_major_holder_ratio"] = pd.NA
        merged["prev_treasury_ratio"] = pd.NA
    merged = merged.merge(
        latest_daily[["ticker", "company", "close", "volume", "shares"]],
        on="ticker",
        how="left",
        suffixes=("", "_daily"),
    )
    merged = merged.merge(feature_latest, on="ticker", how="left")

    missing_mask = (
        (
            merged["company"].isna()
            | merged["shares_outstanding"].isna()
            | merged["float_rate"].isna()
            | merged["current_price"].isna()
        )
        & merged["close"].notna()
    )
    if missing_mask.any():
        merged.loc[missing_mask, "company"] = merged.loc[missing_mask, "company_daily"]
        merged.loc[missing_mask, "market"] = "코스피"
        merged.loc[missing_mask, "sector"] = merged.loc[missing_mask, "gics_sector"]
        merged.loc[missing_mask, "industry"] = merged.loc[missing_mask, "krx_group"]
        merged.loc[missing_mask, "shares_outstanding"] = merged.loc[missing_mask, "shares"]
        merged.loc[missing_mask, "float_rate"] = pd.to_numeric(
            merged.loc[missing_mask, "float_ratio"], errors="coerce"
        ) * 100.0
        merged.loc[missing_mask, "current_price"] = merged.loc[missing_mask, "close"]
        merged.loc[missing_mask, "current_volume"] = merged.loc[missing_mask, "volume"]

    merged["float_rate"] = pd.to_numeric(merged["float_rate"], errors="coerce")
    merged["prev_float_rate"] = pd.to_numeric(merged["prev_float_rate"], errors="coerce")
    merged["foreign_ratio"] = pd.to_numeric(merged["foreign_ratio"], errors="coerce")
    merged["prev_foreign_ratio"] = pd.to_numeric(merged["prev_foreign_ratio"], errors="coerce")
    merged["major_holder_ratio"] = pd.to_numeric(merged["major_holder_ratio"], errors="coerce")
    merged["prev_major_holder_ratio"] = pd.to_numeric(merged["prev_major_holder_ratio"], errors="coerce")
    merged["treasury_ratio"] = pd.to_numeric(merged["treasury_ratio"], errors="coerce")
    merged["prev_treasury_ratio"] = pd.to_numeric(merged["prev_treasury_ratio"], errors="coerce")

    float_from_naver = merged["float_rate"].notna() & (merged["float_rate"] > 0)
    float_from_prev = (~float_from_naver) & merged["prev_float_rate"].notna() & (merged["prev_float_rate"] > 0)
    merged.loc[float_from_prev, "float_rate"] = merged.loc[float_from_prev, "prev_float_rate"]
    merged.loc[float_from_naver, "float_rate_source"] = "naver_actual"
    merged.loc[float_from_prev, "float_rate_source"] = "previous_week"
    merged["float_rate_source"] = merged["float_rate_source"].fillna("missing")

    major_from_naver = merged["major_holder_ratio"].notna()
    major_from_prev = (~major_from_naver) & merged["prev_major_holder_ratio"].notna()
    merged.loc[major_from_prev, "major_holder_ratio"] = merged.loc[major_from_prev, "prev_major_holder_ratio"]
    merged.loc[major_from_naver, "major_holder_source"] = "naver_actual"
    merged.loc[major_from_prev, "major_holder_source"] = "previous_week"
    merged["major_holder_source"] = merged["major_holder_source"].fillna("missing")

    treasury_from_naver = merged["treasury_ratio"].notna()
    treasury_from_prev = (~treasury_from_naver) & merged["prev_treasury_ratio"].notna()
    merged.loc[treasury_from_prev, "treasury_ratio"] = merged.loc[treasury_from_prev, "prev_treasury_ratio"]
    merged.loc[treasury_from_naver, "treasury_source"] = "naver_actual"
    merged.loc[treasury_from_prev, "treasury_source"] = "previous_week"
    merged["treasury_source"] = merged["treasury_source"].fillna("missing")

    foreign_from_prev = merged["foreign_ratio"].isna() & merged["prev_foreign_ratio"].notna()
    merged.loc[foreign_from_prev, "foreign_ratio"] = merged.loc[foreign_from_prev, "prev_foreign_ratio"]

    for column in expected_columns:
        if column not in merged.columns:
            merged[column] = pd.NA

    return merged[expected_columns]


def _append_auto_foreign_history(output_dir: Path, price_df: pd.DataFrame, meta_df: pd.DataFrame) -> Path:
    foreign_path = output_dir / "naver_foreign_holding_weekly.csv"
    if price_df.empty or meta_df.empty or "foreign_ratio" not in meta_df.columns:
        return foreign_path

    snapshot_date = pd.to_datetime(price_df["date"], errors="coerce").dropna()
    if snapshot_date.empty:
        return foreign_path
    as_of_date = snapshot_date.max().strftime("%Y-%m-%d")

    foreign_frame = meta_df[["ticker", "foreign_ratio"]].copy()
    foreign_frame["ticker"] = foreign_frame["ticker"].astype(str).str.zfill(6)
    foreign_frame["foreign_holding_ratio"] = pd.to_numeric(foreign_frame["foreign_ratio"], errors="coerce")
    foreign_frame["foreign_limit_exhaustion_rate"] = pd.NA
    foreign_frame["date"] = as_of_date
    foreign_frame = foreign_frame.drop(columns=["foreign_ratio"])
    foreign_frame = foreign_frame.dropna(subset=["foreign_holding_ratio"])

    if foreign_frame.empty:
        return foreign_path

    if foreign_path.exists():
        existing = pd.read_csv(foreign_path, dtype={"ticker": str})
        combined = pd.concat([existing, foreign_frame], ignore_index=True)
    else:
        combined = foreign_frame

    combined["ticker"] = combined["ticker"].astype(str).str.zfill(6)
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    combined.to_csv(foreign_path, index=False, encoding="utf-8-sig")
    return foreign_path


def run_weekly_collection(limit: int | None = None) -> dict[str, object]:
    config = load_config()
    tickers = get_default_tickers(config.project_root)
    if limit is not None:
        tickers = tickers[:limit]

    output_dir = config.project_root / "data" / "incoming" / "auto"
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_meta_path = output_dir / "naver_stock_meta_weekly.csv"
    previous_meta_df = pd.DataFrame()
    if previous_meta_path.exists():
        try:
            previous_meta_df = pd.read_csv(previous_meta_path, dtype={"ticker": str})
        except Exception:
            previous_meta_df = pd.DataFrame()

    naver = NaverCollector().collect(tickers)
    naver_df = pd.DataFrame(naver.rows)
    naver_df = _apply_sql_fallback_meta(config.project_root, tickers, naver_df, previous_meta_df=previous_meta_df)
    market_by_ticker = {}
    naver_price_fallback = {}
    if not naver_df.empty:
        market_by_ticker = dict(zip(naver_df["ticker"], naver_df["market"]))
        naver_price_fallback = {
            row["ticker"]: {"current_price": row.get("current_price"), "current_volume": row.get("current_volume")}
            for row in naver_df.to_dict("records")
        }

    yahoo = YahooCollector().collect(
        tickers,
        market_by_ticker=market_by_ticker,
        naver_price_fallback=naver_price_fallback,
    )
    dart = DartCollector(config.open_dart_api_key).collect(tickers)

    pd.DataFrame(yahoo.rows).to_csv(output_dir / "yahoo_price_daily.csv", index=False, encoding="utf-8-sig")
    naver_df.to_csv(output_dir / "naver_stock_meta_weekly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dart.rows).to_csv(output_dir / "dart_major_holder_weekly.csv", index=False, encoding="utf-8-sig")
    foreign_history_path = _append_auto_foreign_history(output_dir, pd.DataFrame(yahoo.rows), naver_df)

    summary = {
        "tickers": len(tickers),
        "yahoo": asdict(yahoo),
        "naver": asdict(naver),
        "dart": asdict(dart),
        "foreign_history_path": str(foreign_history_path),
        "float_rate_source_counts": naver_df.get("float_rate_source", pd.Series(dtype="object")).value_counts(dropna=False).to_dict(),
        "major_holder_source_counts": naver_df.get("major_holder_source", pd.Series(dtype="object")).value_counts(dropna=False).to_dict(),
        "treasury_source_counts": naver_df.get("treasury_source", pd.Series(dtype="object")).value_counts(dropna=False).to_dict(),
    }
    try:
        summary["feature_build"] = build_weekly_features()
    except Exception as exc:
        summary["feature_build"] = {"error": str(exc)}
    (output_dir / "weekly_collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary

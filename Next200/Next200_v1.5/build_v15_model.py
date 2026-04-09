from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from src.config import load_config
from src.eligibility import apply_eligibility_filter
from src.sql_dump import load_table


V15_FEATURES = [
    "prev_was_member",
    "period_rank",
    "dist_from_200",
    "float_mktcap_rank",
    "sector_relative_rank",
    "float_dist_from_200",
    "sector_rank",
    "avg_mktcap",
    "rank_change",
    "last_foreign_ratio",
    "float_rate",
    "non_float_ratio",
    "major_holder_ratio",
    "avg_foreign_ratio",
]


def period_sort_key(period_value: str) -> tuple[int, int]:
    year_text, half_text = str(period_value).split("_")
    return int(year_text), 1 if half_text == "H1" else 2


def build_foreign_aggregate(
    foreign_holding: pd.DataFrame,
    period_frame: pd.DataFrame,
    target_period: str,
    prev_period: str | None,
) -> pd.DataFrame:
    if foreign_holding.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "avg_foreign_ratio",
                "last_foreign_ratio",
                "avg_exhaustion_rate",
                "foreign_change",
            ]
        )

    foreign = foreign_holding.copy()
    foreign["ticker"] = foreign["ticker"].astype(str).str.zfill(6)
    foreign["obs_date"] = pd.to_datetime(
        pd.to_numeric(foreign["ym"], errors="coerce").astype("Int64").astype(str) + "01",
        format="%Y%m%d",
        errors="coerce",
    )
    foreign["foreign_holding_ratio"] = pd.to_numeric(foreign["foreign_holding_ratio"], errors="coerce")
    foreign["foreign_limit_exhaustion_rate"] = pd.to_numeric(
        foreign["foreign_limit_exhaustion_rate"], errors="coerce"
    )
    foreign = foreign.dropna(subset=["obs_date"]).sort_values(["ticker", "obs_date"])

    period_lookup = period_frame.set_index("period")
    current_period = period_lookup.loc[target_period]
    current_start = pd.to_datetime(current_period["period_start"], errors="coerce")
    current_end = pd.to_datetime(current_period["period_end"], errors="coerce")
    current = foreign[(foreign["obs_date"] >= current_start) & (foreign["obs_date"] <= current_end)].copy()

    current_agg = pd.DataFrame(
        columns=["ticker", "avg_foreign_ratio", "last_foreign_ratio", "avg_exhaustion_rate"]
    )
    if not current.empty:
        current_sorted = current.sort_values(["ticker", "obs_date"])
        current_agg = (
            current_sorted.groupby("ticker")
            .agg(
                avg_foreign_ratio=("foreign_holding_ratio", "mean"),
                last_foreign_ratio=("foreign_holding_ratio", "last"),
                avg_exhaustion_rate=("foreign_limit_exhaustion_rate", "mean"),
            )
            .reset_index()
        )

    prev_agg = pd.DataFrame(columns=["ticker", "prev_avg_foreign_ratio"])
    if prev_period and prev_period in period_lookup.index:
        prev_period_row = period_lookup.loc[prev_period]
        prev_start = pd.to_datetime(prev_period_row["period_start"], errors="coerce")
        prev_end = pd.to_datetime(prev_period_row["period_end"], errors="coerce")
        prev = foreign[(foreign["obs_date"] >= prev_start) & (foreign["obs_date"] <= prev_end)].copy()
        if not prev.empty:
            prev_agg = (
                prev.groupby("ticker")
                .agg(prev_avg_foreign_ratio=("foreign_holding_ratio", "mean"))
                .reset_index()
            )

    merged = current_agg.merge(prev_agg, on="ticker", how="left")
    merged["foreign_change"] = merged["avg_foreign_ratio"] - merged["prev_avg_foreign_ratio"].fillna(0.0)
    return merged.drop(columns=["prev_avg_foreign_ratio"], errors="ignore")


def build_historical_snapshot(config) -> pd.DataFrame:
    feature_krx = load_table(config.sql_dump_path, "feature_krx")
    labels = load_table(config.sql_dump_path, "labels")
    major_holder = load_table(config.sql_dump_path, "major_holder")
    foreign_holding = load_table(config.sql_dump_path, "foreign_holding")
    stock_meta = load_table(config.sql_dump_path, "stock_meta")
    period_df = load_table(config.sql_dump_path, "period")
    filter_flag = load_table(config.sql_dump_path, "filter_flag")

    feature_krx["ticker"] = feature_krx["ticker"].astype(str).str.zfill(6)
    labels["ticker"] = labels["ticker"].astype(str).str.zfill(6)
    major_holder["ticker"] = major_holder["ticker"].astype(str).str.zfill(6)
    stock_meta["ticker"] = stock_meta["ticker"].astype(str).str.zfill(6)
    if not filter_flag.empty:
        filter_flag["ticker"] = filter_flag["ticker"].astype(str).str.zfill(6)
        filter_flag["flag_date"] = filter_flag["flag_date"].astype(str)

    feature_krx["period"] = feature_krx["period"].astype(str)
    labels["period"] = labels["period"].astype(str)
    major_holder["period"] = major_holder["period"].astype(str)
    period_df["period"] = period_df["period"].astype(str)

    historical_periods = sorted(feature_krx["period"].dropna().unique().tolist(), key=period_sort_key)
    prev_period_map = {
        period: historical_periods[idx - 1] if idx > 0 else None
        for idx, period in enumerate(historical_periods)
    }

    frames: list[pd.DataFrame] = []
    for period in historical_periods:
        frame = feature_krx.loc[feature_krx["period"] == period].copy()
        label_frame = labels.loc[labels["period"] == period].copy()
        if frame.empty or label_frame.empty:
            continue

        frame = frame.merge(
            label_frame[["period", "ticker", "was_member", "label_in", "label_out", "actual_rank", "is_member"]],
            on=["period", "ticker"],
            how="left",
        )

        major_frame = major_holder.loc[major_holder["period"] == period].copy()
        frame = frame.merge(
            major_frame[
                ["period", "ticker", "major_holder_ratio", "treasury_ratio", "non_float_ratio", "float_rate"]
            ],
            on=["period", "ticker"],
            how="left",
        )

        foreign_agg = build_foreign_aggregate(
            foreign_holding=foreign_holding,
            period_frame=period_df,
            target_period=period,
            prev_period=prev_period_map[period],
        )
        frame = frame.merge(foreign_agg, on="ticker", how="left")

        frame = frame.merge(
            stock_meta[["ticker", "is_not_common", "is_reits", "list_date", "ksic_sector"]],
            on="ticker",
            how="left",
        )
        if not filter_flag.empty:
            flag_frame = filter_flag.loc[
                filter_flag["flag_date"] == period,
                ["ticker", "is_managed", "is_warning"],
            ].copy()
            frame = frame.merge(flag_frame, on="ticker", how="left")

        prev_period = prev_period_map[period]
        if prev_period is not None:
            prev_feature = feature_krx.loc[feature_krx["period"] == prev_period, ["ticker", "period_rank"]].copy()
            prev_feature = prev_feature.rename(columns={"period_rank": "prev_rank"})
            frame = frame.merge(prev_feature, on="ticker", how="left")
        else:
            frame["prev_rank"] = pd.NA

        frame["prev_was_member"] = pd.to_numeric(frame["was_member"], errors="coerce")
        frame["period_rank"] = pd.to_numeric(frame["period_rank"], errors="coerce")
        frame["prev_rank"] = pd.to_numeric(frame["prev_rank"], errors="coerce")
        frame["rank_change"] = frame["period_rank"] - frame["prev_rank"]

        frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce")
        period_row = period_df.loc[period_df["period"] == period]
        period_end = pd.to_datetime(period_row.iloc[0]["period_end"], errors="coerce") if not period_row.empty else pd.NaT
        frame["period_end"] = period_end
        frame["months_listed"] = (
            (frame["period_end"].dt.year - frame["list_date"].dt.year) * 12
            + (frame["period_end"].dt.month - frame["list_date"].dt.month)
        )

        frame = apply_eligibility_filter(frame, period_end_date=period_end)
        frame = frame.loc[frame["prev_was_member"].notna()].copy()

        frame["gics_sector"] = frame["gics_sector"].fillna("기타")
        frame["sector_rank"] = frame.groupby("gics_sector")["period_rank"].rank(method="first").astype(int)
        frame["sector_count"] = frame.groupby("gics_sector")["ticker"].transform("count")
        frame["sector_relative_rank"] = frame["sector_rank"] / frame["sector_count"]

        frame["float_rate"] = pd.to_numeric(frame["float_rate"], errors="coerce").fillna(0.0)
        frame["non_float_ratio"] = (
            pd.to_numeric(frame["major_holder_ratio"], errors="coerce").fillna(0.0)
            + pd.to_numeric(frame["treasury_ratio"], errors="coerce").fillna(0.0)
        )
        frame["float_mktcap"] = (
            pd.to_numeric(frame["avg_mktcap"], errors="coerce").fillna(0.0) * frame["float_rate"]
        )
        frame["float_mktcap_rank"] = frame["float_mktcap"].rank(method="first", ascending=False).astype(int)
        frame["dist_from_200"] = pd.to_numeric(frame["period_rank"], errors="coerce") - 200
        frame["float_dist_from_200"] = frame["float_mktcap_rank"] - 200

        numeric_fill_zero = [
            "major_holder_ratio",
            "treasury_ratio",
            "avg_foreign_ratio",
            "last_foreign_ratio",
            "foreign_change",
            "turnover_ratio",
            "avg_mktcap",
            "prev_was_member",
            "rank_change",
            "avg_exhaustion_rate",
        ]
        for column in numeric_fill_zero:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        frame["is_member"] = pd.to_numeric(frame["is_member"], errors="coerce").fillna(0).astype(int)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_v15_package() -> Path:
    config = load_config()
    project_root = config.project_root
    base_package_path = project_root / "data" / "raw" / "model_package.pkl"
    output_path = project_root / "data" / "raw" / "model_package_v1_5.pkl"

    with base_package_path.open("rb") as handle:
        base_package = pickle.load(handle)

    snapshot = build_historical_snapshot(config)
    if snapshot.empty:
        raise ValueError("historical snapshot is empty")

    train_periods = list(base_package.get("train_periods", []))
    test_periods = list(base_package.get("test_periods", []))
    train_df = snapshot.loc[snapshot["period"].isin(train_periods)].copy()
    if train_df.empty:
        raise ValueError("training snapshot is empty")

    X_train = train_df[V15_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_train = train_df["is_member"].astype(int)

    model = ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    v15_package = dict(base_package)
    v15_package["model"] = model
    v15_package["model_name"] = "ExtraTrees"
    v15_package["method"] = "ExtraTrees 14 features"
    v15_package["model_version"] = "v1.5"
    v15_package["features"] = list(V15_FEATURES)
    v15_package["created_at"] = datetime.now().isoformat(timespec="seconds")
    v15_package["train_periods"] = train_periods
    v15_package["test_periods"] = test_periods

    with output_path.open("wb") as handle:
        pickle.dump(v15_package, handle)

    return output_path


if __name__ == "__main__":
    path = build_v15_package()
    print(path)

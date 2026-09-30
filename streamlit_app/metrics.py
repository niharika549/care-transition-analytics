"""
Reusable KPI / metrics functions for the UAC Care Transition Analytics project.
Import this module from both your EDA notebook and your Streamlit app so the
logic only lives in one place.
"""
import numpy as np
import pandas as pd
from scipy import stats


def load_cleaned_data(path="cleaned_data.csv"):
    return pd.read_csv(path, parse_dates=["date"])


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Transfer Efficiency Ratio, Discharge Effectiveness, Pipeline
    Throughput, net flows, and cumulative backlog columns to df."""
    df = df.copy()
    df["transfer_efficiency_ratio"] = df["transferred_out_cbp"] / df["in_cbp_custody"]
    df["discharge_effectiveness"] = df["discharged_hhs"] / df["in_hhs_care"]

    df["total_entries"] = df["apprehended_cbp"] + df["transferred_out_cbp"]
    df["total_exits"] = df["transferred_out_cbp"] + df["discharged_hhs"]
    df["pipeline_throughput"] = df["total_exits"] / df["total_entries"]

    df["net_cbp_flow"] = df["apprehended_cbp"] - df["transferred_out_cbp"]
    df["net_hhs_flow"] = df["transferred_out_cbp"] - df["discharged_hhs"]
    df["cbp_backlog_cum"] = df["net_cbp_flow"].cumsum()
    df["hhs_backlog_cum"] = df["net_hhs_flow"].cumsum()

    df["discharge_eff_roll30"] = df["discharge_effectiveness"].rolling(30, min_periods=10).mean()
    df["discharge_eff_roll_std30"] = df["discharge_effectiveness"].rolling(30, min_periods=10).std()
    df["transfer_eff_roll30"] = df["transfer_efficiency_ratio"].rolling(30, min_periods=10).mean()
    return df


def detect_backlog_periods(df: pd.DataFrame, min_days: int = 10) -> pd.DataFrame:
    """Finds runs of consecutive reporting days where net_hhs_flow > 0
    (HHS care load growing faster than it's being discharged)."""
    d = df.copy()
    d["backlog_flag"] = d["net_hhs_flow"] > 0
    d["grp"] = (d["backlog_flag"] != d["backlog_flag"].shift()).cumsum()
    runs = (
        d[d["backlog_flag"]]
        .groupby("grp")
        .agg(start=("date", "min"), end=("date", "max"),
             days=("date", "count"), total_backlog_growth=("net_hhs_flow", "sum"))
    )
    return runs[runs["days"] >= min_days].sort_values("days", ascending=False)


def detect_outcome_drops(df: pd.DataFrame, std_threshold: float = 2.0) -> pd.DataFrame:
    """Flags days where discharge effectiveness fell more than
    `std_threshold` rolling-standard-deviations below its 30-day rolling mean."""
    d = df.copy()
    d["drop_flag"] = d["discharge_effectiveness"] < (
        d["discharge_eff_roll30"] - std_threshold * d["discharge_eff_roll_std30"]
    )
    return d[d["drop_flag"]][["date", "discharge_effectiveness", "discharge_eff_roll30"]]


def weekday_weekend_ttest(df: pd.DataFrame, column: str = "discharge_effectiveness"):
    """Independent-samples t-test comparing weekday vs weekend values of `column`.
    Returns (weekday_mean, weekend_mean, t_stat, p_value)."""
    weekday_vals = df.loc[~df["is_weekend"], column].dropna()
    weekend_vals = df.loc[df["is_weekend"], column].dropna()
    if len(weekday_vals) < 2 or len(weekend_vals) < 2:
        return weekday_vals.mean(), weekend_vals.mean(), float("nan"), float("nan")
    t_stat, p_value = stats.ttest_ind(weekday_vals, weekend_vals, equal_var=False)
    return weekday_vals.mean(), weekend_vals.mean(), t_stat, p_value


def forecast_linear(df: pd.DataFrame, column: str = "discharge_effectiveness", horizon_days: int = 30):
    """Simple linear-trend forecast of `column` for the next `horizon_days` days.
    Returns a DataFrame with columns: date, forecast (fitted on the full history's
    day-index vs value, extrapolated forward). Uses only rows where column is not NaN."""
    d = df.dropna(subset=[column]).copy()
    if len(d) < 5:
        return pd.DataFrame(columns=["date", "forecast"])
    x = np.arange(len(d))
    y = d[column].values
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(len(d), len(d) + horizon_days)
    future_dates = pd.date_range(d["date"].max() + pd.Timedelta(days=1), periods=horizon_days)
    forecast_vals = slope * future_x + intercept
    return pd.DataFrame({"date": future_dates, "forecast": forecast_vals}), slope


def kpi_status(value: float, good_threshold: float, warn_threshold: float, higher_is_better: bool = True) -> str:
    """Returns 'green', 'amber', or 'red' for a KPI scorecard given two thresholds."""
    if pd.isna(value):
        return "amber"
    if higher_is_better:
        if value >= good_threshold:
            return "green"
        elif value >= warn_threshold:
            return "amber"
        return "red"
    else:
        if value <= good_threshold:
            return "green"
        elif value <= warn_threshold:
            return "amber"
        return "red"


def compute_pipeline_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """Composite 0-100 'Pipeline Health Index' synthesizing all three KPIs plus
    backlog behavior into a single interpretable score. Not part of the brief's
    KPI list — an original synthesis metric.

    Weighting rationale (documented for the paper):
      - Discharge Effectiveness: 35% — the metric closest to the program's actual
        humanitarian outcome (children successfully reunified with a sponsor).
      - Transfer Efficiency Ratio: 25% — speed of the first pipeline stage.
      - Pipeline Throughput: 25% — whether the system is keeping pace overall.
      - Backlog Stability: 15% — inverse of same-day net HHS backlog growth;
        penalizes days actively accumulating backlog.
    Each component is min-max normalized within the given df's own range before
    weighting, so the index is relative to the selected date range.
    """
    d = df.copy()

    def norm(s):
        s = s.astype(float)
        lo, hi = s.min(), s.max()
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)

    n_transfer = norm(d["transfer_efficiency_ratio"])
    n_discharge = norm(d["discharge_effectiveness"])
    n_throughput = norm(d["pipeline_throughput"].clip(upper=d["pipeline_throughput"].quantile(0.99)))
    # backlog stability: invert net_hhs_flow so lower backlog growth = higher score
    n_backlog = 1 - norm(d["net_hhs_flow"])

    d["health_index"] = 100 * (
        0.35 * n_discharge + 0.25 * n_transfer + 0.25 * n_throughput + 0.15 * n_backlog
    )
    return d


def quantified_recommendations(df: pd.DataFrame) -> dict:
    """Translates statistical findings into estimated real-world impact numbers
    for use in policy recommendations."""
    weekday = df.loc[~df["is_weekend"]]
    weekend = df.loc[df["is_weekend"]]
    weekend_rate = weekend["discharge_effectiveness"].mean()
    weekday_gap = weekend_rate - weekday["discharge_effectiveness"]
    potential_extra_discharges = (weekday_gap.clip(lower=0) * weekday["in_hhs_care"]).sum()

    backlog_periods = detect_backlog_periods(df, min_days=1)
    total_backlog_growth = backlog_periods["total_backlog_growth"].sum() if not backlog_periods.empty else 0.0

    return {
        "weekday_mean": weekday["discharge_effectiveness"].mean(),
        "weekend_mean": weekend_rate,
        "potential_extra_discharges_if_weekday_matched_weekend": round(potential_extra_discharges),
        "total_backlog_growth_across_periods": round(total_backlog_growth),
    }

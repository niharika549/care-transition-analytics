"""
Care Transition Efficiency & Placement Outcome Analytics — Streamlit Dashboard
Run locally with: streamlit run app.py
Requires: cleaned_data.csv and metrics.py in the same folder.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from metrics import (
    load_cleaned_data, compute_kpis, detect_backlog_periods, detect_outcome_drops,
    weekday_weekend_ttest, forecast_linear, kpi_status,
    compute_pipeline_health_index, quantified_recommendations,
)

st.set_page_config(page_title="UAC Care Transition Analytics", layout="wide")

# ---------- Navy/Black dark theme (CSS + Plotly template) ----------
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #05070F 0%, #0A0E1A 45%, #0F1B33 100%); }
    section[data-testid="stSidebar"] { background-color: #05070F; border-right: 1px solid #1C2942; }
    h1, h2, h3, h4, .stMarkdown p { color: #E8ECF4 !important; }
    div[data-testid="stMetric"] {
        background-color: #0F1B33; border: 1px solid #1C2942; border-radius: 10px;
        padding: 12px 14px;
    }
    div[data-testid="stMetricValue"] { color: #4C8BF5 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #0F1B33; border-radius: 8px 8px 0 0; color: #B8C2D9;
    }
    .stTabs [aria-selected="true"] { background-color: #1C2942 !important; color: #4C8BF5 !important; }
    div[data-testid="stExpander"], div[data-testid="stDataFrame"] { background-color: #0F1B33; }
    .stAlert { background-color: #0F1B33; }
    hr { border-color: #1C2942; }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,27,51,0.4)",
    font=dict(color="#E8ECF4"),
)

def style_fig(fig):
    fig.update_layout(**PLOTLY_DARK)
    return fig

# ---------- Load & prepare data ----------
@st.cache_data
def get_data():
    df = load_cleaned_data("cleaned_data.csv")
    df = compute_kpis(df)
    df = compute_pipeline_health_index(df)
    return df

df = get_data()

st.title("Care Transition Efficiency & Placement Outcome Analytics")
st.caption("HHS Unaccompanied Alien Children (UAC) Program — process efficiency dashboard")

st.markdown(
    "The UAC Program moves children through a multi-stage pipeline — apprehension and CBP custody, "
    "transfer to HHS care, sheltering and case management, and discharge to a vetted sponsor. This "
    "dashboard measures **how efficiently** children move through that pipeline: transfer and discharge "
    "speed, where backlogs accumulate, and whether placement outcomes are improving or deteriorating over time."
)
st.divider()

# ---------- Sidebar: User Capabilities ----------
st.sidebar.header("Controls")

min_date, max_date = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
fdf = df.loc[mask].copy()

st.sidebar.subheader("Ratio toggles")
show_transfer = st.sidebar.checkbox("Transfer Efficiency Ratio", value=True)
show_discharge = st.sidebar.checkbox("Discharge Effectiveness", value=True)
show_throughput = st.sidebar.checkbox("Pipeline Throughput", value=False)

st.sidebar.subheader("Alert thresholds")
backlog_min_days = st.sidebar.slider("Backlog: min consecutive days to flag", 3, 30, 10)
stability_std = st.sidebar.slider("Outcome drop: std-dev threshold", 1.0, 4.0, 2.0, 0.5)

# ---------- Alert banner ----------
recent = fdf.tail(30)
if not recent.empty:
    latest_eff = recent["discharge_effectiveness"].iloc[-1]
    latest_roll = recent["discharge_eff_roll30"].iloc[-1]
    latest_std = recent["discharge_eff_roll_std30"].iloc[-1]
    if pd.notna(latest_roll) and pd.notna(latest_std) and latest_eff < (latest_roll - stability_std * latest_std):
        st.warning(
            f"⚠️ Latest discharge effectiveness ({latest_eff:.3f}) is more than "
            f"{stability_std} std devs below its 30-day rolling average ({latest_roll:.3f})."
        )

tabs = st.tabs(["Pipeline Flow", "Efficiency Panels", "Bottleneck Detection", "Outcome Trends", "Health Index & Recommendations"])

# ---------- KPI Scorecard ----------
st.subheader("System Health Scorecard")
status_colors = {"green": "🟢", "amber": "🟡", "red": "🔴"}

avg_transfer = fdf["transfer_efficiency_ratio"].mean()
avg_discharge = fdf["discharge_effectiveness"].mean()
avg_throughput = fdf["pipeline_throughput"].mean()
n_backlogs = len(detect_backlog_periods(fdf, min_days=backlog_min_days))

s1 = kpi_status(avg_transfer, good_threshold=0.7, warn_threshold=0.5, higher_is_better=True)
s2 = kpi_status(avg_discharge, good_threshold=0.03, warn_threshold=0.015, higher_is_better=True)
s3 = kpi_status(avg_throughput, good_threshold=1.0, warn_threshold=0.85, higher_is_better=True)
s4 = kpi_status(n_backlogs, good_threshold=0, warn_threshold=2, higher_is_better=False)

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric(f"{status_colors[s1]} Transfer Efficiency", f"{avg_transfer:.2f}")
sc2.metric(f"{status_colors[s2]} Discharge Effectiveness", f"{avg_discharge:.3f}")
sc3.metric(f"{status_colors[s3]} Pipeline Throughput", f"{avg_throughput:.2f}")
sc4.metric(f"{status_colors[s4]} Backlog Periods", f"{n_backlogs}")
st.caption("Scorecard thresholds are illustrative defaults — calibrate against agency-defined targets before operational use.")
st.divider()

# ---------- Tab 1: Pipeline Flow Visualization ----------
with tabs[0]:
    st.subheader("Care Pipeline Flow")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg daily apprehended", f"{fdf['apprehended_cbp'].mean():.1f}")
    c2.metric("Avg CBP custody load", f"{fdf['in_cbp_custody'].mean():.0f}")
    c3.metric("Avg HHS care load", f"{fdf['in_hhs_care'].mean():.0f}")
    c4.metric("Avg daily discharged", f"{fdf['discharged_hhs'].mean():.1f}")

    # --- Sankey: CBP custody -> HHS care -> Sponsor placement ---
    total_apprehended = fdf["apprehended_cbp"].sum()
    total_transferred = fdf["transferred_out_cbp"].sum()
    total_discharged = fdf["discharged_hhs"].sum()
    still_in_cbp = max(total_apprehended - total_transferred, 0)
    still_in_hhs = max(total_transferred - total_discharged, 0)

    sankey = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=18,
            label=["Apprehended (CBP)", "CBP Custody", "HHS Care", "Discharged (Sponsor)", "Remaining in CBP", "Remaining in HHS"],
            color=["#4C78A8", "#4C78A8", "#F58518", "#54A24B", "#B0B0B0", "#B0B0B0"],
        ),
        link=dict(
            source=[0, 1, 2, 1, 2],
            target=[1, 2, 3, 4, 5],
            value=[total_apprehended, total_transferred, total_discharged, still_in_cbp, still_in_hhs],
            color=["rgba(76,120,168,0.4)", "rgba(245,133,24,0.4)", "rgba(84,162,75,0.4)",
                   "rgba(176,176,176,0.3)", "rgba(176,176,176,0.3)"],
        ),
    ))
    sankey.update_layout(title="Care Pipeline Flow: Apprehension → CBP Custody → HHS Care → Sponsor Placement", height=420)
    st.plotly_chart(style_fig(sankey), width='stretch')
    st.info(
        f"💡 Over the selected period, **{total_apprehended:,.0f}** children were apprehended, "
        f"**{total_transferred:,.0f}** transferred to HHS, and **{total_discharged:,.0f}** discharged to sponsors — "
        f"a total-period discharge rate of **{(total_discharged / total_apprehended * 100 if total_apprehended else 0):.1f}%** of apprehensions."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["in_cbp_custody"], name="In CBP Custody"))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["in_hhs_care"], name="In HHS Care", yaxis="y2"))
    fig.update_layout(
        title="CBP Custody Load vs HHS Care Load Over Time",
        yaxis=dict(title="In CBP Custody"),
        yaxis2=dict(title="In HHS Care", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=450,
    )
    st.plotly_chart(style_fig(fig), width='stretch')

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=fdf["date"], y=fdf["apprehended_cbp"], name="Apprehended"))
    fig2.add_trace(go.Bar(x=fdf["date"], y=fdf["transferred_out_cbp"], name="Transferred to HHS"))
    fig2.add_trace(go.Bar(x=fdf["date"], y=fdf["discharged_hhs"], name="Discharged (sponsor placement)"))
    fig2.update_layout(title="Daily Flow Volumes", barmode="overlay", height=400)
    st.plotly_chart(style_fig(fig2), width='stretch')

    st.download_button(
        "⬇ Download filtered data (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="uac_filtered_data.csv",
        mime="text/csv",
    )

# ---------- Tab 2: Transfer & Discharge Efficiency Panels ----------
with tabs[1]:
    st.subheader("Transition Efficiency Metrics")
    fig = go.Figure()
    if show_transfer:
        fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["transfer_eff_roll30"], name="Transfer Efficiency Ratio (30d avg)"))
    if show_discharge:
        fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["discharge_eff_roll30"], name="Discharge Effectiveness (30d avg)", yaxis="y2"))
    if show_throughput:
        fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["pipeline_throughput"], name="Pipeline Throughput", mode="markers", opacity=0.4))
    fig.update_layout(
        title="Efficiency Ratios Over Time",
        yaxis=dict(title="Transfer Efficiency Ratio"),
        yaxis2=dict(title="Discharge Effectiveness", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=450,
    )
    st.plotly_chart(style_fig(fig), width='stretch')

    colA, colB = st.columns(2)
    with colA:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wk = fdf.groupby("dow")[["transfer_efficiency_ratio", "discharge_effectiveness"]].mean().reindex(order)
        fig3 = px.bar(wk, y="discharge_effectiveness", title="Avg Discharge Effectiveness by Day of Week")
        st.plotly_chart(style_fig(fig3), width='stretch')
    with colB:
        monthly = fdf.groupby("year_month")["discharge_effectiveness"].mean().reset_index()
        fig4 = px.line(monthly, x="year_month", y="discharge_effectiveness", markers=True,
                        title="Month-over-Month Discharge Effectiveness")
        st.plotly_chart(style_fig(fig4), width='stretch')

    if not wk.empty and wk["discharge_effectiveness"].notna().any():
        best_day = wk["discharge_effectiveness"].idxmax()
        worst_day = wk["discharge_effectiveness"].idxmin()
        st.info(
            f"💡 **{best_day}** has the highest average discharge effectiveness in this range, "
            f"**{worst_day}** the lowest. Avg Transfer Efficiency Ratio: **{fdf['transfer_efficiency_ratio'].mean():.2f}**, "
            f"Avg Discharge Effectiveness: **{fdf['discharge_effectiveness'].mean():.3f}**."
        )

    st.markdown("#### Statistical test: Weekday vs Weekend discharge effectiveness")
    wd_mean, we_mean, t_stat, p_val = weekday_weekend_ttest(fdf, "discharge_effectiveness")
    tcol1, tcol2, tcol3 = st.columns(3)
    tcol1.metric("Weekday mean", f"{wd_mean:.4f}")
    tcol2.metric("Weekend mean", f"{we_mean:.4f}")
    tcol3.metric("p-value", f"{p_val:.4f}" if pd.notna(p_val) else "n/a")
    if pd.notna(p_val):
        if p_val < 0.05:
            st.success(f"✅ Statistically significant difference (p = {p_val:.4f} < 0.05, Welch's t-test). The weekday/weekend gap is unlikely to be random noise.")
        else:
            st.warning(f"⚠️ Not statistically significant (p = {p_val:.4f} ≥ 0.05). The observed gap could plausibly be due to chance in this date range.")

# ---------- Tab 3: Bottleneck Detection ----------
with tabs[2]:
    st.subheader("Backlog & Bottleneck Detection")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["cbp_backlog_cum"], name="Cumulative Net CBP Flow"))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["hhs_backlog_cum"], name="Cumulative Net HHS Flow"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    backlog_periods = detect_backlog_periods(fdf, min_days=backlog_min_days)
    for _, row in backlog_periods.iterrows():
        fig.add_vrect(x0=row["start"], x1=row["end"], fillcolor="red", opacity=0.15, line_width=0)

    fig.update_layout(title="Cumulative Backlog (shaded = detected backlog period)", height=450)
    st.plotly_chart(style_fig(fig), width='stretch')

    st.markdown(f"**{len(backlog_periods)} backlog period(s) detected** (≥{backlog_min_days} consecutive days of HHS care load growing faster than discharges):")
    st.dataframe(backlog_periods, width='stretch')

    if not backlog_periods.empty:
        longest = backlog_periods.loc[backlog_periods["days"].idxmax()]
        st.info(
            f"💡 The longest backlog period ran **{longest['days']} days** "
            f"({longest['start'].date() if hasattr(longest['start'],'date') else longest['start']} to "
            f"{longest['end'].date() if hasattr(longest['end'],'date') else longest['end']}), "
            f"accumulating a net backlog growth of **{longest['total_backlog_growth']:,.0f}**."
        )
    else:
        st.info("💡 No sustained backlog period found at the current threshold — try lowering the slider.")

    st.download_button(
        "⬇ Download backlog periods (CSV)",
        data=backlog_periods.to_csv().encode("utf-8"),
        file_name="backlog_periods.csv",
        mime="text/csv",
    )

# ---------- Tab 4: Outcome Trend Analysis ----------
with tabs[3]:
    st.subheader("Placement Outcome Stability")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["discharge_effectiveness"], name="Discharge Effectiveness (daily)", opacity=0.3))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["discharge_eff_roll30"], name="30-day rolling avg"))

    forecast_result = forecast_linear(fdf, "discharge_effectiveness", horizon_days=30)
    if isinstance(forecast_result, tuple):
        forecast_df, slope = forecast_result
        if not forecast_df.empty:
            fig.add_trace(go.Scatter(
                x=forecast_df["date"], y=forecast_df["forecast"], name="30-day forecast (linear trend)",
                line=dict(dash="dash", color="red"),
            ))
    else:
        forecast_df, slope = pd.DataFrame(), 0

    fig.update_layout(title="Discharge Effectiveness: Daily, Rolling Trend & 30-Day Forecast", height=420)
    st.plotly_chart(style_fig(fig), width='stretch')

    if not forecast_df.empty:
        direction = "improving 📈" if slope > 0 else ("deteriorating 📉" if slope < 0 else "flat ➡️")
        st.info(
            f"💡 Linear trend over the selected range is **{direction}** "
            f"(slope ≈ {slope:.6f}/day). Projected discharge effectiveness in 30 days: "
            f"**{forecast_df['forecast'].iloc[-1]:.4f}**, vs current 30-day avg of **{fdf['discharge_eff_roll30'].dropna().iloc[-1] if fdf['discharge_eff_roll30'].notna().any() else float('nan'):.4f}**. "
            "Note: a simple linear extrapolation — treat as directional, not a precise prediction."
        )

    fig2 = px.line(fdf, x="date", y="discharge_eff_roll_std30", title="Outcome Volatility (30-day rolling std dev)")
    st.plotly_chart(style_fig(fig2), width='stretch')

    drops = detect_outcome_drops(fdf, std_threshold=stability_std)
    st.markdown(f"**{len(drops)} sudden-drop date(s) flagged** (discharge effectiveness > {stability_std} std devs below rolling mean):")
    st.dataframe(drops, width='stretch')

    avg_vol = fdf["discharge_eff_roll_std30"].mean()
    st.info(
        f"💡 Average 30-day outcome volatility over this range: **{avg_vol:.4f}**. "
        f"{'This period shows elevated instability — consider narrowing the date range to investigate.' if len(drops) > 3 else 'Outcomes look relatively stable at the current threshold.'}"
    )

    st.download_button(
        "⬇ Download flagged drop dates (CSV)",
        data=drops.to_csv(index=False).encode("utf-8"),
        file_name="outcome_drops.csv",
        mime="text/csv",
    )

# ---------- Tab 5: Pipeline Health Index & Quantified Recommendations ----------
with tabs[4]:
    st.subheader("Pipeline Health Index")
    st.caption(
        "An original composite score (0–100) synthesizing all three brief KPIs plus backlog behavior "
        "into one number: 35% Discharge Effectiveness, 25% Transfer Efficiency Ratio, 25% Pipeline "
        "Throughput, 15% Backlog Stability. Weighted toward discharge effectiveness since that's the "
        "outcome closest to the program's actual humanitarian goal — successful sponsor reunification. "
        "Each component is normalized within the selected date range."
    )

    current_index = fdf["health_index"].mean()
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_index,
        title={"text": "Avg Pipeline Health Index (selected range)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#4C8BF5"},
            "steps": [
                {"range": [0, 33], "color": "#3A1414"},
                {"range": [33, 66], "color": "#3A2E14"},
                {"range": [66, 100], "color": "#14311A"},
            ],
        },
    ))
    gauge.update_layout(height=320)
    st.plotly_chart(style_fig(gauge), width='stretch')

    fig_hi = go.Figure()
    fig_hi.add_trace(go.Scatter(x=fdf["date"], y=fdf["health_index"], name="Daily Health Index", opacity=0.3))
    fig_hi.add_trace(go.Scatter(x=fdf["date"], y=fdf["health_index"].rolling(30, min_periods=10).mean(), name="30-day rolling avg"))
    fig_hi.update_layout(title="Pipeline Health Index Over Time", height=380)
    st.plotly_chart(style_fig(fig_hi), width='stretch')

    st.divider()
    st.subheader("Quantified Policy Recommendations")
    recs = quantified_recommendations(fdf)
    rc1, rc2 = st.columns(2)
    rc1.metric(
        "Potential extra discharges",
        f"+{recs['potential_extra_discharges_if_weekday_matched_weekend']:,.0f}",
        help="Estimated additional children discharged over this period if weekday discharge "
             "effectiveness matched the (statistically significantly higher) weekend rate.",
    )
    rc2.metric(
        "Total backlog growth",
        f"{recs['total_backlog_growth_across_periods']:,.0f}",
        help="Sum of net HHS care-load growth across all days where transfers into HHS outpaced discharges.",
    )
    st.info(
        f"💡 **Recommendation:** Weekday discharge effectiveness (avg {recs['weekday_mean']:.4f}) "
        f"significantly trails the weekend rate (avg {recs['weekend_mean']:.4f}, p = {weekday_weekend_ttest(fdf)[3]:.4f}). "
        f"If weekday case-processing capacity matched weekend levels, an estimated "
        f"**{recs['potential_extra_discharges_if_weekday_matched_weekend']:,.0f} additional children** "
        "could have been discharged to sponsors over this period — suggesting a concrete staffing/"
        "workflow investigation target: what enables faster case resolution on weekends, and can it "
        "be replicated on weekdays?"
    )

st.caption("Data source: HHS Unaccompanied Alien Children Program daily reports.")

# ---------- Data limitations footer ----------
_full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
_missing_days = len(_full_range) - len(df)
st.divider()
st.caption(
    f"**Data limitations:** This dataset covers {len(df)} reporting days out of "
    f"{len(_full_range)} calendar days in range ({_missing_days} days have no report, "
    f"concentrated on Fridays/Saturdays) — treat gaps as missing reports, not zero activity. "
    f"System Health Scorecard thresholds (green/amber/red) are illustrative defaults set for this "
    f"analysis, not official HHS/ORR performance targets. The 30-day forecast on the Outcome Trends "
    f"tab is a simple linear extrapolation and should be read as directional, not predictive."
)

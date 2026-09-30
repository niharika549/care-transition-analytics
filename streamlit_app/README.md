# UAC Care Transition Analytics — Streamlit Dashboard

## Folder contents
- `app.py` — the dashboard
- `metrics.py` — KPI/backlog/stability calculation functions (imported by app.py)
- `cleaned_data.csv` — the cleaned dataset
- `requirements.txt` — Python packages needed

## How to run (Windows, PowerShell)

1. Open PowerShell and navigate to this folder:
   ```
   cd path\to\this\folder
   ```
2. (Recommended) create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```
5. It will open automatically in your browser at `http://localhost:8501`. If not, open that URL manually.

## What's in the dashboard
- **Pipeline Flow** tab — CBP custody vs HHS care load over time, daily flow volumes
- **Efficiency Panels** tab — Transfer Efficiency Ratio & Discharge Effectiveness trends, weekday/monthly breakdowns
- **Bottleneck Detection** tab — cumulative backlog chart with detected backlog periods shaded, adjustable via sidebar slider
- **Outcome Trends** tab — discharge effectiveness stability, volatility, and flagged sudden-drop dates

## Sidebar controls (matches the brief's "User Capabilities")
- Date range selection
- Ratio-based metric toggles (show/hide each efficiency ratio)
- Threshold-based alerts — sliders control backlog detection sensitivity and outcome-drop sensitivity; a warning banner appears at the top when the latest data crosses the threshold

## Notes for your submission
- Take screenshots of each tab for your research paper / executive summary.
- If deploying (e.g., Streamlit Community Cloud) instead of just running locally, push this folder to a GitHub repo and connect it there — no code changes needed.

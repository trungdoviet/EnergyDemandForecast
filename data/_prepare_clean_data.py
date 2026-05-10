"""
Prepare clean monthly CSVs for the Taiwan Energy Demand V3 notebook.

Inputs (already downloaded into this folder):
- ember_monthly_full.csv     (Ember monthly electricity, all countries; symlinked from V2)
- tw_inflation_imf.json      (IMF Datamapper inflation %, annual; drop forecast years > 2025)
- tw_gdp_imf.json            (IMF Datamapper GDP per capita PPP, annual)
- tw_gdpgrowth_imf.json      (IMF Datamapper real GDP growth %, annual)
- tw_temp_taipei.json        (NASA POWER monthly avg temperature)
- tw_temp_taichung.json
- tw_temp_kaohsiung.json

Outputs:
- tw_electricity_monthly.csv     (Date, demand_twh, fossil_share_pct, clean_share_pct, co2_intensity)
- tw_temperature_monthly.csv     (Date, t_taipei, t_taichung, t_kaohsiung, t_national_avg, cooling_degrees)
- tw_macro_annual.csv            (Year, gdp_per_capita_usd, gdp_growth_pct, inflation_pct)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent
OUT_DIR = DATA_DIR


def prepare_electricity():
    df = pd.read_csv(DATA_DIR / "ember_monthly_full.csv")
    tw = df[df["ISO 3 code"] == "TWN"].copy()
    tw["Date"] = pd.to_datetime(tw["Date"])

    demand = tw[(tw["Category"] == "Electricity demand") & (tw["Unit"] == "TWh")][
        ["Date", "Value"]
    ].rename(columns={"Value": "demand_twh"})

    fossil_share = tw[
        (tw["Category"] == "Electricity generation")
        & (tw["Subcategory"] == "Aggregate fuel")
        & (tw["Variable"] == "Fossil")
        & (tw["Unit"] == "%")
    ][["Date", "Value"]].rename(columns={"Value": "fossil_share_pct"})

    clean_share = tw[
        (tw["Category"] == "Electricity generation")
        & (tw["Subcategory"] == "Aggregate fuel")
        & (tw["Variable"] == "Clean")
        & (tw["Unit"] == "%")
    ][["Date", "Value"]].rename(columns={"Value": "clean_share_pct"})

    co2 = tw[
        (tw["Category"] == "Power sector emissions")
        & (tw["Subcategory"] == "CO2 intensity")
    ][["Date", "Value"]].rename(columns={"Value": "co2_intensity_g_per_kwh"})

    out = (
        demand.merge(fossil_share, on="Date", how="left")
        .merge(clean_share, on="Date", how="left")
        .merge(co2, on="Date", how="left")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    out.to_csv(OUT_DIR / "tw_electricity_monthly.csv", index=False)
    print(f"electricity: {out.shape}, {out.Date.min().date()} -> {out.Date.max().date()}")
    return out


def _nasa_to_df(path: Path, col: str) -> pd.DataFrame:
    raw = json.loads(path.read_text())["properties"]["parameter"]["T2M"]
    rows = []
    for k, v in raw.items():
        if k.endswith("13"):  # NASA annual mean
            continue
        year = int(k[:4])
        month = int(k[4:])
        if not (1 <= month <= 12):
            continue
        rows.append({"Date": pd.Timestamp(year=year, month=month, day=1), col: float(v)})
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def prepare_temperature():
    taipei = _nasa_to_df(DATA_DIR / "tw_temp_taipei.json", "t_taipei_c")
    taichung = _nasa_to_df(DATA_DIR / "tw_temp_taichung.json", "t_taichung_c")
    kaohsiung = _nasa_to_df(DATA_DIR / "tw_temp_kaohsiung.json", "t_kaohsiung_c")
    out = taipei.merge(taichung, on="Date").merge(kaohsiung, on="Date")
    out["t_national_avg_c"] = out[["t_taipei_c", "t_taichung_c", "t_kaohsiung_c"]].mean(axis=1)
    out["cooling_degrees"] = (out["t_national_avg_c"] - 24).clip(lower=0)
    out["heating_degrees"] = (18 - out["t_national_avg_c"]).clip(lower=0)
    out.to_csv(OUT_DIR / "tw_temperature_monthly.csv", index=False)
    print(f"temperature: {out.shape}, {out.Date.min().date()} -> {out.Date.max().date()}")
    return out


def _imf_series(path: Path, code: str) -> dict:
    return json.loads(path.read_text())["values"][code]["TWN"]


def prepare_macro_annual():
    inf = _imf_series(DATA_DIR / "tw_inflation_imf.json", "PCPIPCH")
    gdp_pc = _imf_series(DATA_DIR / "tw_gdp_imf.json", "NGDPDPC")
    gdp_g = _imf_series(DATA_DIR / "tw_gdpgrowth_imf.json", "NGDP_RPCH")

    rows = []
    years = sorted({int(y) for y in (set(inf) | set(gdp_pc) | set(gdp_g))})
    for y in years:
        if y < 1995 or y > 2025:  # drop forecast years
            continue
        rows.append({
            "Year": y,
            "inflation_pct": float(inf.get(str(y))) if str(y) in inf else np.nan,
            "gdp_per_capita_usd": float(gdp_pc.get(str(y))) if str(y) in gdp_pc else np.nan,
            "gdp_growth_pct": float(gdp_g.get(str(y))) if str(y) in gdp_g else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "tw_macro_annual.csv", index=False)
    print(f"macro: {out.shape}, {out.Year.min()} -> {out.Year.max()}")
    return out


if __name__ == "__main__":
    e = prepare_electricity()
    t = prepare_temperature()
    m = prepare_macro_annual()
    print("\n--- sanity ---")
    print("elec head:\n", e.head(3))
    print("\ntemp head:\n", t.head(3))
    print("\nmacro head:\n", m.head(3))
    print("\nmacro tail:\n", m.tail(3))

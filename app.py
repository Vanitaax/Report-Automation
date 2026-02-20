import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import importlib.util
import re
from pbi_client import (
    run_dax,
    get_last_dax_error,
    get_wind_forecast,
    get_solar_forecast,
    get_niv_forecast,
    get_ic_flows,
    get_temperature_forecast,
    get_availability_matrix,
    get_price_curve,
)

AVAILABILITY_UNITS_FILTER = [
    "CDCL-1",
    "CNQPS-1",
    "CNQPS-2",
    "CNQPS-3",
    "CNQPS-4",
    "EECL-1",
    "GRAI-6",
    "GRAI-7",
    "GRAI-8",
    "KILLPG-1",
    "KILLPG-2",
    "TAYL2G",
    "TAYL3G",
]

UNIT_STATUS_REQUIRED_COLS = ["UnitName", "WD Status", "DA Schedule", "DA + 1 Schedule", "DA + 2 Schedule"]
UNIT_STATUS_OPTIONS = [
    "",
    "Baseload",
    "Standing",
    "BOA'd On + DP",
    "BOA'd Thru + DP",
    "R/T & MP",
    "STOR",
    "Unavailable",
    "Outage",
    "Offline",
]
UNIT_STATUS_STORE_PATH = Path(".data") / "unit_status_manual.csv"
EXPORT_DIR = Path(r"C:\Users\V01013\OneDrive - Uniper SE\Documents\Report")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _read_unit_status_store() -> pd.DataFrame:
    if not UNIT_STATUS_STORE_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(UNIT_STATUS_STORE_PATH)
    except Exception:
        return pd.DataFrame()


def _load_manual_status_for_date(save_date: str, selected_time: str | None) -> pd.DataFrame:
    store = _read_unit_status_store()
    if store.empty or "SaveDate" not in store.columns:
        return pd.DataFrame(columns=UNIT_STATUS_REQUIRED_COLS)
    out = store[store["SaveDate"].astype(str) == str(save_date)].copy()
    if "Select Time" in out.columns:
        time_key = "" if selected_time is None else str(selected_time)
        out = out[out["Select Time"].fillna("").astype(str) == time_key]
    keep = [c for c in UNIT_STATUS_REQUIRED_COLS if c in out.columns]
    if not keep:
        return pd.DataFrame(columns=UNIT_STATUS_REQUIRED_COLS)
    return out[keep].drop_duplicates(subset=["UnitName"], keep="last")


def _save_manual_status_for_date(save_date: str, selected_time: str | None, status_df: pd.DataFrame) -> None:
    UNIT_STATUS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = status_df[UNIT_STATUS_REQUIRED_COLS].copy()
    for c in UNIT_STATUS_REQUIRED_COLS[1:]:
        clean[c] = clean[c].fillna("").astype(str)
    clean["SaveDate"] = str(save_date)
    clean["Select Time"] = "" if selected_time is None else str(selected_time)
    clean["SavedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    old = _read_unit_status_store()
    if old.empty:
        merged = clean
    else:
        if "SaveDate" not in old.columns:
            old["SaveDate"] = ""
        if "Select Time" not in old.columns:
            old["Select Time"] = ""
        keep_old = ~(
            (old["SaveDate"].astype(str) == str(save_date))
            & (old["Select Time"].fillna("").astype(str) == ("" if selected_time is None else str(selected_time)))
        )
        merged = pd.concat([old.loc[keep_old], clean], ignore_index=True)
    merged.to_csv(UNIT_STATUS_STORE_PATH, index=False)


def _norm_unit(value: str) -> str:
    return str(value).strip().upper().replace("_", "").replace("-", "").replace(" ", "")


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.strip()
    )
    cleaned = cleaned.replace({"": None, "nan": None, "None": None, "null": None})
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_prices_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map: dict[str, str] = {}
    for c in out.columns:
        s = str(c).strip()
        low = s.lower()
        if "type" in low:
            rename_map[c] = "Type"
        elif "sum of wd" in low:
            rename_map[c] = "Sum of WD"
        elif "sum of da" in low:
            rename_map[c] = "Sum of DA"
        elif "sum of weekend" in low:
            rename_map[c] = "Sum of Weekend"
        elif "week ahead" in low and "dark" not in low and "spark" not in low:
            rename_map[c] = "Week Ahead"
        elif "da spark" in low:
            rename_map[c] = "DA Spark"
        elif "da dark" in low:
            rename_map[c] = "DA Dark"
        elif "wknd spark" in low or "wkend spark" in low:
            rename_map[c] = "WKND Spark"
        elif "wknd dark" in low or "wkend dark" in low:
            rename_map[c] = "WKND Dark"
        elif "week ahead spark" in low:
            rename_map[c] = "Week Ahead Spark"
        elif "week ahead dark" in low:
            rename_map[c] = "Week Ahead Dark"
        elif low == "order" or "[order]" in low:
            rename_map[c] = "Order"
    return out.rename(columns=rename_map)


def _normalize_cdc_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename: dict[str, str] = {}
    for c in out.columns:
        low = str(c).strip().lower()
        if "bmu" in low:
            rename[c] = "BMU"
        elif "settlement day" in low or low == "sd" or low.endswith("[sd]"):
            rename[c] = "Settlement Day"
        elif "expected" in low and "generation" in low:
            rename[c] = "Expected Generation"
        elif "fpn" in low:
            rename[c] = "FPN"
        elif "mel_redec" in low or ("mel" in low and "redec" in low):
            rename[c] = "MEL_REDEC"
        elif low.endswith("mel") or "[mel]" in low:
            rename[c] = "MEL"
        elif low.endswith("sel") or "[sel]" in low:
            rename[c] = "SEL"
    out = out.rename(columns=rename)
    return out


def _strip_table_prefix(col: str) -> str:
    s = str(col).strip()
    m = re.search(r"\[([^\]]+)\]$", s)
    return m.group(1) if m else s


def _prepare_forecast_series(df: pd.DataFrame, value_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Date", value_label])

    out = df.copy()
    out.columns = [_strip_table_prefix(c) for c in out.columns]

    def find_col(candidates: list[str]) -> str | None:
        for cand in candidates:
            key = "".join(ch for ch in cand.lower() if ch.isalnum())
            for c in out.columns:
                ckey = "".join(ch for ch in str(c).lower() if ch.isalnum())
                if key == ckey or key in ckey:
                    return c
        return None

    date_col = find_col(["FORECASTDATE", "DELIVERYDATE", "SETTLEMENT DAY", "DATE"])
    time_col = find_col(["TIME"])
    value_col = find_col([value_label, f"{value_label}_value", "VALUE", "WIND_MED", "WIND", "SOLAR"])

    if date_col is None:
        return pd.DataFrame(columns=["Date", value_label])
    if value_col is None:
        value_col = out.columns[-1]

    dt = pd.to_datetime(out[date_col], errors="coerce")
    if time_col is not None:
        t = pd.to_datetime(out[time_col], errors="coerce")
        t_delta = (t - t.dt.normalize()).fillna(pd.Timedelta(0))
        dt = dt.dt.normalize() + t_delta

    series = pd.DataFrame(
        {
            "Date": dt,
            value_label: pd.to_numeric(out[value_col], errors="coerce"),
        }
    ).dropna(subset=["Date", value_label])

    # Match BI tooltip behavior for duplicate timestamp buckets.
    return series.groupby("Date", as_index=False)[value_label].max().sort_values("Date")


def _extract_latest_series(
    table_name: str,
    date_keys: list[str],
    value_keys: list[str],
    max_rows: int = 50000,
) -> pd.DataFrame:
    raw = run_dax(f"EVALUATE TOPN({max_rows}, '{table_name}')")
    if raw.empty:
        return pd.DataFrame(columns=["Date", "Value"])

    df = raw.copy()
    df.columns = [_strip_table_prefix(c) for c in df.columns]

    def pick_col(keys: list[str]) -> str | None:
        for c in df.columns:
            low = c.lower().replace(" ", "").replace("_", "")
            if all(k in low for k in keys):
                return c
        return None

    date_col = None
    for k in date_keys:
        date_col = pick_col([k])
        if date_col:
            break
    value_col = None
    for k in value_keys:
        value_col = pick_col([k])
        if value_col:
            break
    if date_col is None or value_col is None:
        return pd.DataFrame(columns=["Date", "Value"])

    # Try to detect a "forecast run / issue" column and keep latest run only.
    run_col = None
    run_signals = ["issue", "run", "created", "publish", "asof", "validfrom", "updated"]
    for c in df.columns:
        low = c.lower()
        if any(sig in low for sig in run_signals):
            if c not in {date_col, value_col}:
                run_col = c
                break

    out = df[[date_col, value_col] + ([run_col] if run_col else [])].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col, value_col])
    if out.empty:
        return pd.DataFrame(columns=["Date", "Value"])

    if run_col:
        run_as_dt = pd.to_datetime(out[run_col], errors="coerce")
        if run_as_dt.notna().any():
            out["_run"] = run_as_dt
        else:
            out["_run"] = pd.to_numeric(out[run_col], errors="coerce")
            if out["_run"].isna().all():
                out["_run"] = out[run_col].astype(str)
        latest = out["_run"].max()
        out = out[out["_run"] == latest]

    out = out.groupby(date_col, as_index=False)[value_col].sum()
    out = out.rename(columns={date_col: "Date", value_col: "Value"}).sort_values("Date")
    return out


def _infer_temp_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Date", "Temperature"])

    cols = list(df.columns)
    lower = {c: str(c).strip().lower() for c in cols}

    # Prefer forecast-style date columns over generic day/date labels.
    date_priority = [
        "forecastdate",
        "deliverydate",
        "settlementday",
        "settlement_day",
        "date",
        "day",
    ]
    date_col = None
    for key in date_priority:
        for c in cols:
            compact = "".join(ch for ch in lower[c] if ch.isalnum() or ch == "_")
            if key in compact:
                date_col = c
                break
        if date_col is not None:
            break
    if date_col is None:
        date_col = cols[0]

    temp_col = None
    temp_priority = ["temperature", "temp", "value", "normal"]
    for key in temp_priority:
        for c in cols:
            l = lower[c]
            if key in l:
                temp_col = c
                break
        if temp_col is not None:
            break
    if temp_col is None:
        # choose first numeric-like column that is not the date
        for c in cols:
            if c == date_col:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                temp_col = c
                break
    if temp_col is None:
        return pd.DataFrame(columns=["Date", "Temperature"])

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["Temperature"] = pd.to_numeric(df[temp_col], errors="coerce")

    topo_col = next((c for c in cols if "topo" in lower[c]), None)
    valid_col = next((c for c in cols if lower[c] == "valid" or "valid" in lower[c]), None)
    if topo_col is not None:
        out["Topo"] = df[topo_col]
    if valid_col is not None:
        out["Valid"] = df[valid_col]
    return out


UNIT_STATUS_DAX_CANDIDATES = [
    r"""EVALUATE TOPN(5000, 'Query1')""",
    r"""
EVALUATE
SELECTCOLUMNS(
    'Uniper Availability',
    "UnitName", 'Uniper Availability'[UnitName],
    "WD Status", 'Uniper Availability'[WD Status],
    "DA Schedule", 'Uniper Availability'[DA Schedule],
    "DA + 1 Schedule", 'Uniper Availability'[DA + 1 Schedule],
    "DA + 2 Schedule", 'Uniper Availability'[DA + 2 Schedule],
    "Select Time", 'Uniper Availability'[Select Time]
)
""",
    r"""
EVALUATE
SELECTCOLUMNS(
    'REMIT Availability',
    "UnitName", 'REMIT Availability'[bmu],
    "WD Status", 'REMIT Availability'[WD Status],
    "DA Schedule", 'REMIT Availability'[DA Schedule],
    "DA + 1 Schedule", 'REMIT Availability'[DA + 1 Schedule],
    "DA + 2 Schedule", 'REMIT Availability'[DA + 2 Schedule],
    "Select Time", 'REMIT Availability'[Select Time]
)
""",
]


def _pick_column(cols: list[str], keywords: list[str]) -> str | None:
    for c in cols:
        low = c.strip().lower()
        if all(k in low for k in keywords):
            return c
    return None


def _normalize_unit_status_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = list(out.columns)
    if not cols:
        return out

    unit_col = _pick_column(cols, ["unit"]) or _pick_column(cols, ["bmu"]) or cols[0]
    wd_col = _pick_column(cols, ["wd", "status"])
    da_col = _pick_column(cols, ["da", "schedule"])
    da1_col = _pick_column(cols, ["da", "+", "1"]) or _pick_column(cols, ["da", "1", "schedule"])
    da2_col = _pick_column(cols, ["da", "+", "2"]) or _pick_column(cols, ["da", "2", "schedule"])
    time_col = _pick_column(cols, ["time"]) or _pick_column(cols, ["select", "time"])

    rename_map = {unit_col: "UnitName"}
    if wd_col:
        rename_map[wd_col] = "WD Status"
    if da_col:
        rename_map[da_col] = "DA Schedule"
    if da1_col:
        rename_map[da1_col] = "DA + 1 Schedule"
    if da2_col:
        rename_map[da2_col] = "DA + 2 Schedule"
    if time_col:
        rename_map[time_col] = "Select Time"

    out = out.rename(columns=rename_map)
    preferred = [c for c in ["UnitName", "WD Status", "DA Schedule", "DA + 1 Schedule", "DA + 2 Schedule", "Select Time"] if c in out.columns]
    other = [c for c in out.columns if c not in preferred]
    return out[preferred + other]


def _normalize_from_query1(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = list(df.columns)
    canon = {}
    for c in cols:
        key = "".join(ch for ch in str(c).lower() if ch.isalnum())
        canon[key] = c

    def find_col(*keys: str) -> str | None:
        for c in cols:
            low = "".join(ch for ch in str(c).lower() if ch.isalnum())
            if all(k in low for k in keys):
                return c
        return None

    unit_col = canon.get("unitname") or find_col("unit", "name")
    wd_col = find_col("wd", "status")
    da_col = find_col("da", "schedule")
    da1_col = find_col("da", "1", "schedule")
    da2_col = find_col("da", "2", "schedule")
    time_col = find_col("time")

    # Fallbacks for generic Power BI auto column names.
    def _colnum(name: str) -> int | None:
        # Handles names like "Column10", "Query1[Column10]", "Query1[Column 10]".
        m = re.search(r"\[?\s*column\s*(\d+)\s*\]?$", str(name), flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    generic_cols: list[tuple[int, str]] = []
    for c in cols:
        n = _colnum(str(c))
        if n is not None and 5 <= n <= 30:
            generic_cols.append((n, c))
    generic_cols = sorted(generic_cols, key=lambda x: x[0])
    generic_only = [c for _, c in generic_cols]
    by_num = {n: col_name for n, col_name in generic_cols}

    # Prefer exact Query1[ColumnN]-style matches when present.
    wd_col = wd_col or by_num.get(10) or canon.get("column10")
    da_col = da_col or by_num.get(11) or canon.get("column11")
    da1_col = da1_col or by_num.get(12) or canon.get("column12")
    da2_col = da2_col or by_num.get(13) or canon.get("column13")
    time_col = time_col or by_num.get(9) or canon.get("column9")

    # If direct mapping failed, infer a 4-column status block from the most populated generic columns.
    if not (wd_col and da_col and da1_col and da2_col) and len(generic_only) >= 4:
        work = df.copy()
        if unit_col in work.columns:
            work = work[work[unit_col].astype(str).str.strip().ne("")]

        def _filled_count(col_name: str) -> int:
            s = work[col_name].astype(str).str.strip().str.lower()
            return int((~s.isin(["", "nan", "none", "null", "false"])).sum())

        counts = {col_name: _filled_count(col_name) for _, col_name in generic_cols}
        best_block: list[str] = []
        best_score = -1
        nums = [n for n, _ in generic_cols]
        for n in nums:
            block_nums = [n, n + 1, n + 2, n + 3]
            if all(bn in by_num for bn in block_nums):
                cols_block = [by_num[bn] for bn in block_nums]
                score = sum(counts[c] for c in cols_block)
                if score > best_score:
                    best_score = score
                    best_block = cols_block

        if not best_block:
            ranked = sorted(generic_only, key=lambda c: counts[c], reverse=True)
            best_block = ranked[:4]

        if len(best_block) >= 4:
            wd_col = wd_col or best_block[0]
            da_col = da_col or best_block[1]
            da1_col = da1_col or best_block[2]
            da2_col = da2_col or best_block[3]

    # Last-resort fallback: choose best populated text-like columns.
    if not (wd_col and da_col and da1_col and da2_col):
        work = df.copy()
        if unit_col in work.columns:
            work = work[work[unit_col].astype(str).str.strip().ne("")]
        candidate_cols = [c for c in cols if c != unit_col]
        scored: list[tuple[int, int, str]] = []
        for c in candidate_cols:
            s = work[c].astype(str).str.strip()
            non_empty = s[~s.str.lower().isin(["", "nan", "none", "null", "false"])]
            filled = int(non_empty.shape[0])
            if filled == 0:
                continue
            distinct = int(non_empty.nunique())
            # prefer columns with real content and moderate cardinality (status-like fields)
            if 1 <= distinct <= max(60, filled):
                scored.append((filled, -distinct, c))
        scored.sort(reverse=True)
        picked = [c for _, _, c in scored[:4]]
        if len(picked) >= 4:
            wd_col = wd_col or picked[0]
            da_col = da_col or picked[1]
            da1_col = da1_col or picked[2]
            da2_col = da2_col or picked[3]

    if unit_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["UnitName"] = df[unit_col].astype(str)
    if wd_col and wd_col in df.columns:
        out["WD Status"] = df[wd_col]
    if da_col and da_col in df.columns:
        out["DA Schedule"] = df[da_col]
    if da1_col and da1_col in df.columns:
        out["DA + 1 Schedule"] = df[da1_col]
    if da2_col and da2_col in df.columns:
        out["DA + 2 Schedule"] = df[da2_col]
    if time_col and time_col in df.columns:
        out["Select Time"] = df[time_col]
    out = out[out["UnitName"].str.strip().ne("")]
    status_cols = [c for c in ["WD Status", "DA Schedule", "DA + 1 Schedule", "DA + 2 Schedule"] if c in out.columns]
    if status_cols:
        out = out[out[status_cols].notna().any(axis=1)]
    return out


def export_all_to_excel_sheets(
    datasets: dict[str, pd.DataFrame],
    base_name: str = "Strategy_Dashboard",
    bundle_dir: Path | None = None,
) -> dict[str, object]:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    latest_path = EXPORT_DIR / f"{base_name}_LATEST.xlsx"
    stamped_root = bundle_dir if bundle_dir is not None else EXPORT_DIR
    stamped_path = stamped_root / f"{base_name}_{ts}.xlsx"

    def clean_sheet_name(name: str) -> str:
        bad = r'[]:*?/\\'
        out = "".join("_" if c in bad else c for c in str(name))
        return out[:31]

    summary_rows = []
    exported_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for name, df in datasets.items():
        if df is None:
            continue
        summary_rows.append(
            {
                "Sheet": name,
                "Rows": int(len(df)),
                "Columns": int(len(df.columns)),
                "Exported": exported_at,
            }
        )

    df_summary = pd.DataFrame(summary_rows)

    engine = None
    if importlib.util.find_spec("openpyxl"):
        engine = "openpyxl"
    elif importlib.util.find_spec("xlsxwriter"):
        engine = "xlsxwriter"
    else:
        raise RuntimeError("Excel export requires openpyxl or xlsxwriter. Install with: pip install openpyxl")

    written_paths: list[Path] = []
    write_errors: dict[str, str] = {}
    for path in (latest_path, stamped_path):
        try:
            with pd.ExcelWriter(path, engine=engine) as writer:
                df_summary.to_excel(writer, sheet_name="Summary", index=False)
                for name, df in datasets.items():
                    if df is None or df.empty:
                        continue
                    df.to_excel(writer, sheet_name=clean_sheet_name(name), index=False)
            written_paths.append(path)
        except PermissionError as exc:
            write_errors[str(path)] = str(exc)

    if not written_paths:
        raise RuntimeError(
            "Excel export failed. Close open workbook(s) in Excel/Power BI and try again."
        )

    primary_path = latest_path if latest_path in written_paths else written_paths[0]
    return {
        "primary_path": primary_path,
        "latest_path": latest_path,
        "stamped_path": stamped_path,
        "latest_written": latest_path in written_paths,
        "written_paths": written_paths,
        "write_errors": write_errors,
    }


def build_export_datasets(refresh_nonce: int) -> dict[str, pd.DataFrame]:
    wind_df, solar_df = load_forecasts(refresh_nonce)
    prices_df = _normalize_prices_columns(load_prices_table_main(refresh_nonce))
    niv_df = load_niv_forecast(refresh_nonce)
    ic_df = load_ic_flows(refresh_nonce)
    temp_df = load_temperature(refresh_nonce)
    avail_df = load_availability_matrix(refresh_nonce)

    wind = wind_df.copy()
    solar = solar_df.copy()
    if not wind.empty and wind.shape[1] >= 2:
        wind = wind.rename(columns={wind.columns[0]: "Date", wind.columns[1]: "Wind"})
    if not solar.empty and solar.shape[1] >= 2:
        solar = solar.rename(columns={solar.columns[0]: "Date", solar.columns[1]: "Solar"})
    if not wind.empty:
        wind["Date"] = pd.to_datetime(wind["Date"], errors="coerce")
        wind["Wind"] = pd.to_numeric(wind["Wind"], errors="coerce")
        wind = wind.dropna(subset=["Date"]).groupby("Date", as_index=False)["Wind"].sum()
    if not solar.empty:
        solar["Date"] = pd.to_datetime(solar["Date"], errors="coerce")
        solar["Solar"] = pd.to_numeric(solar["Solar"], errors="coerce")
        solar = solar.dropna(subset=["Date"]).groupby("Date", as_index=False)["Solar"].sum()
    if not wind.empty or not solar.empty:
        wind_solar = pd.merge(wind, solar, on="Date", how="outer").sort_values("Date")
    else:
        wind_solar = pd.DataFrame(columns=["Date", "Wind", "Solar"])

    return {
        "Wind_Solar": wind_solar,
        "Prices": prices_df,
        "NIV": niv_df,
        "IC_Flows": ic_df,
        "Temperature": temp_df,
        "Availability": avail_df,
    }


def _safe_file_stem(name: str) -> str:
    allowed = []
    for ch in str(name):
        if ch.isalnum() or ch in {"_", "-"}:
            allowed.append(ch)
        elif ch in {" ", "/", "\\"}:
            allowed.append("_")
    out = "".join(allowed).strip("_")
    return out or "report"


def build_export_figures(refresh_nonce: int, datasets: dict[str, pd.DataFrame]) -> dict[str, go.Figure]:
    figures: dict[str, go.Figure] = {}

    ws = datasets.get("Wind_Solar", pd.DataFrame()).copy()
    if not ws.empty and {"Date", "Wind", "Solar"}.issubset(ws.columns):
        ws["Date"] = pd.to_datetime(ws["Date"], errors="coerce")
        ws["Wind"] = pd.to_numeric(ws["Wind"], errors="coerce").fillna(0)
        ws["Solar"] = pd.to_numeric(ws["Solar"], errors="coerce").fillna(0)
        ws = ws.dropna(subset=["Date"]).sort_values("Date")
        if not ws.empty:
            fig = go.Figure()
            fig.add_bar(x=ws["Date"], y=ws["Wind"], name="Wind", marker_color="#2ca02c")
            fig.add_bar(x=ws["Date"], y=ws["Solar"], name="Solar", marker_color="#ff7f0e")
            fig.update_layout(
                barmode="stack",
                template="plotly_white",
                title="Wind & Solar Forecast",
                xaxis_title="Date",
                yaxis_title="Forecast",
                legend=dict(orientation="h", y=1.03),
            )
            figures["Overview_Wind_Solar"] = fig

    margins_df = load_margins(refresh_nonce).copy()
    if not margins_df.empty and len(margins_df.columns) >= 3:
        margins_df = margins_df.rename(
            columns={margins_df.columns[0]: "Date", margins_df.columns[1]: "Demand", margins_df.columns[2]: "MarginForecast"}
        )
        margins_df["Date"] = pd.to_datetime(margins_df["Date"], errors="coerce")
        margins_df["Demand"] = pd.to_numeric(margins_df["Demand"], errors="coerce")
        margins_df["MarginForecast"] = pd.to_numeric(margins_df["MarginForecast"], errors="coerce")
        margins_df = margins_df.dropna(subset=["Date"])
        margins_df = margins_df[margins_df[["Demand", "MarginForecast"]].notna().any(axis=1)]
        if not margins_df.empty:
            plot_df = margins_df.sort_values("Date")
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=plot_df["Date"],
                    y=plot_df["Demand"],
                    name="Demand",
                    marker_color="#2e86ff",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=plot_df["Date"],
                    y=plot_df["MarginForecast"],
                    name="MarginForecast",
                    marker_color="#f97316",
                )
            )
            fig.update_layout(
                template="plotly_white",
                barmode="group",
                title="Peak Demand & Margin",
                yaxis_title="MW",
                legend=dict(orientation="h", y=1.03, title_text=""),
            )
            figures["Overview_Peak_Demand_Margin"] = fig

    niv_df = datasets.get("NIV", pd.DataFrame()).copy()
    if not niv_df.empty:
        if "SP" not in niv_df.columns and len(niv_df.columns) > 0:
            niv_df = niv_df.rename(columns={niv_df.columns[0]: "SP"})
        rename_map = {}
        for c in niv_df.columns:
            low = str(c).strip().lower()
            if "ngt" in low and "forecast" in low and "error" in low:
                rename_map[c] = "NGT_Forecast_Error"
            elif "outturn" in low and "niv" in low:
                rename_map[c] = "Outturn_NIV"
            elif "uniper" in low and "niv" in low:
                rename_map[c] = "Uniper_NIV_Forecast"
            elif "ngc" in low and "forecast" in low:
                rename_map[c] = "NGC_NIV_Forecast"
        if rename_map:
            niv_df = niv_df.rename(columns=rename_map)
        required = {"SP", "NGT_Forecast_Error", "Outturn_NIV", "Uniper_NIV_Forecast", "NGC_NIV_Forecast"}
        if required.issubset(niv_df.columns):
            fig = go.Figure()
            fig.add_bar(x=niv_df["SP"], y=niv_df["NGT_Forecast_Error"], name="NGT forecast error", marker_color="#E3C84C")
            fig.add_scatter(x=niv_df["SP"], y=niv_df["Outturn_NIV"], name="Outturn NIV", mode="lines", yaxis="y2")
            fig.add_scatter(x=niv_df["SP"], y=niv_df["Uniper_NIV_Forecast"], name="Uniper NIV Forecast", mode="lines", yaxis="y2")
            fig.add_scatter(x=niv_df["SP"], y=niv_df["NGC_NIV_Forecast"], name="NGC NIV Forecast", mode="lines", yaxis="y2")
            fig.update_layout(
                template="plotly_white",
                title="NIV Forecast",
                xaxis_title="SP",
                yaxis=dict(title="NGT Forecast Error (MW)"),
                yaxis2=dict(title="NIV Forecast (MW)", overlaying="y", side="right"),
                legend=dict(orientation="h", y=1.03),
            )
            figures["Overview_NIV_Forecast"] = fig

    ic_df = datasets.get("IC_Flows", pd.DataFrame()).copy()
    if not ic_df.empty:
        if "Settlement Day" not in ic_df.columns and len(ic_df.columns) > 0:
            ic_df = ic_df.rename(columns={ic_df.columns[0]: "Settlement Day"})
        ic_df["Settlement Day"] = pd.to_datetime(ic_df["Settlement Day"], errors="coerce")
        ic_df = ic_df.dropna(subset=["Settlement Day"]).sort_values("Settlement Day")
        series = [c for c in ic_df.columns if c != "Settlement Day"]
        if not ic_df.empty and series:
            fig = go.Figure()
            for col in series:
                fig.add_trace(
                    go.Scatter(
                        x=ic_df["Settlement Day"],
                        y=pd.to_numeric(ic_df[col], errors="coerce"),
                        mode="lines",
                        stackgroup="flows",
                        name=col,
                    )
                )
            fig.update_layout(template="plotly_white", title="IC Flows", xaxis_title="Settlement Day", yaxis_title="MW")
            figures["Overview_IC_Flows"] = fig

    temp_df = datasets.get("Temperature", pd.DataFrame()).copy()
    if not temp_df.empty and {"Date", "Temperature"}.issubset(temp_df.columns):
        temp_df["Date"] = pd.to_datetime(temp_df["Date"], errors="coerce")
        temp_df["Temperature"] = pd.to_numeric(temp_df["Temperature"], errors="coerce")
        temp_df = temp_df.dropna(subset=["Date", "Temperature"]).sort_values("Date")
        if not temp_df.empty:
            fig = px.line(temp_df, x="Date", y="Temperature", title="Temperature Forecast")
            fig.update_layout(template="plotly_white")
            figures["Overview_Temperature"] = fig

    curve_df = load_price_curve(refresh_nonce).copy()
    if not curve_df.empty:
        if "Settlement Day" not in curve_df.columns and len(curve_df.columns) > 0:
            curve_df = curve_df.rename(columns={curve_df.columns[0]: "Settlement Day"})
        curve_df["Settlement Day"] = pd.to_datetime(curve_df["Settlement Day"], errors="coerce")
        curve_df = curve_df.dropna(subset=["Settlement Day"]).sort_values("Settlement Day")
        y_cols = [c for c in ["MIP", "DA Price", "System Price"] if c in curve_df.columns]
        if not curve_df.empty and y_cols:
            fig = go.Figure()
            for col in y_cols:
                fig.add_trace(go.Scatter(x=curve_df["Settlement Day"], y=curve_df[col], mode="lines", name=col))
            fig.update_layout(template="plotly_white", title="Price Curve", xaxis_title="Settlement Day", yaxis_title="Price")
            figures["Prices_Curve"] = fig

    # CDC/KILLPG, CNQPS, GRAI/EECL small multiples
    cdc_df = load_cdc_killpg(refresh_nonce).copy()
    if not cdc_df.empty:
        cdc_df = _normalize_cdc_columns(cdc_df)
        if "Settlement Day" in cdc_df.columns:
            cdc_df["Settlement Day"] = pd.to_datetime(cdc_df["Settlement Day"], errors="coerce")
        for c in ["FPN", "Expected Generation", "MEL", "MEL_REDEC", "SEL"]:
            if c in cdc_df.columns:
                cdc_df[c] = _coerce_numeric_series(cdc_df[c])
        cdc_df = cdc_df.dropna(subset=[c for c in ["BMU", "Settlement Day"] if c in cdc_df.columns])

        if not cdc_df.empty and "BMU" in cdc_df.columns:
            metrics = [m for m in ["FPN", "Expected Generation", "MEL", "MEL_REDEC", "SEL"] if m in cdc_df.columns]
            color_map = {
                "FPN": "#2e86ff",
                "Expected Generation": "#1d2fa5",
                "MEL": "#f97316",
                "MEL_REDEC": "#7e22ce",
                "SEL": "#ec4899",
            }
            all_bmus = sorted(cdc_df["BMU"].astype(str).unique().tolist())

            def _pick_bmus(priority: list[str], fallback_contains: list[str], max_n: int = 4) -> list[str]:
                picked: list[str] = []
                for p in priority:
                    if p in all_bmus and p not in picked:
                        picked.append(p)
                if not picked:
                    for b in all_bmus:
                        up = b.upper()
                        if any(k in up for k in fallback_contains):
                            picked.append(b)
                        if len(picked) >= max_n:
                            break
                return picked[:max_n]

            def _add_group(group_name: str, bmus: list[str]) -> None:
                for bmu in bmus:
                    sub = cdc_df[cdc_df["BMU"].astype(str) == str(bmu)].sort_values("Settlement Day")
                    if sub.empty:
                        continue
                    fig = go.Figure()
                    for m in metrics:
                        fig.add_trace(
                            go.Scatter(
                                x=sub["Settlement Day"],
                                y=sub[m],
                                mode="lines",
                                name=m,
                                line=dict(color=color_map.get(m, "#334155"), width=2),
                            )
                        )
                    fig.update_layout(
                        template="plotly_white",
                        title=f"{group_name} - {bmu}",
                        xaxis_title="Settlement Day",
                        yaxis_title="MW",
                        legend=dict(orientation="h", y=1.04),
                    )
                    figures[f"{group_name}_{bmu}"] = fig

            cdc_killpg_bmus = _pick_bmus(
                ["T_CDCL-1", "T_KILLPG-1", "T_KILLPG-2"],
                ["CDCL", "KILLPG"],
                max_n=3,
            )
            cnqps_bmus = _pick_bmus(
                ["T_CNQPS-1", "T_CNQPS-2", "T_CNQPS-3", "T_CNQPS-4"],
                ["CNQPS"],
                max_n=4,
            )
            grai_eecl_bmus = _pick_bmus(
                ["T_EECL-1", "T_GRAI-6", "T_GRAI-7", "T_GRAI-8"],
                ["GRAI", "EECL"],
                max_n=4,
            )

            _add_group("CDC_KILLPG", cdc_killpg_bmus)
            _add_group("CNQPS", cnqps_bmus)
            _add_group("GRAI_EECL", grai_eecl_bmus)

    return figures


def build_export_tables(refresh_nonce: int, datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    prices_df = datasets.get("Prices", pd.DataFrame()).copy()
    if not prices_df.empty:
        prices_df = _normalize_prices_columns(prices_df)
        for col in ["Sum of WD", "Sum of DA", "Sum of Weekend", "Week Ahead"]:
            if col in prices_df.columns:
                prices_df[col] = _coerce_numeric_series(prices_df[col])
        if "Type" in prices_df.columns:
            prices_df = prices_df[prices_df["Type"].astype(str).str.strip().ne("")]
            type_series = prices_df["Type"].astype(str).str.lower()
            is_power = type_series.str.contains("£/mwh", regex=False)
        else:
            is_power = pd.Series(False, index=prices_df.index)

        display_cols = [c for c in ["Type", "Sum of WD", "Sum of DA", "Sum of Weekend", "Week Ahead"] if c in prices_df.columns]
        if display_cols:
            commodities = prices_df.loc[~is_power, display_cols].copy()
            power = prices_df.loc[is_power, display_cols].copy()
            if not commodities.empty:
                tables["Prices_Commodities"] = commodities
            if not power.empty:
                tables["Prices_Power"] = power

    spark_df = _normalize_prices_columns(load_prices_sparkdark_table(refresh_nonce)).copy()
    spark_cols = [
        c
        for c in [
            "Type",
            "DA Spark",
            "DA Dark",
            "WKND Spark",
            "WKND Dark",
            "Week Ahead Spark",
            "Week Ahead Dark",
        ]
        if c in spark_df.columns
    ]
    if spark_cols:
        spark_df = spark_df[spark_cols].copy()
        for c in spark_cols:
            if c != "Type":
                spark_df[c] = _coerce_numeric_series(spark_df[c])
        if "Type" in spark_df.columns:
            spark_df = spark_df[spark_df["Type"].astype(str).str.contains("£/MWh", case=False, na=False)]
        if not spark_df.empty:
            tables["Prices_Spark_Dark"] = spark_df

    return tables


def export_visual_reports(
    figures: dict[str, go.Figure],
    tables: dict[str, pd.DataFrame] | None = None,
    base_name: str = "Strategy_Dashboard",
    bundle_dir: Path | None = None,
) -> dict[str, object]:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    stamped_dir = bundle_dir if bundle_dir is not None else (EXPORT_DIR / f"{base_name}_{ts}")
    stamped_dir.mkdir(parents=True, exist_ok=True)

    has_kaleido = importlib.util.find_spec("kaleido") is not None
    html_paths: list[Path] = []
    image_paths: list[Path] = []
    pdf_paths: list[Path] = []
    table_html_paths: list[Path] = []
    combined_html_path = stamped_dir / f"{base_name}_All_Reports.html"
    image_errors: dict[str, str] = {}

    for name, fig in figures.items():
        stem = _safe_file_stem(name)
        if has_kaleido:
            try:
                image_path = stamped_dir / f"{stem}.png"
                pdf_path = stamped_dir / f"{stem}.pdf"
                fig.write_image(str(image_path), format="png", width=1600, height=900, scale=2)
                fig.write_image(str(pdf_path), format="pdf", width=1600, height=900, scale=2)
                image_paths.append(image_path)
                pdf_paths.append(pdf_path)
            except Exception as exc:
                image_errors[str(stamped_dir / stem)] = str(exc)

    # Keep only one consolidated HTML output (no per-report HTML files).

    # Single consolidated HTML report with all figures + tables.
    fig_names = list(figures.keys())
    table_names = list((tables or {}).keys())
    overview_figs = [n for n in fig_names if n.startswith("Overview_")]
    price_figs = [n for n in fig_names if n.startswith("Prices_")]
    unit_figs = [n for n in fig_names if n.startswith("CDC_") or n.startswith("CNQPS_") or n.startswith("GRAI_")]
    other_figs = [n for n in fig_names if n not in overview_figs + price_figs + unit_figs]
    ordered_fig_names = overview_figs + price_figs + other_figs + unit_figs
    ordered_table_names = sorted(table_names)
    section_names = ordered_fig_names + ordered_table_names
    with combined_html_path.open("w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{base_name} - All Reports</title>"
            "<style>"
            "body{font-family:Segoe UI,Arial,sans-serif;background:#f4f6fb;color:#0f172a;margin:0;padding:0;}"
            ".wrap{max-width:1400px;margin:0 auto;padding:24px;}"
            ".head{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:18px 20px;}"
            ".head h1{margin:0 0 8px 0;font-size:32px;}"
            ".meta{color:#475569;font-size:14px;margin-bottom:10px;}"
            ".toc a{display:inline-block;margin:4px 8px 4px 0;padding:6px 10px;background:#e2e8f0;"
            "border-radius:999px;color:#0f172a;text-decoration:none;font-size:13px;}"
            ".section{margin-top:18px;background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px;}"
            ".section h2{margin:0 0 10px 0;font-size:22px;}"
            "table{border-collapse:collapse;width:100%;font-size:13px;}"
            "th,td{border:1px solid #dbe3ee;padding:8px;text-align:right;}"
            "th:first-child,td:first-child{text-align:left;}"
            "th{background:#0b1f3b;color:#f8fafc;}"
            "</style></head><body><div class='wrap'>"
        )
        f.write(
            "<div class='head'>"
            f"<h1>{base_name} - All Reports</h1>"
            f"<div class='meta'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>"
            "<div class='toc'>"
        )
        for name in section_names:
            sid = _safe_file_stem(name)
            f.write(f"<a href='#{sid}'>{name.replace('_', ' ')}</a>")
        f.write("</div></div>")

        first_plot = True
        for name in ordered_fig_names:
            fig = figures[name]
            sid = _safe_file_stem(name)
            f.write(f"<div class='section' id='{sid}'><h2>{name.replace('_', ' ')}</h2>")
            f.write(
                fig.to_html(
                    full_html=False,
                    include_plotlyjs="cdn" if first_plot else False,
                    config={"displaylogo": False},
                )
            )
            first_plot = False
            f.write("</div>")

        for name in ordered_table_names:
            df = (tables or {}).get(name)
            if df is None or df.empty:
                continue
            sid = _safe_file_stem(name)
            f.write(f"<div class='section' id='{sid}'><h2>{name.replace('_', ' ')}</h2>")
            f.write(df.to_html(index=False, border=0))
            f.write("</div>")

        f.write("</div></body></html>")

    return {
        "latest_dir": stamped_dir,
        "stamped_dir": stamped_dir,
        "has_kaleido": has_kaleido,
        "combined_html_path": combined_html_path,
        "html_paths": html_paths,
        "table_html_paths": table_html_paths,
        "image_paths": image_paths,
        "pdf_paths": pdf_paths,
        "image_errors": image_errors,
    }


st.set_page_config(page_title="Strategy Report Dashboard", layout="wide")

st.markdown(
    """
    <style>
      @import url("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css");
      :root {
        --bg: #f4f6fb;
        --card: #0b1f3b;
        --card-soft: #132a4a;
        --text: #0f172a;
        --muted: #64748b;
        --accent: #1f6feb;
      }
      .stApp {
        background: linear-gradient(120deg, #eef2ff 0%, #f8fafc 45%, #f1f5f9 100%);
      }
      .block-container {padding-top: 1.2rem;}
      .kpi {
        border-radius: 14px;
        padding: 16px 18px;
        background: var(--card);
        color: white;
        box-shadow: 0 8px 24px rgba(12, 32, 61, 0.18);
      }
      .kpi-label {font-size: 12px; opacity: 0.85; letter-spacing: 0.3px;}
      .kpi-value {font-size: 26px; font-weight: 700; margin-top: 6px;}
      .section-title {font-size: 22px; font-weight: 700; color: var(--text);}
      .header-container {
        padding: 36px 40px 30px 40px;
        background: #ffffff;
        border-radius: 16px;
        color: #0f172a;
        margin-top: 16px;
        margin-bottom: 26px;
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
      }
      .header-title {
        font-size: clamp(42px, 4vw, 58px);
        font-weight: 800;
        line-height: 1.04;
        margin-bottom: 12px;
        letter-spacing: 0.1px;
      }
      .header-sub {
        font-size: 18px;
        line-height: 1.35;
        color: #475569;
      }
      .header-badges {
        margin-top: 16px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }
      .header-badge {
        display: inline-block;
        padding: 7px 12px;
        font-size: 13px;
        font-weight: 600;
        border-radius: 999px;
        background: #f1f5f9;
        color: #0f172a;
        border: 1px solid #dbe3ee;
      }
      @media (max-width: 900px) {
        .header-container { padding: 24px 20px 20px 20px; min-height: 0; }
        .header-title { font-size: 32px; }
        .header-sub { font-size: 14px; }
        .header-badge { font-size: 12px; }
      }
      .section-card {
        background: white;
        border-radius: 16px;
        padding: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
      }
      .muted {color: var(--muted);}
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f3b 0%, #0a1a31 60%, #071221 100%);
        padding-top: 1.2rem;
      }
      [data-testid="stSidebar"] * {
        color: #e2e8f0;
      }
      [data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
      }
      .sidebar-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
      }
      .sidebar-logo {
        width: 46px;
        height: 46px;
        border-radius: 12px;
        background: linear-gradient(145deg, #f59e0b, #fbbf24);
        color: #0b1f3b;
        display: grid;
        place-items: center;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(245, 158, 11, 0.35);
      }
      .sidebar-title {
        font-weight: 700;
        letter-spacing: 0.3px;
        font-size: 1.18rem;
      }
      .nav-caption {font-size: 12px; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.12em;}
      .sidebar-chip {
        font-size: 11px;
        padding: 4px 8px;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        display: inline-block;
        margin-top: 6px;
      }
      .nav-link {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 15px 16px;
        margin: 11px 0;
        border-radius: 14px;
        text-decoration: none;
        color: #e2e8f0;
        background: rgba(255,255,255,0.03);
        transition: background 140ms ease, transform 140ms ease;
      }
      .nav-link span:last-child {
        font-size: 1.02rem;
        font-weight: 600;
      }
      .nav-link:hover {
        background: rgba(255,255,255,0.10);
        transform: translateX(2px);
      }
      .nav-link.active {
        background: rgba(255,255,255,0.16);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.18);
      }
      .nav-icon {
        width: 22px;
        text-align: center;
        font-size: 18px;
        opacity: 0.9;
      }
      /* Radio group polish */
      div[role="radiogroup"] > label {
        border-radius: 12px;
        padding: 8px 10px;
        margin: 4px 0;
        transition: background 140ms ease, transform 140ms ease;
        background: rgba(255,255,255,0.03);
      }
      div[role="radiogroup"] > label:hover {
        background: rgba(255,255,255,0.10);
        transform: translateX(2px);
      }
      div[role="radiogroup"] > label span {
        font-size: 0.96rem;
        font-weight: 500;
      }
      div[role="radiogroup"] { display: none; }
      /* Sidebar button */
      [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(180deg, #f8fafc, #e2e8f0);
        color: #0b1f3b !important;
        -webkit-text-fill-color: #0b1f3b;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        font-weight: 700;
        height: 44px;
        opacity: 1;
      }
      [data-testid="stSidebar"] .stButton {
        margin-top: 8px;
      }
      [data-testid="stSidebar"] .stButton > button:hover {
        background: #e2e8f0;
        color: #071221 !important;
        -webkit-text-fill-color: #071221;
      }
      [data-testid="stSidebar"] .stButton > button:active {
        background: #dbe3ee;
        color: #071221 !important;
      }
      [data-testid="stSidebar"] .stButton > button:disabled {
        background: #cfd8e3;
        color: #334155 !important;
        -webkit-text-fill-color: #334155;
        opacity: 1;
      }
      .sidebar-footer {
        font-size: 12px;
        opacity: 0.7;
        margin-top: 0.9rem;
      }
      .sidebar-divider {
        height: 1px;
        background: rgba(255,255,255,0.08);
        margin: 18px 0 14px 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class='sidebar-header'>
          <div class='sidebar-logo'>S</div>
          <div class='sidebar-title'>Strategy Dashboard</div>
        </div>
        <div class='sidebar-chip'>Live • Internal</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='nav-caption'>Navigation</div>", unsafe_allow_html=True)
    nav_items = [
        ("Overview", "fa-solid fa-chart-column"),
        ("Availability/DA Schedule", "fa-regular fa-calendar-days"),
        ("Prices", "fa-solid fa-sterling-sign"),
        ("CDC/KILLPG", "fa-regular fa-file-lines"),
        ("CNQPS", "fa-regular fa-compass"),
        ("GRAI/EECL", "fa-solid fa-bolt"),
    ]
    qp = st.query_params
    page = qp.get("page", "Overview")
    valid_pages = [p for p, _ in nav_items]
    if page not in valid_pages:
        page = "Overview"
    st.session_state["page"] = page
    for label, icon in nav_items:
        active = "active" if label == page else ""
        st.markdown(
            f"""
<a class='nav-link {active}' href='?page={label}'>
  <span class='nav-icon'><i class='{icon}'></i></span>
  <span>{label}</span>
</a>
""",
            unsafe_allow_html=True,
        )
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    refresh = st.button("Refresh data", use_container_width=True, type="primary")
    export_now = st.button("Export to Excel (Power BI)", use_container_width=True)
    st.markdown("<div class='sidebar-footer'>v1.0 • Internal</div>", unsafe_allow_html=True)

if "refresh_nonce" not in st.session_state:
    st.session_state["refresh_nonce"] = 0


@st.cache_data(show_spinner="Running DAX query...")
def load_query(dax_query: str, refresh_nonce: int) -> pd.DataFrame:
    _ = refresh_nonce
    return run_dax(dax_query)


@st.cache_data(show_spinner="Loading forecast data...")
def load_forecasts(refresh_nonce: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ = refresh_nonce
    wind_df = get_wind_forecast()
    solar_df = get_solar_forecast()
    return wind_df, solar_df


@st.cache_data(show_spinner="Loading margins data...")
def load_margins(refresh_nonce: int) -> pd.DataFrame:
    _ = refresh_nonce
    dax = """
    EVALUATE
    SUMMARIZECOLUMNS(
        'Margins'[Date],
        "Demand", SUM('Margins'[Demand]),
        "MarginForecast", SUM('Margins'[Margin Forecast])
    )
    ORDER BY 'Margins'[Date]
    """
    return run_dax(dax)


@st.cache_data(show_spinner="Loading NIV forecast data...")
def load_niv_forecast(refresh_nonce: int) -> pd.DataFrame:
    _ = refresh_nonce
    return get_niv_forecast()


@st.cache_data(show_spinner="Loading IC flows data...")
def load_ic_flows(refresh_nonce: int) -> pd.DataFrame:
    _ = refresh_nonce
    return get_ic_flows()


@st.cache_data(ttl=60 * 10)
def load_temperature(refresh_nonce: int) -> pd.DataFrame:
    _ = refresh_nonce
    # Prefer forecast tables (ECMWF/GFS) and fallback to climate normals.
    candidate_queries = [
        ("ECMWF", "EVALUATE TOPN(5000, 'ECMWF')"),
        ("GFS", "EVALUATE TOPN(5000, 'GFS')"),
        ("climate_uk", "EVALUATE TOPN(5000, 'climate_uk')"),
    ]

    candidates: list[tuple[str, pd.DataFrame]] = []
    for source_name, dax in candidate_queries:
        try:
            raw = run_dax(dax)
        except Exception:
            continue
        if raw.empty:
            continue
        cand = _infer_temp_frame(raw)
        if cand.empty:
            continue
        cand = cand.dropna(subset=["Date", "Temperature"])
        if cand.empty:
            continue
        if "Topo" in cand.columns:
            sub = cand[cand["Topo"].astype(str).str.lower() == "uk"]
            if not sub.empty:
                cand = sub
        if "Valid" in cand.columns:
            valid = cand["Valid"].astype(str).str.lower()
            sub = cand[valid.isin(["1", "true", "yes", "y"])]
            if not sub.empty:
                cand = sub
        cand = cand.sort_values("Date")
        if not cand.empty:
            candidates.append((source_name, cand))

    if candidates:
        # Prefer true forecast sources over climate normals.
        today = pd.Timestamp.now().normalize()
        forecast_candidates = [(s, d) for s, d in candidates if s in {"ECMWF", "GFS"}]
        if forecast_candidates:
            recent = [(s, d) for s, d in forecast_candidates if d["Date"].max() >= (today - pd.Timedelta(days=30))]
            pick_pool = recent if recent else forecast_candidates
            best_source, best_df = max(pick_pool, key=lambda x: x[1]["Date"].max())
        else:
            best_source, best_df = max(candidates, key=lambda x: x[1]["Date"].max())

        # Avoid plotting a full-year climatology style if we have a forecast feed.
        if best_source in {"ECMWF", "GFS"}:
            best_df = best_df[best_df["Date"] >= (today - pd.Timedelta(days=7))]
        return best_df.sort_values("Date")

    # Last resort: existing client query path.
    df = get_temperature_forecast()
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Temperature"])
    if "Date" not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: "Date"})
    if "Temperature" not in df.columns and len(df.columns) > 1:
        df = df.rename(columns={df.columns[1]: "Temperature"})
    if "Date" not in df.columns or "Temperature" not in df.columns:
        return pd.DataFrame(columns=["Date", "Temperature"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    out = df.dropna(subset=["Date", "Temperature"]).sort_values("Date")
    if out.empty:
        return pd.DataFrame(columns=["Date", "Temperature"])
    return out


@st.cache_data(show_spinner="Loading availability matrix...")
def load_availability_matrix(refresh_nonce: int) -> pd.DataFrame:
    _ = refresh_nonce
    df = get_availability_matrix()
    return df


@st.cache_data(show_spinner="Loading unit status table...")
def load_unit_status_table(refresh_nonce: int) -> pd.DataFrame:
    _ = refresh_nonce
    def _col_by_suffix(columns: list[str], suffix: str) -> str | None:
        target = suffix.lower().strip()
        for c in columns:
            if str(c).lower().strip().endswith(target):
                return c
        return None

    # Explicit Query1 mapping (based on your model field list).
    try:
        q1 = run_dax("EVALUATE TOPN(5000, 'Query1')")
        if not q1.empty:
            cols = list(q1.columns)
            unit_col = _col_by_suffix(cols, "[unit name]") or _col_by_suffix(cols, "unit name")
            wd_col = _col_by_suffix(cols, "[column10]") or _col_by_suffix(cols, "column10")
            da_col = _col_by_suffix(cols, "[column11]") or _col_by_suffix(cols, "column11")
            da1_col = _col_by_suffix(cols, "[column12]") or _col_by_suffix(cols, "column12")
            da2_col = _col_by_suffix(cols, "[column13]") or _col_by_suffix(cols, "column13")
            time_col = _col_by_suffix(cols, "[column9]") or _col_by_suffix(cols, "column9")

            if unit_col:
                out = pd.DataFrame()
                out["UnitName"] = q1[unit_col].astype(str)
                if wd_col:
                    out["WD Status"] = q1[wd_col]
                if da_col:
                    out["DA Schedule"] = q1[da_col]
                if da1_col:
                    out["DA + 1 Schedule"] = q1[da1_col]
                if da2_col:
                    out["DA + 2 Schedule"] = q1[da2_col]
                if time_col:
                    out["Select Time"] = q1[time_col]
                out = out[out["UnitName"].str.strip().ne("")]
                status_cols = [c for c in ["WD Status", "DA Schedule", "DA + 1 Schedule", "DA + 2 Schedule"] if c in out.columns]
                if status_cols:
                    # Keep rows where at least one status column has meaningful content.
                    keep_mask = pd.Series(False, index=out.index)
                    for c in status_cols:
                        s = out[c].astype(str).str.strip().str.lower()
                        keep_mask = keep_mask | (~s.isin(["", "nan", "none", "null", "false"]))
                    out = out[keep_mask]
                if not out.empty:
                    return out
    except Exception:
        pass

    for dax in UNIT_STATUS_DAX_CANDIDATES:
        try:
            df = run_dax(dax)
        except Exception:
            continue
        if not df.empty:
            if "Query1" in dax:
                out = _normalize_from_query1(df)
                if not out.empty:
                    return out
            return _normalize_unit_status_columns(df)
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading prices data...")
def load_prices_table_main(refresh_nonce: int) -> pd.DataFrame:
    _ = refresh_nonce
    dax = r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Prices'[Type],
    "Order", MAX('Prices'[Order]),
    "Sum of WD", SUM('Prices'[WD]),
    "Sum of DA", SUM('Prices'[DA]),
    "Sum of Weekend", SUM('Prices'[Weekend]),
    "Week Ahead", SUM('Prices'[Week Ahead])
)
ORDER BY [Order]
"""
    return run_dax(dax)


@st.cache_data(show_spinner="Loading price curve...")
def load_price_curve(refresh_nonce: int) -> pd.DataFrame:
    _ = refresh_nonce
    return get_price_curve()


@st.cache_data(show_spinner="Loading CDC/KILLPG data...")
def load_cdc_killpg(refresh_nonce: int) -> pd.DataFrame:
    _ = refresh_nonce
    dax_candidates: list[str] = []
    cdc_tables = [
        "Query2",
        "Query 2",
        "Query3",
        "Query 3",
        "Query4",
        "Query 4",
        "CDC/KILLPG",
        "CDC_KILLPG",
        "REMIT Availability",
        "IC Flows 2",
    ]
    date_cols = ["SETTLEMENT DAY", "SETTLEMENT_DAY", "Settlement Day", "SD", "Date"]
    for table in cdc_tables:
        for dc in date_cols:
            dax_candidates.append(
                f"""
EVALUATE
SUMMARIZECOLUMNS(
    '{table}'[BMU],
    '{table}'[{dc}],
    "FPN", SUM('{table}'[FPN]),
    "Expected Generation", SUM('{table}'[Expected Generation]),
    "MEL", SUM('{table}'[MEL]),
    "MEL_REDEC", SUM('{table}'[MEL_REDEC]),
    "SEL", SUM('{table}'[SEL])
)
ORDER BY '{table}'[BMU], '{table}'[{dc}]
"""
            )
    for dax in dax_candidates:
        try:
            df = run_dax(dax)
        except Exception:
            continue
        if not df.empty:
            return _normalize_cdc_columns(df)

    # Fallback: pull raw rows and infer column names when SUMMARIZECOLUMNS fails.
    def _canon(name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    def _pick(cols: list[str], keys: list[str]) -> str | None:
        ck = [_canon(k) for k in keys]
        for c in cols:
            cc = _canon(c)
            if all(k in cc for k in ck):
                return c
        return None

    for table_name in cdc_tables:
        try:
            raw = run_dax(f"EVALUATE TOPN(200000, '{table_name}')")
        except Exception:
            continue
        if raw.empty:
            continue

        out = raw.copy()
        out = _normalize_cdc_columns(out)
        cols = list(out.columns)

        rename_map: dict[str, str] = {}
        bmu_col = "BMU" if "BMU" in cols else _pick(cols, ["bmu"]) or _pick(cols, ["unit"])
        sd_col = (
            ("Settlement Day" if "Settlement Day" in cols else None)
            or _pick(cols, ["settlement", "day"])
            or _pick(cols, ["sd"])
            or _pick(cols, ["date"])
        )
        fpn_col = "FPN" if "FPN" in cols else _pick(cols, ["fpn"])
        exp_col = "Expected Generation" if "Expected Generation" in cols else _pick(cols, ["expected", "generation"])
        mel_col = "MEL" if "MEL" in cols else _pick(cols, ["mel"])
        redec_col = "MEL_REDEC" if "MEL_REDEC" in cols else _pick(cols, ["mel", "redec"])
        sel_col = "SEL" if "SEL" in cols else _pick(cols, ["sel"])

        if bmu_col and bmu_col != "BMU":
            rename_map[bmu_col] = "BMU"
        if sd_col and sd_col != "Settlement Day":
            rename_map[sd_col] = "Settlement Day"
        if fpn_col and fpn_col != "FPN":
            rename_map[fpn_col] = "FPN"
        if exp_col and exp_col != "Expected Generation":
            rename_map[exp_col] = "Expected Generation"
        if mel_col and mel_col != "MEL":
            rename_map[mel_col] = "MEL"
        if redec_col and redec_col != "MEL_REDEC":
            rename_map[redec_col] = "MEL_REDEC"
        if sel_col and sel_col != "SEL":
            rename_map[sel_col] = "SEL"
        if rename_map:
            out = out.rename(columns=rename_map)

        if "BMU" not in out.columns or "Settlement Day" not in out.columns:
            continue
        metric_cols = [c for c in ["FPN", "Expected Generation", "MEL", "MEL_REDEC", "SEL"] if c in out.columns]
        if not metric_cols:
            continue

        out["Settlement Day"] = pd.to_datetime(out["Settlement Day"], errors="coerce")
        for c in metric_cols:
            out[c] = _coerce_numeric_series(out[c])
        out = out.dropna(subset=["BMU", "Settlement Day"])
        if out.empty:
            continue

        out = out.groupby(["BMU", "Settlement Day"], as_index=False)[metric_cols].sum()
        return out.sort_values(["BMU", "Settlement Day"])

    # Final broad fallback: try other known semantic-model tables and infer CDC-like structure.
    broad_candidates = [
        "Query1",
        "Prices",
        "Margins",
        "System Price",
        "Within Day Prices",
        "DA price",
        "DA Price",
    ]
    for table_name in broad_candidates:
        try:
            raw = run_dax(f"EVALUATE TOPN(50000, '{table_name}')")
        except Exception:
            continue
        if raw.empty:
            continue
        out = _normalize_cdc_columns(raw)
        cols = list(out.columns)
        if "BMU" not in cols:
            continue
        if "Settlement Day" not in cols:
            alt = _pick(cols, ["sd"]) or _pick(cols, ["date"])
            if alt:
                out = out.rename(columns={alt: "Settlement Day"})
        metric_cols = [c for c in ["FPN", "Expected Generation", "MEL", "MEL_REDEC", "SEL"] if c in out.columns]
        if "Settlement Day" not in out.columns or not metric_cols:
            continue
        out["Settlement Day"] = pd.to_datetime(out["Settlement Day"], errors="coerce")
        for c in metric_cols:
            out[c] = _coerce_numeric_series(out[c])
        out = out.dropna(subset=["BMU", "Settlement Day"])
        if out.empty:
            continue
        out = out.groupby(["BMU", "Settlement Day"], as_index=False)[metric_cols].sum()
        return out.sort_values(["BMU", "Settlement Day"])

    return pd.DataFrame()


@st.cache_data(show_spinner="Loading spark/dark prices...")
def load_prices_sparkdark_table(refresh_nonce: int) -> pd.DataFrame:
    _ = refresh_nonce
    dax_candidates = [
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Prices'[Type],
    "Order", MAX('Prices'[Order]),
    "DA Spark", SUM('Prices'[DA Spark]),
    "DA Dark", SUM('Prices'[DA Dark]),
    "WKND Spark", SUM('Prices'[WKND Spark]),
    "WKND Dark", SUM('Prices'[WKND Dark]),
    "Week Ahead Spark", SUM('Prices'[Week Ahead Spark]),
    "Week Ahead Dark", SUM('Prices'[Week Ahead Dark])
)
ORDER BY [Order]
""",
        r"""
EVALUATE
SUMMARIZECOLUMNS(
    'Prices'[Type],
    "Order", MAX('Prices'[Order]),
    "DA Spark", [DA Spark],
    "DA Dark", [DA Dark],
    "WKND Spark", [WKND Spark],
    "WKND Dark", [WKND Dark],
    "Week Ahead Spark", [Week Ahead Spark],
    "Week Ahead Dark", [Week Ahead Dark]
)
ORDER BY [Order]
""",
    ]
    for dax in dax_candidates:
        try:
            df = run_dax(dax)
        except Exception:
            continue
        if not df.empty:
            return df
    return pd.DataFrame()


if export_now:
    try:
        with st.spinner("Exporting workbook and visuals..."):
            bundle_dir = EXPORT_DIR / f"{datetime.now().strftime('%Y-%m-%d')} Report Data Images"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            export_datasets = build_export_datasets(st.session_state.get("refresh_nonce", 0))
            excel_info = export_all_to_excel_sheets(export_datasets, bundle_dir=bundle_dir)
            export_figures = build_export_figures(st.session_state.get("refresh_nonce", 0), export_datasets)
            export_tables = build_export_tables(st.session_state.get("refresh_nonce", 0), export_datasets)
            visuals_info = export_visual_reports(export_figures, tables=export_tables, bundle_dir=bundle_dir)
        st.success(f"Excel saved: {excel_info['primary_path']}")
        if not excel_info["latest_written"]:
            st.warning(
                "LATEST workbook is locked (likely open in Excel/Power BI). "
                f"Saved timestamped file instead: {excel_info['stamped_path']}"
            )
        st.success(f"Visual exports saved in: {visuals_info['latest_dir']}")
        st.info(f"Combined HTML report: {visuals_info['combined_html_path']}")
        if visuals_info.get("table_html_paths"):
            st.info(f"Table HTML exports: {len(visuals_info['table_html_paths'])}")
        st.info(f"Date bundle folder: {bundle_dir}")
        if not visuals_info["has_kaleido"]:
            st.warning(
                "Visual HTML files were exported, but PNG/PDF were skipped because `kaleido` is not installed. "
                "Install with: python -m pip install kaleido"
            )
        elif visuals_info.get("image_errors"):
            st.warning(
                "Visual HTML files were exported. Some PNG/PDF exports failed due to browser renderer issues. "
                "Try restarting Edge/Windows, then re-export."
            )
    except Exception as exc:
        st.error(f"Export failed: {exc}")


if refresh:
    st.session_state["refresh_nonce"] = int(st.session_state.get("refresh_nonce", 0)) + 1
    st.cache_data.clear()
    st.session_state["last_refresh"] = datetime.now()
    st.rerun()

last_refresh = st.session_state.get("last_refresh")
if last_refresh is None:
    last_refresh = datetime.now()
    st.session_state["last_refresh"] = last_refresh
last_refresh_label = last_refresh.strftime("%d %b %Y %H:%M")
st.markdown(
    f"""
<div class="header-container">
  <div class="header-title">Strategy Report Dashboard</div>
  <div class="header-sub">Live from Power BI semantic model. Use Refresh data to re-run the query.</div>
  <div class="header-badges">
    <span class="header-badge">Workspace: UGC UK</span>
    <span class="header-badge">Model: Strategy</span>
    <span class="header-badge">Last refresh: {last_refresh_label}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if page == "Overview":
    dax = """
    EVALUATE
    TOPN(100, 'Query1')
    """

    df = load_query(dax, st.session_state["refresh_nonce"])
    wind_df, solar_df = load_forecasts(st.session_state["refresh_nonce"])
    dax_error_detail = get_last_dax_error()
    if dax_error_detail:
        st.error(f"Power BI connection issue: {dax_error_detail}")

    st.markdown("<div class='section-title'>Wind & Solar Forecast</div>", unsafe_allow_html=True)
    with st.spinner("Loading forecast data..."):
        wind_df, solar_df = load_forecasts(st.session_state["refresh_nonce"])

    if not wind_df.empty and not solar_df.empty:
        wind_df.columns = ["Date", "Wind"]
        solar_df.columns = ["Date", "Solar"]

        wind_df["Date"] = pd.to_datetime(wind_df["Date"], errors="coerce")
        solar_df["Date"] = pd.to_datetime(solar_df["Date"], errors="coerce")
        wind_df["Wind"] = pd.to_numeric(wind_df["Wind"], errors="coerce")
        solar_df["Solar"] = pd.to_numeric(solar_df["Solar"], errors="coerce")

        wind_df = (
            wind_df.dropna(subset=["Date", "Wind"])
            .groupby("Date", as_index=False)["Wind"]
            .sum()
        )
        solar_df = (
            solar_df.dropna(subset=["Date", "Solar"])
            .groupby("Date", as_index=False)["Solar"]
            .sum()
        )

        forecast_df = pd.merge(wind_df, solar_df, on="Date", how="outer").dropna(subset=["Date"])
        forecast_df = forecast_df.sort_values("Date")
        forecast_df["Wind"] = pd.to_numeric(forecast_df.get("Wind"), errors="coerce").fillna(0)
        forecast_df["Solar"] = pd.to_numeric(forecast_df.get("Solar"), errors="coerce").fillna(0)
        bar_width = None
        if len(forecast_df) > 1:
            step = forecast_df["Date"].diff().dropna().median()
            if pd.notna(step):
                bar_width = max(int(step.total_seconds() * 1000 * 0.36), 1)

        fig = go.Figure()
        fig.add_bar(
            x=forecast_df["Date"],
            y=forecast_df["Wind"],
            name="Wind",
            marker_color="#2ca02c",
            opacity=0.95,
            width=bar_width,
            hovertemplate="%{x|%d/%m/%Y %H:%M}<br>Wind %{y:,.2f}<extra></extra>",
        )
        solar_base = forecast_df["Wind"].clip(lower=0)
        fig.add_bar(
            x=forecast_df["Date"],
            y=forecast_df["Solar"],
            base=solar_base,
            customdata=forecast_df[["Solar"]],
            name="Solar",
            marker_color="#ff7f0e",
            opacity=0.95,
            width=bar_width,
            hovertemplate="%{x|%d/%m/%Y %H:%M}<br>Solar %{customdata[0]:,.2f}<extra></extra>",
        )

        fig.update_layout(
            barmode="overlay",
            template="plotly_white",
            height=460,
            bargap=0.22,
            legend=dict(orientation="h", x=0, y=-0.12),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Date",
            yaxis_title="Forecast",
            xaxis=dict(tickformat="%b %d", hoverformat="%d/%m/%Y %H:%M"),
            yaxis=dict(tickformat="~s"),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Wind or Solar data is empty. Check the DAX queries or data availability.")

    st.divider()

    st.markdown("<div class='section-title'>Peak Demand & Margin</div>", unsafe_allow_html=True)
    with st.spinner("Loading margins data..."):
        margins_df = load_margins(st.session_state["refresh_nonce"])

    if not margins_df.empty:
        margins_df.columns = ["Date", "Demand", "MarginForecast"]
        margins_df["Date"] = pd.to_datetime(margins_df["Date"], errors="coerce")
        margins_df["Demand"] = pd.to_numeric(margins_df["Demand"], errors="coerce")
        margins_df["MarginForecast"] = pd.to_numeric(margins_df["MarginForecast"], errors="coerce")
        margins_df = margins_df.dropna(subset=["Date"])
        margins_df = margins_df[margins_df[["Demand", "MarginForecast"]].notna().any(axis=1)]

        today = pd.Timestamp.now().normalize()
        future_df = margins_df[margins_df["Date"] >= (today + pd.Timedelta(days=1))].copy()
        if not future_df.empty:
            next_days = future_df["Date"].dt.normalize().drop_duplicates().sort_values().head(4)
            margins_df = future_df[future_df["Date"].dt.normalize().isin(next_days)].copy()

        margins_df = margins_df.sort_values("Date")
        margins_df["DayLabel"] = margins_df["Date"].dt.strftime("%d %b")

        fig = px.bar(
            margins_df,
            x="DayLabel",
            y=["Demand", "MarginForecast"],
            barmode="group",
            labels={"DayLabel": "Date", "value": "MW"},
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No margins data returned. Check the DAX query or data availability.")

    st.divider()

    st.markdown("<div class='section-title'>NIV Forecast</div>", unsafe_allow_html=True)
    with st.spinner("Loading NIV forecast data..."):
        df_niv = load_niv_forecast(st.session_state["refresh_nonce"])

    if not df_niv.empty:
        # Normalize SP column name for plotting
        if "SP" not in df_niv.columns and len(df_niv.columns) > 0:
            df_niv = df_niv.rename(columns={df_niv.columns[0]: "SP"})
        # Normalize NIV measure column names (in case aliases were not applied)
        rename_map = {}
        for c in df_niv.columns:
            low = str(c).strip().lower()
            if "ngt" in low and "forecast" in low and "error" in low:
                rename_map[c] = "NGT_Forecast_Error"
            elif "outturn" in low and "niv" in low:
                rename_map[c] = "Outturn_NIV"
            elif "uniper" in low and "niv" in low:
                rename_map[c] = "Uniper_NIV_Forecast"
            elif "ngc" in low and "forecast" in low:
                rename_map[c] = "NGC_NIV_Forecast"
        if rename_map:
            df_niv = df_niv.rename(columns=rename_map)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=df_niv["SP"],
                y=df_niv["NGT_Forecast_Error"],
                name="NGT forecast error",
                marker_color="#E3C84C",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_niv["SP"],
                y=df_niv["Outturn_NIV"],
                name="Outturn NIV",
                mode="lines",
                line=dict(color="black", width=3),
                yaxis="y2",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_niv["SP"],
                y=df_niv["Uniper_NIV_Forecast"],
                name="Uniper NIV Forecast",
                mode="lines",
                line=dict(color="#1f77b4", width=3),
                yaxis="y2",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_niv["SP"],
                y=df_niv["NGC_NIV_Forecast"],
                name="NGC NIV Forecast",
                mode="lines",
                line=dict(color="#d62728", width=3),
                yaxis="y2",
            )
        )

        fig.update_layout(
            template="plotly_white",
            barmode="overlay",
            xaxis=dict(title="SP"),
            yaxis=dict(title="NGT Forecast Error (MW)"),
            yaxis2=dict(title="NIV Forecast (MW)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No NIV forecast data returned. Check the DAX query or data availability.")

    st.divider()

    st.markdown("<div class='section-title'>IC Flows</div>", unsafe_allow_html=True)
    with st.spinner("Loading IC flows data..."):
        ic_df = load_ic_flows(st.session_state["refresh_nonce"])

    if not ic_df.empty:
        if "Settlement Day" not in ic_df.columns and len(ic_df.columns) > 0:
            ic_df = ic_df.rename(columns={ic_df.columns[0]: "Settlement Day"})

        ic_df["Settlement Day"] = pd.to_datetime(ic_df["Settlement Day"], errors="coerce")
        ic_df = ic_df.dropna(subset=["Settlement Day"]).sort_values("Settlement Day")

        series = [c for c in ic_df.columns if c != "Settlement Day"]
        fig = go.Figure()
        for col in series:
            fig.add_trace(
                go.Scatter(
                    x=ic_df["Settlement Day"],
                    y=ic_df[col],
                    mode="lines",
                    stackgroup="flows",
                    name=col,
                )
            )

        fig.update_layout(
            template="plotly_white",
            height=480,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Settlement Day",
            yaxis_title="MW",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No IC flows data returned. Check the DAX query or data availability.")

    st.divider()

    st.markdown("<div class='section-title'>Temperature Forecast</div>", unsafe_allow_html=True)
    df_temp = load_temperature(st.session_state["refresh_nonce"])
    if df_temp.empty:
        st.info("No temperature rows returned from Power BI.")
    else:
        fig = px.line(df_temp, x="Date", y="Temperature")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

else:
    if page == "Prices":
        st.title("Prices")
        st.caption("Commodity prices table (Type / WD / DA / Weekend / Week Ahead).")

        with st.spinner("Loading prices data..."):
            prices_df = load_prices_table_main(st.session_state["refresh_nonce"])

        if prices_df.empty:
            st.warning("No prices data returned.")
        else:
            prices_df = _normalize_prices_columns(prices_df)
            for col in ["Sum of WD", "Sum of DA", "Sum of Weekend", "Week Ahead"]:
                if col in prices_df.columns:
                    prices_df[col] = _coerce_numeric_series(prices_df[col])
            if "Type" in prices_df.columns:
                prices_df = prices_df[prices_df["Type"].astype(str).str.strip().ne("")]
            value_cols = [c for c in ["Sum of WD", "Sum of DA", "Sum of Weekend", "Week Ahead"] if c in prices_df.columns]
            if value_cols:
                prices_df = prices_df.dropna(subset=value_cols, how="all")
            prices_df = prices_df.drop(columns=["Order"], errors="ignore")
            display_cols = [c for c in ["Type", "Sum of WD", "Sum of DA", "Sum of Weekend", "Week Ahead"] if c in prices_df.columns]
            if not display_cols or prices_df.empty:
                st.warning("Prices table returned no populated rows for Type/WD/DA/Weekend/Week Ahead.")
            else:
                table = prices_df[display_cols].copy()
                # Split into commodity and power-style rows for readability.
                type_series = table["Type"].astype(str).str.lower()
                is_power = type_series.str.contains("£/mwh", regex=False)
                commodities = table[~is_power].copy()
                power = table[is_power].copy()

                value_cols = [c for c in display_cols if c != "Type"]

                def _styled(df_in: pd.DataFrame):
                    table_styles = [
                        {
                            "selector": "th.col_heading",
                            "props": [
                                ("background-color", "#0b1f3b"),
                                ("color", "#f8fafc"),
                                ("font-weight", "700"),
                                ("font-size", "13px"),
                                ("text-transform", "none"),
                                ("letter-spacing", "0.2px"),
                                ("border-bottom", "1px solid #1e3a5f"),
                            ],
                        },
                        {
                            "selector": "th.row_heading",
                            "props": [("display", "none")],
                        },
                        {
                            "selector": "th.blank",
                            "props": [("display", "none")],
                        },
                    ]
                    sty = (
                        df_in.style
                        .set_table_styles(table_styles, overwrite=False)
                        .format({c: "{:,.2f}" for c in value_cols}, na_rep="")
                        .set_properties(subset=["Type"], **{"font-weight": "600"})
                        .set_properties(subset=value_cols, **{"text-align": "right"})
                    )
                    for c in value_cols:
                        if c in df_in.columns and df_in[c].notna().any():
                            col = pd.to_numeric(df_in[c], errors="coerce")
                            cmin = col.min()
                            cmax = col.max()
                            def _col_bg(v):
                                if pd.isna(v):
                                    return ""
                                if cmax == cmin:
                                    frac = 0.5
                                else:
                                    frac = (float(v) - float(cmin)) / (float(cmax) - float(cmin))
                                # light green ramp without external libs
                                r = 238 - int(40 * frac)
                                g = 250 - int(20 * frac)
                                b = 234 - int(80 * frac)
                                return f"background-color: rgb({r},{g},{b});"
                            sty = sty.applymap(_col_bg, subset=[c])
                    return sty

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("<div class='section-title'>Commodities</div>", unsafe_allow_html=True)
                    if commodities.empty:
                        st.info("No commodity rows.")
                    else:
                        st.dataframe(_styled(commodities), use_container_width=True, height=260, hide_index=True)
                with c2:
                    st.markdown("<div class='section-title'>Power</div>", unsafe_allow_html=True)
                    if power.empty:
                        st.info("No power rows.")
                    else:
                        st.dataframe(_styled(power), use_container_width=True, height=260, hide_index=True)

                st.markdown("<div class='section-title'>Spark / Dark</div>", unsafe_allow_html=True)
                spark_df = _normalize_prices_columns(load_prices_sparkdark_table(st.session_state["refresh_nonce"]))
                spark_cols = [
                    c
                    for c in [
                        "Type",
                        "DA Spark",
                        "DA Dark",
                        "WKND Spark",
                        "WKND Dark",
                        "Week Ahead Spark",
                        "Week Ahead Dark",
                    ]
                    if c in spark_df.columns
                ]
                if not spark_cols:
                    st.info("No spark/dark rows returned.")
                else:
                    spark_df = spark_df[spark_cols + [c for c in ["Order"] if c in spark_df.columns]].copy()
                    for c in spark_cols:
                        if c != "Type":
                            spark_df[c] = _coerce_numeric_series(spark_df[c])
                    if "Type" in spark_df.columns:
                        spark_df = spark_df[spark_df["Type"].astype(str).str.contains("£/MWh", case=False, na=False)]
                    metric_cols = [c for c in spark_cols if c != "Type"]
                    if metric_cols:
                        spark_df = spark_df.dropna(subset=metric_cols, how="all")
                    spark_df = spark_df.drop(columns=["Order"], errors="ignore")

                    def _sparkdark_style(df_in: pd.DataFrame):
                        table_styles = [
                            {
                                "selector": "th.col_heading",
                                "props": [
                                    ("background-color", "#0b1f3b"),
                                    ("color", "#f8fafc"),
                                    ("font-weight", "700"),
                                    ("font-size", "13px"),
                                    ("border-bottom", "1px solid #1e3a5f"),
                                ],
                            },
                            {"selector": "th.row_heading", "props": [("display", "none")]},
                            {"selector": "th.blank", "props": [("display", "none")]},
                        ]
                        sty = (
                            df_in.style
                            .set_table_styles(table_styles, overwrite=False)
                            .format({c: "{:,.2f}" for c in metric_cols}, na_rep="")
                            .set_properties(subset=["Type"], **{"font-weight": "600"})
                            .set_properties(subset=metric_cols, **{"text-align": "right"})
                        )
                        for c in metric_cols:
                            if c in df_in.columns and df_in[c].notna().any():
                                col = pd.to_numeric(df_in[c], errors="coerce")
                                cmin = col.min()
                                cmax = col.max()
                                def _bg(v):
                                    if pd.isna(v):
                                        return ""
                                    if cmax == cmin:
                                        frac = 0.5
                                    else:
                                        frac = (float(v) - float(cmin)) / (float(cmax) - float(cmin))
                                    # red -> green ramp
                                    r = 230 - int(110 * frac)
                                    g = 170 + int(70 * frac)
                                    b = 170 - int(70 * frac)
                                    return f"background-color: rgb({r},{g},{b});"
                                sty = sty.applymap(_bg, subset=[c])
                        return sty

                    if spark_df.empty:
                        st.info("No spark/dark rows returned.")
                    else:
                        st.dataframe(_sparkdark_style(spark_df), use_container_width=True, height=250, hide_index=True)

                st.markdown("<div class='section-title'>Price Curve</div>", unsafe_allow_html=True)
                curve_df = load_price_curve(st.session_state["refresh_nonce"])
                if curve_df.empty:
                    st.info("No price-curve data returned.")
                else:
                    # Normalize date column name from joined DAX output.
                    date_col = None
                    for c in curve_df.columns:
                        low = str(c).strip().lower()
                        if "settlement day" in low:
                            date_col = c
                            break
                    if date_col is None and len(curve_df.columns) > 0:
                        date_col = curve_df.columns[0]

                    curve_df = curve_df.rename(columns={date_col: "Settlement Day"})
                    curve_df["Settlement Day"] = pd.to_datetime(curve_df["Settlement Day"], errors="coerce")
                    for c in ["MIP", "System Price", "DA Price"]:
                        if c in curve_df.columns:
                            curve_df[c] = _coerce_numeric_series(curve_df[c])
                    curve_df = curve_df.dropna(subset=["Settlement Day"]).sort_values("Settlement Day")

                    # Keep a common horizon so all visible price series align on the same time window.
                    present_series = [c for c in ["MIP", "DA Price", "System Price"] if c in curve_df.columns]
                    if present_series:
                        max_by_series = []
                        for c in present_series:
                            sub = curve_df[curve_df[c].notna()]
                            if not sub.empty:
                                max_by_series.append(sub["Settlement Day"].max())
                        if max_by_series:
                            common_end = min(max_by_series)
                            curve_df = curve_df[curve_df["Settlement Day"] <= common_end].copy()

                    fig_curve = go.Figure()
                    if "MIP" in curve_df.columns:
                        fig_curve.add_trace(
                            go.Scatter(
                                x=curve_df["Settlement Day"],
                                y=curve_df["MIP"],
                                mode="lines",
                                name="MIP",
                                line=dict(color="#2e86ff", width=2),
                            )
                        )
                    if "DA Price" in curve_df.columns:
                        fig_curve.add_trace(
                            go.Scatter(
                                x=curve_df["Settlement Day"],
                                y=curve_df["DA Price"],
                                mode="lines",
                                name="DA Price",
                                line=dict(color="#243b99", width=2),
                            )
                        )
                    if "System Price" in curve_df.columns:
                        fig_curve.add_trace(
                            go.Scatter(
                                x=curve_df["Settlement Day"],
                                y=curve_df["System Price"],
                                mode="lines",
                                name="System Price",
                                line=dict(color="#f97316", width=2),
                            )
                        )
                    fig_curve.update_layout(
                        template="plotly_white",
                        height=320,
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h"),
                        xaxis_title="Settlement Day",
                        yaxis_title="Price",
                    )
                    st.plotly_chart(fig_curve, use_container_width=True)
    elif page == "CDC/KILLPG":
        st.title("CDC / KILLPG")
        st.caption("FPN, Expected Generation, MEL, MEL_REDEC, SEL by BMU and settlement day.")

        df_cdc = load_cdc_killpg(st.session_state["refresh_nonce"])
        if df_cdc.empty:
            st.info(
                "Could not load CDC/KILLPG data yet with current schema assumptions. "
                "Share the exact source table + column names and I will wire it exactly."
            )
        else:
            df_cdc = _normalize_cdc_columns(df_cdc)
            if "Settlement Day" in df_cdc.columns:
                df_cdc["Settlement Day"] = pd.to_datetime(df_cdc["Settlement Day"], errors="coerce")
            for c in ["FPN", "Expected Generation", "MEL", "MEL_REDEC", "SEL"]:
                if c in df_cdc.columns:
                    df_cdc[c] = _coerce_numeric_series(df_cdc[c])
            df_cdc = df_cdc.dropna(subset=[c for c in ["BMU", "Settlement Day"] if c in df_cdc.columns])

            bmus = sorted(df_cdc["BMU"].astype(str).unique().tolist()) if "BMU" in df_cdc.columns else []
            default_bmus = [b for b in ["T_CDCL-1", "T_KILLPG-1", "T_KILLPG-2"] if b in bmus] or bmus[:3]
            selected_bmus = st.multiselect("BMU", bmus, default=default_bmus, key="cdc_bmu")

            color_map = {
                "FPN": "#2e86ff",
                "Expected Generation": "#1d2fa5",
                "MEL": "#f97316",
                "MEL_REDEC": "#7e22ce",
                "SEL": "#ec4899",
            }
            metrics = [m for m in ["FPN", "Expected Generation", "MEL", "MEL_REDEC", "SEL"] if m in df_cdc.columns]
            for bmu in selected_bmus:
                sub = df_cdc[df_cdc["BMU"].astype(str) == str(bmu)].sort_values("Settlement Day")
                if sub.empty:
                    continue
                fig = go.Figure()
                for m in metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=sub["Settlement Day"],
                            y=sub[m],
                            mode="lines",
                            name=m,
                            line=dict(color=color_map.get(m, "#334155"), width=2),
                        )
                    )
                fig.update_layout(
                    template="plotly_white",
                    height=260,
                    margin=dict(l=10, r=10, t=30, b=10),
                    title=dict(text=str(bmu), x=0.5, xanchor="center"),
                    legend=dict(orientation="h", y=-0.18, x=0),
                    xaxis_title=None,
                    yaxis_title=None,
                )
                st.plotly_chart(fig, use_container_width=True)
    elif page == "CNQPS":
        st.title("CNQPS")
        st.caption("FPN, Expected Generation, MEL, MEL_REDEC, SEL by BMU and settlement day.")

        df_cdc = load_cdc_killpg(st.session_state["refresh_nonce"])
        if df_cdc.empty:
            st.info("No CNQPS data returned.")
        else:
            df_cdc = _normalize_cdc_columns(df_cdc)
            if "Settlement Day" in df_cdc.columns:
                df_cdc["Settlement Day"] = pd.to_datetime(df_cdc["Settlement Day"], errors="coerce")
            for c in ["FPN", "Expected Generation", "MEL", "MEL_REDEC", "SEL"]:
                if c in df_cdc.columns:
                    df_cdc[c] = _coerce_numeric_series(df_cdc[c])
            df_cdc = df_cdc.dropna(subset=[c for c in ["BMU", "Settlement Day"] if c in df_cdc.columns])

            # Match screenshot: focus CNQPS small-multiples.
            cnqps_defaults = ["T_CNQPS-1", "T_CNQPS-2", "T_CNQPS-3", "T_CNQPS-4"]
            all_bmus = sorted(df_cdc["BMU"].astype(str).unique().tolist())
            default_bmus = [b for b in cnqps_defaults if b in all_bmus]
            selected_bmus = st.multiselect(
                "BMU",
                all_bmus,
                default=default_bmus if default_bmus else [b for b in all_bmus if "CNQPS" in b.upper()],
                key="cnqps_bmu",
            )

            color_map = {
                "FPN": "#2e86ff",
                "Expected Generation": "#1d2fa5",
                "MEL": "#f97316",
                "MEL_REDEC": "#7e22ce",
                "SEL": "#ec4899",
            }
            metrics = [m for m in ["FPN", "Expected Generation", "MEL", "MEL_REDEC", "SEL"] if m in df_cdc.columns]

            col_left, col_right = st.columns(2)
            for i, bmu in enumerate(selected_bmus):
                sub = df_cdc[df_cdc["BMU"].astype(str) == str(bmu)].sort_values("Settlement Day")
                if sub.empty:
                    continue
                fig = go.Figure()
                for m in metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=sub["Settlement Day"],
                            y=sub[m],
                            mode="lines",
                            name=m,
                            line=dict(color=color_map.get(m, "#334155"), width=2),
                        )
                    )
                fig.update_layout(
                    template="plotly_white",
                    height=290,
                    margin=dict(l=10, r=10, t=30, b=10),
                    title=dict(text=str(bmu), x=0.5, xanchor="center"),
                    legend=dict(orientation="h", y=-0.2, x=0),
                    xaxis_title=None,
                    yaxis_title=None,
                )
                (col_left if i % 2 == 0 else col_right).plotly_chart(fig, use_container_width=True)
    elif page == "GRAI/EECL":
        st.title("GRAI / EECL")
        st.caption("FPN, Expected Generation, MEL, MEL_REDEC, SEL by BMU and settlement day.")

        df_cdc = load_cdc_killpg(st.session_state["refresh_nonce"])
        if df_cdc.empty:
            st.info("No GRAI/EECL data returned.")
        else:
            df_cdc = _normalize_cdc_columns(df_cdc)
            if "Settlement Day" in df_cdc.columns:
                df_cdc["Settlement Day"] = pd.to_datetime(df_cdc["Settlement Day"], errors="coerce")
            for c in ["FPN", "Expected Generation", "MEL", "MEL_REDEC", "SEL"]:
                if c in df_cdc.columns:
                    df_cdc[c] = _coerce_numeric_series(df_cdc[c])
            df_cdc = df_cdc.dropna(subset=[c for c in ["BMU", "Settlement Day"] if c in df_cdc.columns])

            grai_defaults = ["T_EECL-1", "T_GRAI-6", "T_GRAI-7", "T_GRAI-8"]
            all_bmus = sorted(df_cdc["BMU"].astype(str).unique().tolist())
            default_bmus = [b for b in grai_defaults if b in all_bmus]
            selected_bmus = st.multiselect(
                "BMU",
                all_bmus,
                default=default_bmus if default_bmus else [b for b in all_bmus if ("GRAI" in b.upper() or "EECL" in b.upper())],
                key="grai_eecl_bmu",
            )

            color_map = {
                "FPN": "#2e86ff",
                "Expected Generation": "#1d2fa5",
                "MEL": "#f97316",
                "MEL_REDEC": "#7e22ce",
                "SEL": "#ec4899",
            }
            metrics = [m for m in ["FPN", "Expected Generation", "MEL", "MEL_REDEC", "SEL"] if m in df_cdc.columns]

            col_left, col_right = st.columns(2)
            for i, bmu in enumerate(selected_bmus):
                sub = df_cdc[df_cdc["BMU"].astype(str) == str(bmu)].sort_values("Settlement Day")
                if sub.empty:
                    continue
                fig = go.Figure()
                for m in metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=sub["Settlement Day"],
                            y=sub[m],
                            mode="lines",
                            name=m,
                            line=dict(color=color_map.get(m, "#334155"), width=2),
                        )
                    )
                fig.update_layout(
                    template="plotly_white",
                    height=290,
                    margin=dict(l=10, r=10, t=30, b=10),
                    title=dict(text=str(bmu), x=0.5, xanchor="center"),
                    legend=dict(orientation="h", y=-0.2, x=0),
                    xaxis_title=None,
                    yaxis_title=None,
                )
                (col_left if i % 2 == 0 else col_right).plotly_chart(fig, use_container_width=True)
    elif page == "Availability/DA Schedule":
        st.title("Availability / DA Schedule")
        st.caption("REMIT Availability matrix (Unit x SD)")

        with st.spinner("Loading availability matrix..."):
            avail = load_availability_matrix(st.session_state["refresh_nonce"])

        if avail.empty:
            st.warning("No availability data returned.")
        else:
            # Normalize column names
            cols = list(avail.columns)
            if len(cols) >= 3:
                avail = avail.rename(columns={cols[0]: "Unit", cols[1]: "SD", cols[2]: "AvailableCapacity"})

            avail["SD"] = pd.to_datetime(avail["SD"], errors="coerce")
            avail["AvailableCapacity"] = pd.to_numeric(avail["AvailableCapacity"], errors="coerce")
            # Source comes as summed half-hourly capacity per day for many units; convert back to MW level.
            avail["AvailableCapacity"] = avail["AvailableCapacity"] / 48.0
            avail = avail.dropna(subset=["Unit", "SD"])
            avail["UnitNorm"] = avail["Unit"].astype(str).map(_norm_unit)

            wanted_norm = [_norm_unit(u) for u in AVAILABILITY_UNITS_FILTER]
            avail_filtered = avail[
                avail["UnitNorm"].isin(wanted_norm)
                | avail["UnitNorm"].apply(lambda x: any(x.endswith(w) for w in wanted_norm))
            ].copy()

            if avail_filtered.empty:
                st.warning("No station-code matches found for the requested list; showing unfiltered data.")
                avail_filtered = avail.copy()
                unit_order = sorted(avail_filtered["Unit"].astype(str).unique())
            else:
                dedup = avail_filtered[["Unit", "UnitNorm"]].drop_duplicates()
                unit_order = []
                seen = set()
                for w in wanted_norm:
                    candidates = dedup[dedup["UnitNorm"].apply(lambda x: x == w or x.endswith(w))]
                    if not candidates.empty:
                        unit_value = str(candidates.iloc[0]["Unit"])
                        if unit_value not in seen:
                            unit_order.append(unit_value)
                            seen.add(unit_value)
                for unit_value in dedup["Unit"].astype(str).tolist():
                    if unit_value not in seen:
                        unit_order.append(unit_value)
                        seen.add(unit_value)
            avail = avail_filtered

            # Pivot to matrix
            matrix = (
                avail.pivot_table(
                    index="Unit",
                    columns="SD",
                    values="AvailableCapacity",
                    aggfunc="sum",
                )
                .reindex(unit_order if unit_order else sorted(avail["Unit"].astype(str).unique()))
            )
            matrix = matrix.drop(columns=["UnitNorm"], errors="ignore")

            def _cell_color(val, vmin, vmax):
                if pd.isna(val):
                    return ""
                if vmax == vmin:
                    frac = 0.0
                else:
                    frac = (val - vmin) / (vmax - vmin)
                # Green ramp: light -> dark
                base = 245 - int(120 * frac)
                green = 255 - int(60 * frac)
                blue = 245 - int(140 * frac)
                return f"background-color: rgb({base},{green},{blue});"

            vmin = float(matrix.min().min()) if not matrix.empty else 0.0
            vmax = float(matrix.max().max()) if not matrix.empty else 1.0
            styled = matrix.style.format(precision=0).applymap(
                lambda v: _cell_color(v, vmin, vmax)
            )
            st.dataframe(styled, use_container_width=True, height=520)

            st.divider()
            st.markdown("<div class='section-title'>Unit Status Table</div>", unsafe_allow_html=True)
            status_df = load_unit_status_table(st.session_state["refresh_nonce"])
            required_cols = UNIT_STATUS_REQUIRED_COLS
            base_status = pd.DataFrame({"UnitName": AVAILABILITY_UNITS_FILTER})
            for c in required_cols[1:]:
                base_status[c] = ""

            merged_status = base_status.copy()
            selected_time = None
            if not status_df.empty:
                src = status_df.copy()

                if "Select Time" in src.columns:
                    src["Select Time"] = src["Select Time"].astype(str).str.strip()
                    time_vals = sorted(
                        [
                            t
                            for t in src["Select Time"].dropna().unique().tolist()
                            if t and t.lower() not in {"nan", "none", "false", "true"}
                        ]
                    )
                    if time_vals:
                        selected_time = st.selectbox("Select Time", time_vals, index=0, key="status_time")
                        src = src[src["Select Time"] == selected_time]

                src["UnitNorm"] = src["UnitName"].astype(str).map(_norm_unit) if "UnitName" in src.columns else ""
                wanted_map = {_norm_unit(u): u for u in AVAILABILITY_UNITS_FILTER}

                def _map_unit_target(norm_val: str) -> str | None:
                    for wn, unit in wanted_map.items():
                        if norm_val == wn or str(norm_val).endswith(wn):
                            return unit
                    return None

                src["UnitTarget"] = src["UnitNorm"].astype(str).map(_map_unit_target)
                src = src[src["UnitTarget"].notna()].copy()

                keep_src = [c for c in required_cols[1:] if c in src.columns]
                if keep_src:
                    src = src[["UnitTarget"] + keep_src].drop_duplicates(subset=["UnitTarget"], keep="first")
                    merged_status = merged_status.merge(
                        src, left_on="UnitName", right_on="UnitTarget", how="left", suffixes=("", "_src")
                    )
                    for c in keep_src:
                        src_col = f"{c}_src"
                        if src_col in merged_status.columns:
                            merged_status[c] = merged_status[src_col].fillna(merged_status[c])
                    merged_status = merged_status.drop(columns=[c for c in merged_status.columns if c.endswith("_src")] + ["UnitTarget"], errors="ignore")

            # Overlay manual entries saved for the day (and selected time, if used).
            save_date = pd.Timestamp.now().strftime("%Y-%m-%d")
            manual_today = _load_manual_status_for_date(save_date, selected_time)
            if not manual_today.empty:
                merged_status = merged_status.merge(
                    manual_today,
                    on="UnitName",
                    how="left",
                    suffixes=("", "_manual"),
                )
                for c in required_cols[1:]:
                    mcol = f"{c}_manual"
                    if mcol in merged_status.columns:
                        merged_status[c] = merged_status[mcol].where(
                            merged_status[mcol].astype(str).str.strip().ne(""),
                            merged_status[c],
                        )
                merged_status = merged_status.drop(columns=[c for c in merged_status.columns if c.endswith("_manual")], errors="ignore")

            for c in required_cols[1:]:
                merged_status[c] = merged_status[c].fillna("").astype(str)

            st.caption("Manual workaround: table below + row editor with dropdowns (prevents popup overlap).")
            st.dataframe(merged_status[required_cols], use_container_width=True, height=420, hide_index=True)

            st.markdown("**Edit One Unit**")
            selected_unit = st.selectbox("Unit", AVAILABILITY_UNITS_FILTER, key="unit_status_unit")
            row = merged_status[merged_status["UnitName"] == selected_unit]
            current = row.iloc[0] if not row.empty else pd.Series({c: "" for c in required_cols})

            def _idx_for(val: str) -> int:
                try:
                    return UNIT_STATUS_OPTIONS.index(str(val))
                except ValueError:
                    return 0

            with st.form("unit_status_form", clear_on_submit=False):
                c1, c2, c3, c4 = st.columns(4)
                wd_val = c1.selectbox("WD Status", UNIT_STATUS_OPTIONS, index=_idx_for(current.get("WD Status", "")), key="wd_status_form")
                da_val = c2.selectbox("DA Schedule", UNIT_STATUS_OPTIONS, index=_idx_for(current.get("DA Schedule", "")), key="da_status_form")
                da1_val = c3.selectbox("DA + 1 Schedule", UNIT_STATUS_OPTIONS, index=_idx_for(current.get("DA + 1 Schedule", "")), key="da1_status_form")
                da2_val = c4.selectbox("DA + 2 Schedule", UNIT_STATUS_OPTIONS, index=_idx_for(current.get("DA + 2 Schedule", "")), key="da2_status_form")
                save_row = st.form_submit_button("Save Unit Status", type="primary")

            if save_row:
                merged_status.loc[merged_status["UnitName"] == selected_unit, "WD Status"] = wd_val
                merged_status.loc[merged_status["UnitName"] == selected_unit, "DA Schedule"] = da_val
                merged_status.loc[merged_status["UnitName"] == selected_unit, "DA + 1 Schedule"] = da1_val
                merged_status.loc[merged_status["UnitName"] == selected_unit, "DA + 2 Schedule"] = da2_val
                _save_manual_status_for_date(save_date, selected_time, merged_status[required_cols])
                st.success(f"Saved {selected_unit} for {save_date}" + (f" at {selected_time}" if selected_time else "") + ".")

            st.caption(f"Manual entries are stored locally in `{UNIT_STATUS_STORE_PATH}`.")

            if selected_time:
                st.caption(f"Showing source rows for selected time: {selected_time}")
    else:
        st.markdown("<div class='section-title'>Page coming soon</div>", unsafe_allow_html=True)
        st.caption("This section is a placeholder for the selected page.")

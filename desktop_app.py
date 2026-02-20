from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime
import threading
import traceback

import pandas as pd

from pbi_client import (
    get_wind_forecast,
    get_solar_forecast,
    get_niv_forecast,
    get_ic_flows,
    get_temperature_forecast,
    get_availability_matrix,
    get_price_curve,
    run_dax,
)


EXPORT_DIR = Path(r"C:\Users\V01013\OneDrive - Uniper SE\Documents\Report")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_prices_table_main() -> pd.DataFrame:
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


def normalize_prices_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map: dict[str, str] = {}
    for c in out.columns:
        low = str(c).strip().lower()
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
        elif low == "order" or "[order]" in low:
            rename_map[c] = "Order"
    return out.rename(columns=rename_map)


def build_datasets() -> dict[str, pd.DataFrame]:
    wind = get_wind_forecast()
    solar = get_solar_forecast()
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

    prices = normalize_prices_columns(load_prices_table_main())
    niv = get_niv_forecast()
    ic = get_ic_flows()
    temp = get_temperature_forecast()
    avail = get_availability_matrix()
    curve = get_price_curve()
    return {
        "Wind_Solar": wind_solar,
        "Prices": prices,
        "NIV": niv,
        "IC_Flows": ic,
        "Temperature": temp,
        "Availability": avail,
        "Price_Curve": curve,
    }


def export_excel(datasets: dict[str, pd.DataFrame], base_name: str = "Strategy_Dashboard") -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = EXPORT_DIR / f"{base_name}_{ts}.xlsx"
    latest_path = EXPORT_DIR / f"{base_name}_LATEST.xlsx"

    def clean_sheet_name(name: str) -> str:
        bad = r'[]:*?/\\'
        return "".join("_" if c in bad else c for c in str(name))[:31]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in datasets.items():
            if df is None:
                continue
            df.to_excel(writer, sheet_name=clean_sheet_name(name), index=False)

    try:
        with pd.ExcelWriter(latest_path, engine="openpyxl") as writer:
            for name, df in datasets.items():
                if df is None:
                    continue
                df.to_excel(writer, sheet_name=clean_sheet_name(name), index=False)
    except Exception:
        pass
    return out_path


def export_html_tables(datasets: dict[str, pd.DataFrame], base_name: str = "Strategy_Dashboard") -> Path:
    bundle = EXPORT_DIR / f"{datetime.now().strftime('%Y-%m-%d')} Report Data Images" / "HTML"
    bundle.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        if df is None:
            continue
        path = bundle / f"{name}.html"
        with path.open("w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>{}</title>".format(name))
            f.write(
                "<style>body{font-family:Segoe UI,Arial,sans-serif;padding:16px;} "
                "table{border-collapse:collapse;width:100%;font-size:12px;} "
                "th,td{border:1px solid #ccc;padding:6px;} th{background:#0b1f3b;color:#fff;}</style>"
            )
            f.write("</head><body>")
            f.write(f"<h2>{name}</h2>")
            f.write(df.to_html(index=False))
            f.write("</body></html>")
    return bundle


class DesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Strategy Dashboard Desktop")
        self.geometry("1200x760")
        self.datasets: dict[str, pd.DataFrame] = {}

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)
        ttk.Button(top, text="Refresh Data", command=self.refresh_data).pack(side="left")
        ttk.Button(top, text="Export Excel", command=self.export_excel_click).pack(side="left", padx=8)
        ttk.Button(top, text="Export HTML Tables", command=self.export_html_click).pack(side="left")
        self.status = ttk.Label(top, text="Ready")
        self.status.pack(side="left", padx=16)

        selector = ttk.Frame(self)
        selector.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(selector, text="Dataset").pack(side="left")
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(selector, textvariable=self.dataset_var, state="readonly", width=40)
        self.dataset_combo.pack(side="left", padx=8)
        self.dataset_combo.bind("<<ComboboxSelected>>", self.on_dataset_change)

        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.tree = ttk.Treeview(table_frame, show="headings")
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        self.log_text = tk.Text(self, height=8)
        self.log_text.pack(fill="x", padx=10, pady=(0, 10))
        self.refresh_data()

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")

    def set_status(self, msg: str) -> None:
        self.status.configure(text=msg)
        self.update_idletasks()

    def refresh_data(self) -> None:
        def _run() -> None:
            try:
                self.set_status("Refreshing...")
                ds = build_datasets()
                self.datasets = ds
                names = list(ds.keys())
                self.dataset_combo["values"] = names
                if names:
                    self.dataset_var.set(names[0])
                    self.show_dataset(names[0])
                self.log("Refresh complete.")
                self.set_status("Ready")
            except Exception as exc:
                self.log(f"Refresh failed: {exc}")
                self.log(traceback.format_exc())
                self.set_status("Refresh failed")
                messagebox.showerror("Refresh failed", str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def show_dataset(self, name: str) -> None:
        df = self.datasets.get(name, pd.DataFrame()).copy()
        cols = list(df.columns)
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=140, anchor="w")
        for _, row in df.head(500).iterrows():
            self.tree.insert("", "end", values=[row.get(c, "") for c in cols])
        self.log(f"Loaded view: {name} ({len(df)} rows)")

    def on_dataset_change(self, _evt: object) -> None:
        name = self.dataset_var.get()
        if name:
            self.show_dataset(name)

    def export_excel_click(self) -> None:
        if not self.datasets:
            messagebox.showinfo("No data", "Please refresh data first.")
            return
        try:
            path = export_excel(self.datasets)
            self.log(f"Excel exported: {path}")
            messagebox.showinfo("Export complete", f"Excel saved:\n{path}")
        except Exception as exc:
            self.log(f"Excel export failed: {exc}")
            messagebox.showerror("Export failed", str(exc))

    def export_html_click(self) -> None:
        if not self.datasets:
            messagebox.showinfo("No data", "Please refresh data first.")
            return
        try:
            path = export_html_tables(self.datasets)
            self.log(f"HTML tables exported: {path}")
            messagebox.showinfo("Export complete", f"HTML files saved in:\n{path}")
        except Exception as exc:
            self.log(f"HTML export failed: {exc}")
            messagebox.showerror("Export failed", str(exc))


if __name__ == "__main__":
    app = DesktopApp()
    app.mainloop()

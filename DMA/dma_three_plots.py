# dma_plots.py
# Creates:
#   1) E′ vs T (overlay by frequency, log-y)
#   2) E″ vs T (overlay by frequency, log-y)
#   3) tan δ vs T (overlay by frequency, linear-y)
#   4) tan δ vs T per-frequency with Tg (tan δ peak) marked
#
# Usage: put next to your CSVs and run:  python dma_three_plots_plus_tg.py
# Requires: pandas, numpy, matplotlib  (optional: scipy for Savitzky–Golay)

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional smoothing: SciPy if available, else rolling mean
try:
    from scipy.signal import savgol_filter
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ----------------------- config -----------------------
INPUT_GLOB = "*.csv"           # matches e.g. "Sn_Pristine_AllData.xlsx - * Hz.csv"
OUTDIR = "dma_figs"            # output directory
SMOOTH_WINDOW = 31             # odd integer; auto-shrinks for short series
SMOOTH_POLY = 3
# ------------------------------------------------------

def normalise_col(c: str) -> str:
    c2 = str(c)
    for k, v in {"°":"deg", "º":"deg", "Â":"", "µ":"u", "(": "", ")": "", "/":"_per_", " ":"_", "%":"percent"}.items():
        c2 = c2.replace(k, v)
    return c2.strip().lower()

def read_dma_csv(path: str) -> pd.DataFrame:
    # Files have a header row and then a units row to skip
    df = pd.read_csv(path, header=0, skiprows=[1])
    df = df.rename(columns={c: normalise_col(c) for c in df.columns})

    # Coerce to numeric where possible
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    # Standard names
    rename_map = {
        "temperature": "temp_c",
        "tandelta": "tan_delta",
        "storage_modulus": "e_storage_mpa",
        "loss_modulus": "e_loss_mpa",
        "oscillation_strain": "strain_percent",
        "oscillation_stress": "stress_mpa",
        "step_time": "time_s",
    }
    for src, dst in rename_map.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})

    # Frequency (Hz)
    if "angular_frequency" in df.columns:
        df["freq_hz"] = df["angular_frequency"] / (2*np.pi)
    elif "frequency" in df.columns:
        df["freq_hz"] = df["frequency"]
    else:
        df["freq_hz"] = np.nan

    # Keep essentials
    keep = ["temp_c", "freq_hz", "tan_delta", "e_storage_mpa", "e_loss_mpa"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].dropna(subset=["temp_c", "freq_hz", "e_storage_mpa", "e_loss_mpa", "tan_delta"], how="any").copy()

    # Frequency label
    df["freq_label"] = df["freq_hz"].round(0).astype(int).astype(str) + " Hz"
    df["source"] = os.path.basename(path)
    return df

def smooth(y: np.ndarray, window=31, poly=3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 7:
        return y
    # choose an odd window not longer than the series
    win = min(max(5, window), len(y) - (len(y)+1) % 2)
    if win % 2 == 0:
        win = max(5, win - 1)
    if HAVE_SCIPY and win >= (poly + 2):
        return savgol_filter(y, window_length=win, polyorder=min(poly, win-2), mode="interp")
    # fallback: centred rolling mean
    s = pd.Series(y).rolling(window=win, center=True, min_periods=1).mean()
    return s.values

def style_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "grid.alpha": 0.30,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.frameon": False,
    })

def plot_vs_temp(df_all: pd.DataFrame, ycol: str, ylabel: str, title: str, fname: str, logy: bool):
    plt.figure()
    for f, d in df_all.groupby("freq_label"):
        d2 = d.sort_values("temp_c")
        y = smooth(d2[ycol].values, window=SMOOTH_WINDOW, poly=SMOOTH_POLY)
        plt.plot(d2["temp_c"].values, y, linewidth=1.6, label=f)
    plt.xlabel("Temperature (°C)")
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.legend(title="Frequency")
    plt.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    outpath = os.path.join(OUTDIR, fname)
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")

def tg_from_tandelta_peak(temp_c: np.ndarray, tan_delta: np.ndarray) -> float:
    # Smooth then take the maximum
    y = smooth(tan_delta, window=SMOOTH_WINDOW, poly=SMOOTH_POLY)
    idx = int(np.nanargmax(y))
    return float(temp_c[idx])

def plot_tandelta_with_tg_per_frequency(df_all: pd.DataFrame):
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for f, d in df_all.groupby("freq_label"):
        d2 = d.sort_values("temp_c")
        tg_c = tg_from_tandelta_peak(d2["temp_c"].values, d2["tan_delta"].values)
        rows.append((f, tg_c))

        plt.figure()
        plt.plot(d2["temp_c"].values, d2["tan_delta"].values, linewidth=1.6, label=f)
        plt.axvline(tg_c, linestyle="--", linewidth=1.6, label=f"Tg (tan δ peak) ≈ {tg_c:.1f}°C")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("tan δ")
        plt.title(f"tan δ vs Temperature — {f}")
        plt.legend()
        plt.tight_layout()
        outpath = os.path.join(OUTDIR, f"tan_delta_T_with_Tg_{f.replace(' ','_')}.png")
        plt.savefig(outpath, bbox_inches="tight")
        print(f"Saved: {outpath}")
    # Optional: save a small table of Tg by frequency
    tg_df = pd.DataFrame(rows, columns=["Frequency", "Tg_tandelta_peak_C"])
    tg_df.sort_values("Frequency", inplace=True)
    tg_df.to_csv(os.path.join(OUTDIR, "Tg_tandelta_peaks.csv"), index=False)
    print("Saved:", os.path.join(OUTDIR, "Tg_tandelta_peaks.csv"))

def main():
    style_matplotlib()
    paths = sorted(glob.glob(INPUT_GLOB))
    if not paths:
        raise SystemExit("No CSV files found. Adjust INPUT_GLOB.")

    frames = []
    for p in paths:
        try:
            frames.append(read_dma_csv(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not frames:
        raise SystemExit("No readable DMA data found.")

    df = pd.concat(frames, ignore_index=True)
    freqs = ", ".join(sorted(df["freq_label"].unique().tolist()))
    print("Frequencies found:", freqs)

    # Overlay plots
    plot_vs_temp(df, "e_storage_mpa", "Storage modulus E′ (MPa)",
                 "DMA: Storage modulus vs Temperature", "Eprime_vs_T.png", logy=True)
    plot_vs_temp(df, "e_loss_mpa", "Loss modulus E″ (MPa)",
                 "DMA: Loss modulus vs Temperature", "Edoubleprime_vs_T.png", logy=True)
    plot_vs_temp(df, "tan_delta", "tan δ",
                 "DMA: tan δ vs Temperature", "tan_delta_vs_T.png", logy=False)

    # Per-frequency tan δ with Tg marker
    plot_tandelta_with_tg_per_frequency(df)

if __name__ == "__main__":
    main()

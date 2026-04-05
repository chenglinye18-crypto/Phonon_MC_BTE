from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge in-plane interval+rolling-window boxplot datasets: "
            "use new s5050_n100k results for 2/5/10/20/50 nm and legacy results for 100/200/400 nm, "
            "then export summary CSV including box edges and whiskers."
        )
    )
    parser.add_argument(
        "--small-width-csv",
        default="output/in_plane_scatter_s5050_n100k_boxplot_interval_tail5/in_plane_scatter_kappa_boxplot_interval_tail5.csv",
        help="Raw boxplot CSV for 2/5/10/20/50 nm.",
    )
    parser.add_argument(
        "--large-width-csv",
        default="output/in_plane_scatter_width_boxplot_interval_tail5/in_plane_scatter_kappa_boxplot_interval_tail5.csv",
        help="Raw boxplot CSV for legacy 100/200/400 nm results.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/in_plane_scatter_merged_boxplot_summary",
        help="Output directory.",
    )
    return parser.parse_args()


def tukey_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        raise ValueError("empty sample")
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inlier_mask = (arr >= lower_fence) & (arr <= upper_fence)
    inliers = arr[inlier_mask]
    lower_whisker = float(inliers.min()) if inliers.size else float(arr.min())
    upper_whisker = float(inliers.max()) if inliers.size else float(arr.max())
    return {
        "n_samples": int(arr.size),
        "mean_W_mK": float(arr.mean()),
        "std_W_mK": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
        "median_W_mK": float(np.median(arr)),
        "min_W_mK": float(arr.min()),
        "max_W_mK": float(arr.max()),
        "q1_W_mK": q1,
        "q3_W_mK": q3,
        "iqr_W_mK": float(iqr),
        "lower_fence_W_mK": float(lower_fence),
        "upper_fence_W_mK": float(upper_fence),
        "lower_whisker_W_mK": lower_whisker,
        "upper_whisker_W_mK": upper_whisker,
        "n_outliers": int(arr.size - inliers.size),
    }


def main() -> None:
    args = parse_args()
    small_path = Path(args.small_width_csv).expanduser().resolve()
    large_path = Path(args.large_width_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    small_df = pd.read_csv(small_path)
    large_df = pd.read_csv(large_path)

    small_widths = {2.0, 5.0, 10.0, 20.0, 50.0}
    large_widths = {100.0, 200.0, 400.0}

    small_df = small_df.loc[small_df["width_nm"].astype(float).isin(small_widths)].copy()
    small_df["source_dataset"] = "s5050_n100k"
    large_df = large_df.loc[large_df["width_nm"].astype(float).isin(large_widths)].copy()
    large_df["source_dataset"] = "legacy_interval_tail5"

    merged_raw = pd.concat([small_df, large_df], ignore_index=True)
    merged_raw["width_nm"] = merged_raw["width_nm"].astype(float)
    merged_raw["temperature_K"] = merged_raw["temperature_K"].astype(float)

    summary_rows: list[dict[str, object]] = []
    for (temperature_k, width_nm), sub in merged_raw.groupby(["temperature_K", "width_nm"], sort=True):
        stats = tukey_stats(sub["kappa_fourier_W_mK"].to_numpy(dtype=np.float64))
        first = sub.iloc[0]
        row: dict[str, object] = {
            "temperature_K": float(temperature_k),
            "width_nm": float(width_nm),
            "width_label": str(first["width_label"]),
            "source_dataset": ";".join(sorted(sub["source_dataset"].astype(str).unique().tolist())),
        }
        row.update(stats)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(["temperature_K", "width_nm"], kind="stable").reset_index(drop=True)
    raw_out = output_dir / "in_plane_scatter_merged_boxplot_raw_samples.csv"
    summary_out = output_dir / "in_plane_scatter_merged_boxplot_summary.csv"
    merged_raw.to_csv(raw_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    print(f"[ok] raw -> {raw_out}")
    print(f"[ok] summary -> {summary_out}")


if __name__ == "__main__":
    main()

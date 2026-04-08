#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2026 Valerio Lo Brano
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ---------------------------------------------------------------------------
# If you use this software in your research, please cite:
#
#   Lo Brano, V. (2026). Open and Reproducible Estimation of PV
#   Single-Diode Parameters from Datasheet Data. Energy Reports.
# ---------------------------------------------------------------------------
"""
analysis_results.py — Statistical analysis of batch estimation results.

Reads an Excel file produced by batch_validation.py and computes:

  Part 1 — PDF/CDF analysis
    - Probability density and cumulative distribution functions for RMSE
      and runtime, with outlier handling and log-scale plots.
    - Detailed multi-panel analysis per column.

  Part 2 — Advanced statistical analysis
    - Descriptive statistics on RMSE and Error_Pmp.
    - Dominance analysis of partial errors (which error term drives the
      total objective function).
    - Five-parameter statistics grouped by module technology.

Results are saved to an Excel file and figures saved as PNG.

Usage
-----
    python analysis_results.py

Edit the CONFIGURATION section at the top to set file paths.
"""

from __future__ import annotations

__version__ = "1.1.0"
__date__    = "2026-04-08"

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from openpyxl import load_workbook

# ==============================================================================
# CONFIGURATION — edit these paths as needed
# ==============================================================================
EXCEL_FILE    = "pv_batch_analysis_results_deep.xlsx"  # input from batch_validation.py
RESULTS_EXCEL = "results_summary.xlsx"
RESULTS_TXT   = "results_summary.txt"
SAVE_FIGURES  = True   # set to False to suppress PNG output


# ==============================================================================
# PART 1 — PDF / CDF analysis
# ==============================================================================

def analyze_column_robust(data, column_name, ax_pdf, ax_cdf):
    """Analyse a column with outlier handling."""
    clean_data = data[column_name].dropna()

    if len(clean_data) == 0:
        print(f"No valid data in column {column_name}")
        return clean_data

    # Compute basic statistics
    mean_val = clean_data.mean()
    median_val = clean_data.median()

    # Full statistics
    print(f"\n{'='*50}")
    print(f"STATISTICAL ANALYSIS: {column_name}")
    print(f"{'='*50}")
    print(f"Number of samples: {len(clean_data)}")
    print(f"Mean value: {mean_val:.6f}")
    print(f"Median: {median_val:.6f}")
    print(f"Standard deviation: {clean_data.std():.6f}")
    print(f"Minimum: {clean_data.min():.6f}")
    print(f"Maximum: {clean_data.max():.6f}")

    # Identify outliers using IQR
    Q1 = clean_data.quantile(0.25)
    Q3 = clean_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
    print(f"Number of outliers (IQR method): {len(outliers)}")
    print(f"Outlier bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")

    # Improved PDF with adaptive logarithmic bins
    if column_name == "RMSE":
        bin_min = max(clean_data.min(), 1e-4)
        bin_max = clean_data.quantile(0.98)
        bins = np.logspace(np.log10(bin_min), np.log10(bin_max), 40)
        ax_pdf.hist(clean_data, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax_pdf.set_xscale('log')
        ax_pdf.set_yscale('log')
        ax_pdf.set_title(f'PDF - {column_name}\n(Log-Log Scale)', fontsize=12, fontweight='bold')
    else:
        ax_pdf.hist(clean_data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', log=True)
        ax_pdf.set_yscale('log')
        ax_pdf.set_title(f'PDF - {column_name}\n(Log scale)', fontsize=12, fontweight='bold')

    ax_pdf.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax_pdf.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
    ax_pdf.set_xlabel(column_name)
    ax_pdf.set_ylabel('Probability density')
    ax_pdf.legend()
    ax_pdf.grid(True, which="both", ls='--', alpha=0.5)

    # CDF
    sorted_data = np.sort(clean_data)
    cdf = np.arange(len(sorted_data)) / float(len(sorted_data))
    ax_cdf.plot(sorted_data, cdf, label="Complete CDF", color="blue", linewidth=2)

    # CDF without outliers
    threshold = clean_data.quantile(0.98)
    filtered_data = clean_data[clean_data < threshold]
    if len(filtered_data) > 0:
        filtered_sorted = np.sort(filtered_data)
        cdf_filtered = np.arange(len(filtered_sorted)) / float(len(filtered_sorted))
        ax_cdf.plot(filtered_sorted, cdf_filtered, label="Without outliers", color="red", linestyle="--", linewidth=2)

    ax_cdf.set_xlabel(column_name)
    ax_cdf.set_ylabel('Cumulative probability')
    ax_cdf.set_title(f'CDF - {column_name}', fontsize=12, fontweight='bold')
    ax_cdf.grid(True, which="both", ls='--', alpha=0.5)
    ax_cdf.legend()

    return clean_data


def create_detailed_analysis(data, column_name):
    """Create a detailed multi-panel analysis for a single column."""
    clean_data = data[column_name].dropna()

    if len(clean_data) == 0:
        return

    # Create figure with multiple visualisations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Detailed analysis: {column_name}', fontsize=16, fontweight='bold')

    # 1. Standard histogram
    axes[0, 0].hist(clean_data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].set_title('Standard Distribution')
    axes[0, 0].set_xlabel(column_name)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Log-scale histogram
    if column_name == 'RMSE':
        positive_data = clean_data[clean_data > 0]
        if len(positive_data) > 0:
            bin_min = max(positive_data.min(), 1e-4)
            bin_max = positive_data.quantile(0.98)
            bins = np.logspace(np.log10(bin_min), np.log10(bin_max), 40)
            axes[0, 1].hist(positive_data, bins=bins, alpha=0.7, color='lightgreen',
                            edgecolor='black', log=True)
            axes[0, 1].set_xscale('log')
            axes[0, 1].set_title('Distribution (Log-Log Scale)')
        else:
            axes[0, 1].hist(clean_data, bins=50, alpha=0.7, color='lightgreen',
                            edgecolor='black', log=True)
            axes[0, 1].set_title('Distribution (Log Scale)')
    else:
        axes[0, 1].hist(clean_data, bins=30, alpha=0.7, color='lightgreen',
                        edgecolor='black', log=True)
        axes[0, 1].set_title('Distribution (Log Scale)')

    axes[0, 1].set_xlabel(column_name)
    axes[0, 1].set_ylabel('Frequency (Log)')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Box plot
    axes[0, 2].boxplot(clean_data, vert=True)
    axes[0, 2].set_title('Box Plot')
    axes[0, 2].set_ylabel(column_name)
    axes[0, 2].grid(True, alpha=0.3)

    # 4. PDF with kernel density estimation
    if column_name == 'RMSE':
        positive_data = clean_data[clean_data > 0]
        if len(positive_data) > 0:
            bin_min = max(positive_data.min(), 1e-4)
            bin_max = positive_data.quantile(0.98)
            bins = np.logspace(np.log10(bin_min), np.log10(bin_max), 40)
            positive_data.hist(bins=bins, density=True, alpha=0.7, ax=axes[1, 0],
                               color='orange', edgecolor='black', log=True)
            positive_data.plot.density(ax=axes[1, 0], linewidth=2, color='red')
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_title('PDF with KDE (Log-Log Scale)')
        else:
            clean_data.hist(bins=50, density=True, alpha=0.7, ax=axes[1, 0],
                            color='orange', edgecolor='black', log=True)
            clean_data.plot.density(ax=axes[1, 0], linewidth=2, color='red')
            axes[1, 0].set_title('PDF with KDE (Log Scale)')
    else:
        clean_data.hist(bins=50, density=True, alpha=0.7, ax=axes[1, 0],
                        color='orange', edgecolor='black', log=True)
        clean_data.plot.density(ax=axes[1, 0], linewidth=2, color='red')
        axes[1, 0].set_title('PDF with KDE (Log Scale)')

    axes[1, 0].set_xlabel(column_name)
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. CDF
    sorted_data = np.sort(clean_data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axes[1, 1].plot(sorted_data, cdf, 'g-', linewidth=2)
    axes[1, 1].set_title('CDF')
    axes[1, 1].set_xlabel(column_name)
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Q-Q plot for normality assessment
    stats.probplot(clean_data, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q Plot')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    if SAVE_FIGURES:
        fname = f'detailed_analysis_{column_name.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()


# ==============================================================================
# PART 2 — Advanced statistical analysis
# ==============================================================================

def advanced_analysis(df: pd.DataFrame) -> None:
    """Run advanced statistical analysis on batch estimation results.

    Performs three sub-analyses:
      1. Descriptive statistics on RMSE and Error_Pmp.
      2. Dominance analysis of partial errors.
      3. Five-parameter statistics grouped by module technology.

    Results are written to *RESULTS_EXCEL* and *RESULTS_TXT*; figures are
    saved as PNG when SAVE_FIGURES is True.
    """
    logger = logging.getLogger("analysis_results")
    summary_txt = []

    with pd.ExcelWriter(RESULTS_EXCEL, engine='xlsxwriter') as excel_writer:

        # =============================
        # 1. STATISTICAL ANALYSIS OF ERRORS
        # =============================
        for col in ["RMSE", "Error_Pmp"]:
            data = df[col].dropna()
            if data.empty:
                continue

            stats_dict = {
                'max':      data.max(),
                'min':      data.min(),
                'mean':     data.mean(),
                'median':   data.median(),
                'mode':     data.mode().iloc[0] if not data.mode().empty else np.nan,
                'std':      data.std(),
                'skewness': data.skew(),
            }
            mode_val = stats_dict['mode']
            mode_position = data[data == mode_val].index.tolist()

            summary_txt.append(f"\n===== {col} =====")
            for k, v in stats_dict.items():
                summary_txt.append(f"{k}: {v:.6f}")
            summary_txt.append(f"Modal value position(s): {mode_position[:5]} ...")

            stats_df = pd.DataFrame([stats_dict])
            stats_df.to_excel(excel_writer, sheet_name=f"{col}_Stats", index=False)

        # =============================
        # 2. DOMINANCE OF PARTIAL ERRORS
        # =============================
        error_cols = [
            'Error_Voc', 'Error_Isc', 'Error_Vmp',
            'Error_Imp', 'Error_Pmp', 'Error_Stat_Residual',
        ]
        error_cols = [c for c in error_cols if c in df.columns]
        if not error_cols:
            logger.warning(
                "No error columns found in input file — skipping dominance analysis."
            )
        else:
            df_err = df[error_cols].copy()
            df_err = df_err.replace([np.inf, -np.inf], np.nan).dropna(how='any')

            # Mean percentage contribution of each partial error
            total_error = df_err.sum(axis=1)
            perc = (df_err.T / total_error).T * 100.0
            dominance = perc.mean().sort_values(ascending=False)

            summary_txt.append("\n===== Dominance of Partial Errors =====")
            for k, v in dominance.items():
                summary_txt.append(f"{k}: {v:.2f}%")

            dominance_df = dominance.reset_index()
            dominance_df.columns = ['Error Type', 'Mean Contribution (%)']
            dominance_df.to_excel(excel_writer, sheet_name='Partial_Errors', index=False)

            if SAVE_FIGURES:
                plt.figure(figsize=(8, 5))
                sns.barplot(x='Mean Contribution (%)', y='Error Type', data=dominance_df,
                            palette='viridis')
                plt.title('Dominance of Partial Errors')
                plt.tight_layout()
                plt.savefig('partial_error_dominance.png', dpi=300)
                plt.close()

        # =============================
        # 3. PARAMETER STATISTICS BY TECHNOLOGY
        # =============================
        param_cols = ['I_L_ref', 'I_o_ref', 'R_s', 'R_sh_ref', 'a_ref']
        if 'Technology' in df.columns:
            grouped = df.groupby('Technology')[param_cols]
            param_stats = grouped.agg(['mean', 'std']).reset_index()
            param_stats.to_excel(excel_writer, sheet_name='Parameters_by_Tech', index=False)

            if SAVE_FIGURES:
                fig, axes = plt.subplots(1, len(param_cols), figsize=(20, 5))
                for i, param in enumerate(param_cols):
                    sns.boxplot(data=df, x='Technology', y=param, ax=axes[i], palette='Set2')
                    axes[i].set_title(param)
                    axes[i].tick_params(axis='x', rotation=45)
                plt.tight_layout()
                plt.savefig('parameters_by_technology.png', dpi=300)
                plt.close()

            summary_txt.append("\n===== Parameter Statistics by Technology =====")
            summary_txt.append(str(param_stats.head()))

    # =============================
    # Save text results
    # =============================
    with open(RESULTS_TXT, 'w') as f:
        f.write("\n".join(summary_txt))

    print("\n".join(summary_txt))


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Run the complete analysis pipeline (Part 1 + Part 2)."""
    global EXCEL_FILE, RESULTS_EXCEL, RESULTS_TXT, SAVE_FIGURES

    parser = argparse.ArgumentParser(
        prog="pvfit5-analysis",
        description="Statistical analysis of batch estimation results.",
    )
    parser.add_argument("input", nargs="?", default=EXCEL_FILE,
                        help=f"Input Excel file (default: {EXCEL_FILE})")
    parser.add_argument("--output-excel", default=RESULTS_EXCEL,
                        help=f"Output Excel file (default: {RESULTS_EXCEL})")
    parser.add_argument("--output-txt", default=RESULTS_TXT,
                        help=f"Output text file (default: {RESULTS_TXT})")
    parser.add_argument("--no-figures", action="store_true",
                        help="Suppress PNG figure output")
    args = parser.parse_args()

    EXCEL_FILE = args.input
    RESULTS_EXCEL = args.output_excel
    RESULTS_TXT = args.output_txt
    SAVE_FIGURES = not args.no_figures

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("analysis_results")

    input_path = Path(EXCEL_FILE)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading: %s", EXCEL_FILE)
    df = pd.read_excel(input_path)

    # Part 1 — PDF / CDF analysis
    for col in ["RMSE", "Runtime (s)"]:
        if col in df.columns:
            fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(14, 5))
            analyze_column_robust(df, col, ax_pdf, ax_cdf)
            plt.tight_layout()
            if SAVE_FIGURES:
                fname = f"analysis_{col.replace(' ', '_').replace('/', '_')}.png"
                fig.savefig(fname, dpi=150)
                logger.info("Figure saved: %s", fname)
            plt.show()
            create_detailed_analysis(df, col)

    # Part 2 — Advanced analysis
    advanced_analysis(df)
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()

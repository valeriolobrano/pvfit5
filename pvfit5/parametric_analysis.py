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
parametric_analysis.py — Statistical analysis of RMSE and Runtime per Module Type.

This script reads an Excel file containing photovoltaic module data and computes
for each distinct "Module Type" the following statistics for both "RMSE" and "runtime (s)":

    - count (number of samples)
    - mean
    - median
    - standard deviation
    - minimum
    - maximum
    - 5th, 25th, 50th, 75th, and 95th percentiles

Results are saved into a new Excel file with two sheets: one for RMSE and one for runtime.

Compatible with: Python 3.12+, pandas 2.x, numpy 1.26+.
"""

from __future__ import annotations

__version__ = "1.0.0"
__date__    = "2026-04-08"

import logging
from pathlib import Path
import pandas as pd
import numpy as np

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("module_analysis")


def compute_statistics(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    """Compute grouped descriptive statistics for a given numeric column.

    Args:
        df: Input DataFrame.
        value_col: Column name containing numeric values (e.g., "RMSE").
        group_col: Column name to group by (e.g., "Module Type").

    Returns:
        DataFrame with one row per group and computed statistics.
    """
    def _percentile(x: pd.Series, q: float) -> float:
        return np.percentile(x, q)

    stats = df.groupby(group_col, dropna=False)[value_col].agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        p05=lambda x: _percentile(x, 5),
        p25=lambda x: _percentile(x, 25),
        p50=lambda x: _percentile(x, 50),
        p75=lambda x: _percentile(x, 75),
        p95=lambda x: _percentile(x, 95),
    )

    return stats.reset_index()


def main(input_file: str, output_file: str) -> None:
    """Perform the statistical analysis and export results to Excel.

    Args:
        input_file: Path to input Excel file.
        output_file: Path to save output Excel file.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading Excel file: {input_file}")
    df = pd.read_excel(input_path)

    required_cols = {"Module Type", "RMSE", "Runtime (s)"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing expected columns in Excel: {missing}")

    logger.info("Computing statistics per Module Type...")

    rmse_stats = compute_statistics(df, "RMSE", "Module Type")
    runtime_stats = compute_statistics(df, "Runtime (s)", "Module Type")

    # Round for readability
    rmse_stats = rmse_stats.round(6)
    runtime_stats = runtime_stats.round(6)

    logger.info(f"Saving results to {output_file}")
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        rmse_stats.to_excel(writer, index=False, sheet_name="RMSE_Stats")
        runtime_stats.to_excel(writer, index=False, sheet_name="Runtime_Stats")

    logger.info("Analysis completed successfully.")


if __name__ == "__main__":
    try:
        # Example usage: adjust filenames as needed
        main("20k.xlsx", "module_statistics.xlsx")
    except Exception as e:
        logger.exception("Error during analysis: %s", e)
        raise

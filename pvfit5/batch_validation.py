#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_validation.py — Batch parameter estimation for multiple PV modules.

Runs the pvfit5 genetic algorithm on a selection of modules from the
pvlib CEC database and saves results to an Excel file.

Usage examples
--------------
# Analyse 50 modules in alphabetical order
python batch_validation.py -n 50 --selection alpha

# Analyse 50 random modules with a fixed seed
python batch_validation.py -n 50 --selection random --seed 42

# Analyse 10 modules with a custom output filename
python batch_validation.py -n 10 --output my_results.xlsx

# Use the default (5 modules)
python batch_validation.py
"""
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

__version__ = "1.0.0"
__date__    = "2026-04-08"

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import logging
import argparse
from typing import List, Dict
from pvfit5.find_pv_parameters import fit_parameters, PVModuleData, STC, GAConfig
import pvfit5.find_pv_parameters as fpv
from pvlib import pvsystem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CEC database column aliases for robust parameter extraction
CEC_ALIAS = {
    "voc": ["V_oc_ref", "V_oc"],
    "isc": ["I_sc_ref", "I_sc"],
    "vmp": ["V_mp_ref", "V_mp"],
    "imp": ["I_mp_ref", "I_mp"],
    "pmp": ["P_mp_ref", "P_mp"]
}

def _first_existing(series: pd.Series, cols: List[str]) -> float | None:
    """Return first available value from a set of column names."""
    for c in cols:
        if c in series and pd.notna(series[c]):
            return float(series[c])
    return None

def extract_module_info(module_series: pd.Series) -> Dict[str, str]:
    """Extract module type and manufacturer from module series."""
    Technology = ""
    manufacturer = ""

    # Try various possible column names
    for col in ['Technology', 'technology', 'Model']:
        if col in module_series and pd.notna(module_series[col]):
            Technology = str(module_series[col])
            break

    for col in ['Manufacturer', 'Maker', 'Brand']:
        if col in module_series and pd.notna(module_series[col]):
            manufacturer = str(module_series[col])
            break

    return {
        'Technology': Technology,
        'manufacturer': manufacturer
    }

def extract_keypoints_stc(row: pd.Series) -> Dict[str, float] | None:
    """Extract Voc, Isc, Vmp, Imp, Pmp with robust fallbacks (Pmp = Vmp*Imp if absent)."""
    voc = _first_existing(row, CEC_ALIAS["voc"])
    isc = _first_existing(row, CEC_ALIAS["isc"])
    vmp = _first_existing(row, CEC_ALIAS["vmp"])
    imp = _first_existing(row, CEC_ALIAS["imp"])
    pmp = _first_existing(row, CEC_ALIAS["pmp"])

    if pmp is None and (vmp is not None and imp is not None):
        pmp = vmp * imp

    vals = {"voc": voc, "isc": isc, "vmp": vmp, "imp": imp, "pmp": pmp}
    if any(v is None for v in vals.values()):
        return None
    return {k: float(v) for k, v in vals.items()}

def retrieve_cec_dataframe() -> pd.DataFrame:
    """Retrieve the CEC module DataFrame from pvlib."""
    df = pvsystem.retrieve_sam("CECMod")
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("retrieve_sam('CECMod') did not return a DataFrame.")
    return df

def select_modules(df: pd.DataFrame, how: str, n: int, seed: int | None = None) -> List[str]:
    """Select N module names: 'alpha' (first N alphabetically) or 'random'."""
    names = sorted(df.columns)  # In CECMod the module names are the columns
    if how == "alpha":
        return names[:n]
    elif how == "random":
        rng = np.random.default_rng(seed)
        if len(names) <= n:
            return names
        idx = rng.choice(len(names), size=n, replace=False)
        return [names[i] for i in sorted(idx)]
    else:
        raise ValueError("'how' must be 'alpha' or 'random'")

def get_pv_modules_from_database(num_modules=5, selection_method='alpha', seed=None):
    """Extract PV modules from pvlib database using robust parameter extraction"""
    try:
        cec_df = retrieve_cec_dataframe()
        selected_module_names = select_modules(cec_df, selection_method, num_modules, seed)

        modules_data = []

        for module_name in selected_module_names:
            module_series = cec_df[module_name]

            keypoints = extract_keypoints_stc(module_series)

            if keypoints is not None:
                # Extract module info
                module_info = extract_module_info(module_series)

                modules_data.append({
                    'name': module_name,
                    'voc': keypoints['voc'],
                    'isc': keypoints['isc'],
                    'pmax': keypoints['pmp'],
                    'vmp': keypoints['vmp'],
                    'imp': keypoints['imp'],
                    'Technology': module_info['Technology'],
                    'manufacturer': module_info['manufacturer'],
                    'full_series': module_series  # Full series kept for debugging
                })
                logging.info(f"Added module: {module_name}")
                logging.info(f"  Manufacturer: {module_info['manufacturer']}, Type: {module_info['Technology']}")
            else:
                logging.warning(f"Skipping module {module_name}: missing required parameters")

        return modules_data

    except Exception as e:
        logging.error(f"Error accessing database: {e}")
        return []

def run_batch_analysis(modules_data):
    """Run parameter extraction for multiple modules"""
    results = []

    for i, module in enumerate(modules_data, 1):
        logging.info(f"Processing module {i}/{len(modules_data)}: {module['name']}")

        try:
            # Create PV module data object
            nd = PVModuleData(
                voc=module['voc'],
                isc=module['isc'],
                pmax=module['pmax'],
                vmp=module['vmp'],
                imp=module['imp']
            )

            # Configure STC and GA
            stc = STC()
            ga = GAConfig(error_target=1e-3)

            # Monkey-patch the plotting functions to disable them

            # Save original functions
            original_final_plot = fpv._final_plot
            original_init_graph1 = fpv._init_graph1_live
            original_update_graph1 = fpv._update_graph1_live

            # Replace with no-op functions
            def no_op_final_plot(*args, **kwargs):
                fig = plt.figure(figsize=(1, 1))
                plt.close(fig)  # Immediately close the figure
                return fig

            def no_op_init_graph1(*args, **kwargs):
                return None, None, None

            def no_op_update_graph1(*args, **kwargs):
                return None

            # Apply patches
            fpv._final_plot = no_op_final_plot
            fpv._init_graph1_live = no_op_init_graph1
            fpv._update_graph1_live = no_op_update_graph1

            # Also disable interactive mode
            fpv._enable_interactive = lambda: False

            # Run fitting with all plots disabled
            result, summary_string = fit_parameters(
                nd, stc, ga,
                module_name=module['name'],
                seed=None,
                live_plot=False  # Disable all plots for batch processing
            )

            # Restore original functions (in case they're needed elsewhere)
            fpv._final_plot = original_final_plot
            fpv._init_graph1_live = original_init_graph1
            fpv._update_graph1_live = original_update_graph1

            # Add module info to results
            result['module_name'] = module['name']
            result['module_data'] = {
                'voc': module['voc'],
                'isc': module['isc'],
                'pmax': module['pmax'],
                'vmp': module['vmp'],
                'imp': module['imp'],
                'Technology': module.get('Technology', ''),
                'manufacturer': module.get('manufacturer', '')
            }

            results.append(result)
            logging.info(f"Completed: {module['name']} - Error: {result['final_error']:.2e}")

            # Clean up any remaining figures
            plt.close('all')

        except Exception as e:
            logging.error(f"Error processing {module['name']}: {e}")
            # Ensure we clean up even on error
            plt.close('all')
            continue

    return results

def save_to_excel(results, filename="pv_batch_analysis_results_deep.xlsx"):
    """Save results to Excel file"""

    excel_data = []

    for result in results:
        best = result['best_individual']
        sdm = result['result_sdm']
        module_data = result['module_data']

        # Use individual errors from the result dict
        individual_errors = result.get('individual_errors', {})

        # Compute RMSE across the four key-point errors
        rmse = np.sqrt(
            (individual_errors.get('e_voc', 0)**2 +
             individual_errors.get('e_isc', 0)**2 +
             individual_errors.get('e_vmp', 0)**2 +
             individual_errors.get('e_imp', 0)**2) / 4
        )
        # Root Mean Square Error of the normalised relative errors.
        #
        # Interpretation:
        #   RMSE = 0.01  means ~1%  relative RMS error
        #   RMSE = 0.05  means ~5%  relative RMS error
        #   RMSE = 0.10  means ~10% relative RMS error
        # It is a synthetic index that penalises larger errors more heavily
        # (due to squaring) and provides an aggregate measure of the fit
        # quality across all characteristic points of the I-V curve.

        row = {
            'Module Name': result['module_name'],
            'Manufacturer': module_data.get('manufacturer', ''),
            'Module Type': module_data.get('Technology', ''),
            'Voc (V)': module_data['voc'],
            'Isc (A)': module_data['isc'],
            'Pmax (W)': module_data['pmax'],
            'Vmp (V)': module_data['vmp'],
            'Imp (A)': module_data['imp'],
            'alpha_sc (fixed A/°C)': float(fpv.alpha_sc_fixed_abs(module_data['isc'])),
            'a_ref': best['a_ref'],
            'I_L_ref (A)': best['I_L_ref'],
            'I_o_ref (A)': best['I_o_ref'],
            'R_s (Ω)': best['R_s'],
            'R_sh (Ω)': best['R_sh'],
            'nNsVth (V)': best['nNsVth'],
            'Voc_sim (V)': sdm['v_oc'],
            'Isc_sim (A)': sdm['i_sc'],
            'Vmp_sim (V)': sdm['v_mp'],
            'Imp_sim (A)': sdm['i_mp'],
            'Pmax_sim (W)': sdm['p_mp'],
            # Individual errors
            'Error_Voc': individual_errors.get('e_voc', 0),
            'Error_Isc': individual_errors.get('e_isc', 0),
            'Error_Vmp': individual_errors.get('e_vmp', 0),
            'Error_Imp': individual_errors.get('e_imp', 0),
            'Error_Pmp': individual_errors.get('e_pmp', 0),
            'Error_Stat_Residual': individual_errors.get('stat_residual', 0),
            'Final GA Target': result['final_error'],
            'RMSE': rmse,
            'Generations': result['generations_done'],
            'Early Stop': result['early_stop'],
            'Runtime (s)': result['runtime_seconds'],
            'Processors': result['n_processes']
        }
        excel_data.append(row)

    # Create DataFrame and save to Excel
    df = pd.DataFrame(excel_data)

    # Format the Excel output
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='PV Parameters', index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets['PV Parameters']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except Exception:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    logging.info(f"Results saved to {filename}")
    return df

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch analysis of PV modules from pvlib database')
    parser.add_argument('-n', '--num_modules', type=int, default=5,
                       help='Number of modules to analyze (default: 5)')
    parser.add_argument('--selection', choices=['alpha', 'random'], default='alpha',
                       help='Selection method: alpha (alphabetical) or random (default: alpha)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible random selection (default: None)')
    parser.add_argument('--output', type=str, default='pv_batch_analysis_results_deep.xlsx',
                       help='Output Excel filename (default: pv_batch_analysis_results_deep.xlsx)')

    return parser.parse_args()

def main():
    """Main batch analysis function"""
    args = parse_arguments()

    logging.info("Starting batch analysis of PV modules")
    logging.info(f"Arguments: num_modules={args.num_modules}, selection={args.selection}, seed={args.seed}")

    # Get modules from database
    modules_data = get_pv_modules_from_database(
        num_modules=args.num_modules,
        selection_method=args.selection,
        seed=args.seed
    )

    if not modules_data:
        logging.error("No modules found or error accessing database")
        return

    logging.info(f"Found {len(modules_data)} modules for analysis")

    # Display modules being analyzed
    print("\nModules selected for analysis:")
    for module in modules_data:
        print(f"  - {module['name']}: Voc={module['voc']:.1f}V, Isc={module['isc']:.1f}A, Pmax={module['pmax']:.0f}W")

    # Run batch analysis
    results = run_batch_analysis(modules_data)

    if not results:
        logging.error("No results generated from batch analysis")
        return

    # Save to Excel
    df = save_to_excel(results, args.output)

    # Print summary
    logging.info("Batch analysis completed successfully!")
    logging.info(f"Processed {len(results)} modules")
    logging.info(f"Average final error: {df['Final GA Target'].mean():.2e}")
    logging.info(f"Average runtime: {df['Runtime (s)'].mean():.2f} seconds")

    # Print module names that were processed
    print("\nSuccessfully processed modules:")
    for module_name in df['Module Name']:
        print(f"  - {module_name}")

if __name__ == "__main__":
    main()

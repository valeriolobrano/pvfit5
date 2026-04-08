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
#
# Five-Parameter PV Model — Single-Diode Fitting (CEC model, pvlib + DEAP)
#
# Purpose
# -------
# This script estimates the single-diode model (SDM) parameters of a PV module
# at STC using only the PV module datasheet. A Genetic Algorithm (DEAP) minimizes
# the sum of relative errors on Voc, Isc, and Pmp. The fitted parameters are then
# used with `pvlib.pvsystem.singlediode` (Lambert W) to reconstruct the I-V curve.
#
# The script provides:
#   - Graph 1: live visualisation of the adaptation (I-V curve vs PV module key points)
#   - Graph 2: final, publication-ready I-V figure that REMAINS OPEN at the end
#   - Console prints with full summary (runtime in mm:ss, parameters, errors)
#
# Theoretical background
# ----------------------
# Objective function minimized by GA:
#
#   total_error = |V_oc^sim - V_oc*| / V_oc* +
#                  |I_sc^sim - I_sc*| / I_sc* +
#                  |P_mp^sim - P_mp*| / P_mp*
#
# Single-diode model (solved by pvlib):
#
#   I = I_L - I_0 [exp((V + I R_s)/(n N_s V_th)) - 1] - (V + I R_s)/R_sh
#
# Features
# --------
# - Robust GA (tournament, polynomial mutation, elitism, early-stop on E<target).
# - Parallel fitness evaluation (multiprocessing).
# - Graph 1: live adaptation (I-V curve updates during GA) that auto-closes 3 s after end.
# - Graph 2: final plot with annotated error and PV module anchors (kept open).
#
# Tested environment
# ------------------
#   Python          : 3.12.3
#   NumPy           : 2.0.2
#   Matplotlib      : 3.9.2
#   pvlib           : 0.13.1
#   DEAP            : 1.4.2
#   tqdm            : 4.66.5
#
# ---------------------------------------------------------------------------

from __future__ import annotations

__version__ = "1.0.0"
__date__    = "2026-04-08"

# ========================= DEFAULT VALUES ====================================
# These defaults are used when the script is run without CLI arguments.
# When installed via pip, use the command line instead:
#   pvfit5 --voc 36.3 --isc 8.19 --pmax 218.95 --vmp 29.0 --imp 7.55
# Or edit the values below if you prefer to run the script directly.
MODULE_NAME: str = "PVModule"       # Name of the PV module (used for output files)
VOC: float = 36.30                  # Open-circuit voltage at STC (V)
ISC: float = 8.19                   # Short-circuit current at STC (A)
PMAX: float = 218.95                # Maximum power at STC (W)
VMP: float = 29.00                  # Voltage at maximum power point (V)
IMP: float = 7.55                   # Current at maximum power point (A)
ERROR_TARGET: float = 1e-2          # GA early-stop threshold on total relative error
# --- Short-circuit current temperature coefficient ---
# Choose how you specify alpha_sc:
# - "absolute": ALPHA_SC is in A/°C (e.g., 0.05 A/°C)
# - "relative": REL_ALPHA_SC is in 1/°C (e.g., 0.005)
ALPHA_SC_MODE: str = "absolute"     # "absolute" or "relative"
ALPHA_SC: float = 0.05              # A/°C, used if ALPHA_SC_MODE == "absolute"
REL_ALPHA_SC: float = 0.005         # 1/°C, used if ALPHA_SC_MODE == "relative"
# ======================= END OF DEFAULTS =====================================

import argparse
import datetime as _dt
import logging
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools  # type: ignore
from pvlib import pvsystem
from pvlib.pvsystem import calcparams_cec, singlediode
from tqdm import tqdm


# --------------------------- Logging ----------------------------------------

def _setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# --------------------------- Data structures --------------------------------

@dataclass(frozen=True)
class PVModuleData:
    """PV module key points at STC."""
    voc: float   # V
    isc: float   # A
    pmax: float  # W
    vmp: float   # V
    imp: float   # A


@dataclass(frozen=True)
class STC:
    """Standard Test Conditions."""
    effective_irradiance: float = 1000.0  # W/m^2
    cell_temperature_c: float = 25.0      # °C
    EgRef: float = 1.121                  # eV
    dEgdT: float = -0.000267              # eV/K


@dataclass(frozen=True)
class GAConfig:
    """Genetic Algorithm configuration."""
    population_size: int = 500
    num_generations: int = 1500
    cx_prob: float = 0.8
    mut_prob: float = 0.2
    error_target: float = ERROR_TARGET
    tournament_size: int = 3
    eta_mut: float = 20.0
    indpb_mut: float = 0.1
    elite_size: int = 10
    progress_update_period: int = 20 #10  # generations between live updates


@dataclass(frozen=True)
class ConvergenceConfig:
    """Convergence monitoring configuration."""
    error_target: float = ERROR_TARGET
    stall_generations: int = 30000   # Stop if no improvement for N generations
    min_generations: int = 100    # Minimum generations regardless of convergence


# --------------------------- Defaults & ranges -------------------------------
def alpha_sc_fixed_abs(isc_ref: float = ISC) -> float:
    """Return alpha_sc in absolute units A/°C, fixed at STC.

    Args:
        isc_ref: Short-circuit current at STC (A).

    Returns:
        alpha_sc in A/°C.
    """
    if ALPHA_SC_MODE.lower() == "absolute":
        return float(ALPHA_SC)
    # relative mode: convert relative coeff [1/°C] to absolute [A/°C]
    return float(REL_ALPHA_SC * isc_ref)

def default_PVModule() -> PVModuleData:
    """Return target PV module values from USER INPUT."""
    return PVModuleData(voc=VOC, isc=ISC, pmax=PMAX, vmp=VMP, imp=IMP)


def initial_parameter_ranges(nd: PVModuleData) -> dict[str, Tuple[float, float]]:
    """Define physically sensible parameter bounds.

    Args:
        nd: PV module data.

    Returns:
        Dict mapping parameter name -> (min, max).
    """
    r_sh_default = nd.vmp / (0.2 * (nd.isc - nd.imp))
    r_s_default = nd.voc / nd.isc
    return {
        "a_ref": (0.1, 5.0),                               #   diode ideality factor*N_s normalized
        "I_L_ref": (1 * nd.isc, 1.1 * nd.isc),         # A Photocurrent
        "I_o_ref": (1e-16, 1e-9),                          # A Saturation current
        "R_s": (0.0125 * r_s_default, 0.125*r_s_default),  # Ω Series resistance
        "R_sh": (2 * r_sh_default, 50 * r_sh_default),    # Ω Shunt resistance
    }


# --------------------------- GA operators -----------------------------------

def safe_mutate(individual, indpb, eta, low, up):
    """Polynomial mutation with bounds clipping."""
    for i, x in enumerate(individual):
        if random.random() < indpb:
            xl, xu = low[i], up[i]
            delta1 = (x - xl) / (xu - xl)
            delta2 = (xu - x) / (xu - xl)
            r = random.random()
            mut_pow = 1.0 / (eta + 1.0)
            if r < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * r + (1.0 - 2.0 * r) * (xy ** (eta + 1.0))
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy ** (eta + 1.0))
                delta_q = 1.0 - (val ** mut_pow)
            x = x + delta_q * (xu - xl)
            individual[i] = float(np.clip(np.real(x), xl, xu))
    return (individual,)


def safe_mate(ind1, ind2, alpha):
    """Swap-based mating that avoids invalid values."""
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.random() < alpha:
            ind1[i], ind2[i] = float(np.real(ind2[i])), float(np.real(ind1[i]))
    return ind1, ind2


_EPS = 1e-12
def _didv_sdm_at_v(v: float,
                   i_at_v: float,
                   sat_current: float,
                   rs: float,
                   rsh: float,
                   nnsvth: float) -> float:
    """Compute dI/dV at a given voltage using implicit differentiation of SDM.

    Args:
        v: Voltage at which derivative is evaluated (e.g., Vmp).
        i_at_v: Current I(v) computed from the SDM at the same v.
        sat_current: I0 (saturation current) from calcparams_cec.
        rs: Series resistance (Ω).
        rsh: Shunt resistance (Ω).
        nnsvth: a = nNsVth (V).

    Returns:
        dI/dV at voltage v.

    Notes:
        SDM implicit form F(I,V)=0 yields:
            dI/dV = - (∂F/∂V) / (∂F/∂I)
        with
            ∂F/∂V = (I0/a)*exp((V + I*Rs)/a) + 1/Rsh
            ∂F/∂I = 1 + (I0*Rs/a)*exp((V + I*Rs)/a) + Rs/Rsh
    """
    a = float(nnsvth)
    i0 = float(sat_current)
    exp_arg = (v + i_at_v * rs) / max(a, _EPS)
    # Safe exponent
    exp_term = float(np.exp(np.clip(exp_arg, -100.0, 100.0)))

    dFdV = (i0 / max(a, _EPS)) * exp_term + (0.0 if rsh == np.inf else 1.0 / max(rsh, _EPS))
    dFdI = 1.0 + (i0 * rs / max(a, _EPS)) * exp_term + (0.0 if rsh == np.inf else rs / max(rsh, _EPS))

    denom = dFdI if abs(dFdI) > _EPS else np.sign(dFdI) * _EPS or _EPS
    return - dFdV / denom

# --------------------------- Fitness function --------------------------------

def evaluate_sdm(individual: Sequence[float], nd: PVModuleData, stc: STC) -> tuple[float]:
    """Compute the GA objective: sum of relative errors on {Voc, Isc, Pmp}."""
    try:
        a_ref, I_L_ref, I_o_ref, R_s, R_sh = individual
        if R_sh <= 0 or R_s <= 0 or I_L_ref <= 0 or I_o_ref <= 0 or a_ref <= 0:
            return (float("inf"),)

        alpha_sc_abs = alpha_sc_fixed_abs(ISC)

        phot, sat, rs, rsh, nnsvth = calcparams_cec(
            stc.effective_irradiance, stc.cell_temperature_c,
            alpha_sc_abs, a_ref, I_L_ref, I_o_ref, R_sh, R_s, stc.EgRef, stc.dEgdT
        )
        res = singlediode(phot, sat, rs, rsh, nnsvth, method="lambertw")

        # Compute individual relative errors
        e_voc = abs(float(res["v_oc"]) - nd.voc) / nd.voc
        e_isc = abs(float(res["i_sc"]) - nd.isc) / nd.isc
        e_imp = abs(float(res["i_mp"]) - nd.imp) / nd.imp
        e_vmp = abs(float(res["v_mp"]) - nd.vmp) / nd.vmp
        e_pmp = abs(float(res["p_mp"]) - nd.pmax) / nd.pmax

        # Stationarity calculation
        i_at_vmp = float(pvsystem.i_from_v(
            voltage=np.array([nd.vmp], dtype=float),
            photocurrent=phot, saturation_current=sat,
            resistance_series=rs, resistance_shunt=rsh, nNsVth=nnsvth,
            method="lambertw"
        )[0])

        didv_at_vmp = _didv_sdm_at_v(
            v=float(nd.vmp),
            i_at_v=i_at_vmp,
            sat_current=float(sat),
            rs=float(rs),
            rsh=float(rsh),
            nnsvth=float(nnsvth),)
        stat_residual = abs(i_at_vmp + nd.vmp * didv_at_vmp) / max(abs(nd.imp), _EPS)

        # Composite objective
        total_error = 5*e_voc + e_isc + 1*e_vmp + 15*e_imp + 0.3*stat_residual
        #total_error = 5*e_voc + e_isc + 5*e_pmp

        # Store individual errors as an attribute of the individual
        individual.individual_errors = {
            'e_voc': e_voc,
            'e_isc': e_isc,
            'e_vmp': e_vmp,
            'e_imp': e_imp,
            'e_pmp': e_pmp,
            'stat_residual': stat_residual
        }

        return (float(total_error),)
    except Exception:
        return (float("inf"),)

# Note: res["i_mp"] is the current at the maximum power point as found by pvlib.singlediode,
# i.e., at the simulated (Vmp_sim, Imp_sim) where V*I(V) is maximised.
# i_at_vmp is instead the current evaluated at the datasheet Vmp value, i.e.:
# i_at_Vmp = I(V=Vmp_datasheet), where Vmp is the manufacturer-specified value.

# --------------------------- Plot helpers ------------------------------------

def _enable_interactive() -> bool:
    """Enable interactive plotting; returns False if backend is headless."""
    if not hasattr(plt, 'ion'):
        return False
    try:
        plt.ion()
        return True
    except (ImportError, RuntimeError):
        logging.info("Interactive plotting not available; proceeding headless.")
        return False


def _init_graph1_live(nd: PVModuleData):
    """Create Graph 1 (live adaptation) and return (fig, ax, line)."""
    fig, ax = plt.subplots()
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Graph 1 — Live Adaptation: I–V curve vs PV Module Points")
    ax.grid(True)
    # PV module anchors
    ax.plot([0.0, nd.vmp, nd.voc], [nd.isc, nd.imp, 0.0],
            marker="o", linestyle="", label="PV module key points")
    (line,) = ax.plot([], [], "r-", lw=1.5, label="Current best fit (SDM)")
    ax.legend()
    return fig, ax, line


def _update_graph1_live(ax, line, individual, stc: STC) -> None:
    """Update Graph 1 with the curve from the current best individual."""
    try:
        a_ref, I_L_ref, I_o_ref, R_s, R_sh = individual
        alpha_sc = alpha_sc_fixed_abs(ISC)
        phot, sat, rs, rsh, nnsvth = calcparams_cec(
            stc.effective_irradiance, stc.cell_temperature_c,
            alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh, R_s, stc.EgRef, stc.dEgdT
        )
        res = singlediode(phot, sat, rs, rsh, nnsvth, method="lambertw")
        v_sim = np.linspace(0.0, float(res["v_oc"]), 160)
        i_sim = pvsystem.i_from_v(
            voltage=v_sim, photocurrent=phot, saturation_current=sat,
            resistance_series=rs, resistance_shunt=rsh, nNsVth=nnsvth, method="lambertw"
        )
        line.set_data(v_sim, i_sim)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)
    except Exception as exc:
        logging.debug("Live plot update failed: %s", exc)


def _final_plot(v: np.ndarray, i: np.ndarray, nd: PVModuleData, final_error: float, module_name: str):
    """Create Graph 2 (final I–V plot) and return the figure handle.

    Args:
        v: Voltage vector.
        i: Current vector.
        nd: PV module nameplate data.
        final_error: Final objective value.
        module_name: Name of the PV module.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot([0.0, nd.vmp, nd.voc], [nd.isc, nd.imp, 0.0],
            marker="o", linestyle="", label="PV module key points")
    ax.plot(v, i, "r-", label="Single-diode fit (extracted parameters)")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    ax.set_title(f"Graph 2 — {module_name} I–V Curve at STC (Fitted SDM)")
    ax.annotate(f"Final error: {final_error:.2e}",
                xy=(0.02, 0.05), xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
                fontsize=11)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# --------------------------- Utilities --------------------------------------

def _format_mm_ss(seconds: float) -> str:
    """Return a mm:ss string from seconds."""
    seconds = int(round(seconds))
    mm, ss = divmod(seconds, 60)
    return f"{mm:02d}:{ss:02d}"


def _setup_deap_types():
    """Setup DEAP types in a robust way."""
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)


def save_summary_to_file(summary_string: str, module_name: str) -> None:
    """Save the summary string to a text file.

    Args:
        summary_string: Formatted summary string (exactly what's printed to console).
        module_name: Name of the PV module.
    """
    # Create filename by replacing spaces with underscores and adding .txt extension
    filename = f"{module_name.replace(' ', '_')}_CEC.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(summary_string)

    logging.info("Summary saved to: %s", filename)


# --------------------------- Main fitting pipeline --------------------------

def fit_parameters(nd: PVModuleData, stc: STC, ga: GAConfig, module_name: str,
                   seed: int | None = None, live_plot: bool = True) -> tuple[dict, str]:
    """Run GA to estimate SDM parameters and produce plots.

    Args:
        nd: PV module data at STC.
        stc: STC constants.
        ga: GA configuration.
        module_name: Name of the PV module.
        seed: Optional RNG seed.
        live_plot: If True, show Graph 1 (live adaptation).

    Returns:
        Tuple with (results_dict, summary_string).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    param_ranges = initial_parameter_ranges(nd)
    param_order = ["a_ref", "I_L_ref", "I_o_ref", "R_s", "R_sh"]

    # DEAP setup (robust)
    _setup_deap_types()

    toolbox = base.Toolbox()
    for name in param_order:
        lo, hi = param_ranges[name]
        toolbox.register(f"attr_{name}", random.uniform, lo, hi)

    toolbox.register("individual", tools.initCycle,
                     creator.Individual,
                     tuple(getattr(toolbox, f"attr_{n}") for n in param_order), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    low = [param_ranges[k][0] for k in param_order]
    up = [param_ranges[k][1] for k in param_order]

    toolbox.register("mate", safe_mate, alpha=0.5)
    toolbox.register("mutate", safe_mutate, indpb=ga.indpb_mut, eta=ga.eta_mut, low=low, up=up)
    toolbox.register("select", tools.selTournament, tournsize=ga.tournament_size)
    toolbox.register("evaluate", evaluate_sdm, nd=nd, stc=stc)

    # Multiprocessing map with context manager
    nproc = mp.cpu_count()
    pool = None
    try:
        # Use context manager for proper pool handling
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=nproc)
        toolbox.register("map", pool.map)
    except Exception as exc:
        logging.warning("Falling back to serial evaluation: %s", exc)
        toolbox.register("map", map)
        nproc = 1

    # Graph 1 — live adaptation
    interactive = False
    fig1 = ax1 = line1 = None
    if live_plot:
        interactive = _enable_interactive()
        if interactive:
            fig1, ax1, line1 = _init_graph1_live(nd)

    # Initial population + evaluation
    pop = toolbox.population(n=ga.population_size)
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses, strict=False):
        ind.fitness.values = fit

    # Evolution loop
    start_t = time.time()
    start_dt = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    early_stop = False
    last_update = 0
    best_error_history = []
    convergence = ConvergenceConfig()

    for gen in tqdm(range(1, ga.num_generations + 1), desc="Evolution", unit="gen"):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover & mutation

        for c1, c2 in zip(offspring[::2], offspring[1::2], strict=False):
            if random.random() < ga.cx_prob:
                toolbox.mate(c1, c2)
                if hasattr(c1.fitness, "values"):
                    del c1.fitness.values
                if hasattr(c2.fitness, "values"):
                    del c2.fitness.values
        for ind in offspring:
            if random.random() < ga.mut_prob:
                toolbox.mutate(ind)
                if hasattr(ind.fitness, "values"):
                    del ind.fitness.values
        # Evaluate invalid
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid)
        for ind, f in zip(invalid, fits, strict=False):
            ind.fitness.values = f

        # Elitism (optional but preserved)
        pop[:] = offspring

        # Track best error for convergence monitoring
        best_now = tools.selBest(pop, k=1)[0]
        curr_err = best_now.fitness.values[0]
        best_error_history.append(curr_err)

        # Progress logging
        if gen % 100 == 0:
            logging.info("Generation %d - Best error: %.3e", gen, curr_err)

        # Live updates
        if interactive and (gen - last_update >= ga.progress_update_period or gen == 1):
            _update_graph1_live(ax1, line1, best_now, stc)  # Graph 1 update
            last_update = gen

        # Convergence check: stop if all individual errors are below the target
        best_errors = getattr(best_now, "individual_errors", {})
        if best_errors:
            if all(v <= ga.error_target for v in best_errors.values()):
                early_stop = True
                logging.info("Early stop: all individual errors below threshold (%.2e)", ga.error_target)
                break

        # Stall detection
        # Genetic restart: reseed part of population if stagnation persists
        if gen > 100 and gen % 200 == 0:
            recent_errors = best_error_history[-200:] if len(best_error_history) > 200 else best_error_history
            improvement = np.abs(np.min(recent_errors) - curr_err)
            if improvement < 1e-6:
                reseed_fraction = 0.2  # 20% of population replaced
                n_reseed = int(reseed_fraction * len(pop))
                logging.info("Stall detected (ΔE < 1e-6) — reseeding %.0f%% of population", reseed_fraction * 100)
                for i in range(n_reseed):
                    pop[i] = toolbox.individual()
                    pop[i].fitness.values = toolbox.evaluate(pop[i])

        if gen > convergence.min_generations:
            recent_best = min(best_error_history[-convergence.stall_generations:])
            if abs(recent_best - curr_err) < 1e-8:  # No improvement
                early_stop = True
                logging.info("Early stop: convergence stalled for %d generations",
                            convergence.stall_generations)
                break

    runtime = time.time() - start_t
    end_dt = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Pool teardown using context manager
    if pool is not None:
        pool.close()
        pool.join()

    # Best individual and final curve
    best = tools.selBest(pop, k=1)[0]
    final_error = float(evaluate_sdm(best, nd, stc)[0])

    # Store individual errors of the best solution
    individual_errors = getattr(best, 'individual_errors', {})

    a_ref, I_L_ref, I_o_ref, R_s, R_sh = best
    alpha_fixed = alpha_sc_fixed_abs(ISC)
    phot, sat, rs, rsh, nnsvth = calcparams_cec(
        stc.effective_irradiance, stc.cell_temperature_c,
        alpha_fixed, a_ref, I_L_ref, I_o_ref, R_sh, R_s, stc.EgRef, stc.dEgdT
    )
    res = singlediode(phot, sat, rs, rsh, nnsvth, method="lambertw")
    v = np.linspace(0.0, float(res["v_oc"]), 200)
    i = pvsystem.i_from_v(
        voltage=v, photocurrent=phot, saturation_current=sat,
        resistance_series=rs, resistance_shunt=rsh, nNsVth=nnsvth, method="lambertw"
    )

    # --- Close Graph 1 after 3 seconds; then show Graph 2 and keep it open ---
    if interactive and fig1 is not None:
        try:
            plt.pause(3.0)   # keep Graph 1 visible for ~3 s
            plt.close(fig1)  # then close only Graph 1
        except Exception:
            pass

    # Graph 2: final plot (kept open)
    fig2 = _final_plot(v, i, nd, final_error, module_name)
    try:
        plt.ioff()   # ensure the final show() blocks and Graph 2 remains open
    except Exception:
        pass
    plt.show()       # Graph 2 stays open

    # Build summary string for console output and file saving
    summary_lines = []
    summary_lines.append("=== Single-Diode Parameter Extraction Summary ===")
    summary_lines.append("=== Model: CEC ===")
    summary_lines.append(f"Start time         : {start_dt}")
    summary_lines.append(f"End time           : {end_dt}")
    summary_lines.append(f"Runtime (mm:ss)    : {_format_mm_ss(runtime)}")
    summary_lines.append(f"GA generations     : {gen} / {ga.num_generations} {'(early stop)' if early_stop else ''}")
    summary_lines.append(f"Population size    : {ga.population_size}")
    summary_lines.append(f"Crossover prob     : {ga.cx_prob:.2f}")
    summary_lines.append(f"Mutation prob      : {ga.mut_prob:.2f}")
    summary_lines.append(f"Tournament size    : {ga.tournament_size}")
    summary_lines.append(f"Workers (processes): {nproc}")
    summary_lines.append("")
    summary_lines.append("-- PV module (STC) --")
    summary_lines.append(f" Voc*  = {nd.voc:.2f} V   Isc* = {nd.isc:.2f} A   Pmp* = {nd.pmax:.2f} W")
    summary_lines.append(f" Vmp*  = {nd.vmp:.2f} V   Imp* = {nd.imp:.2f} A")
    summary_lines.append("")
    summary_lines.append("-- Fitted parameters (best individual) --")
    summary_lines.append(f" alpha_sc (fixed) = {alpha_fixed:.6g}  a_ref = {a_ref:.6g}")
    summary_lines.append(f" I_L_ref  = {I_L_ref:.6g} A  I_o_ref = {I_o_ref:.6g} A")
    summary_lines.append(f" R_s      = {R_s:.6g} Ω  R_sh = {R_sh:.6g} Ω")
    summary_lines.append(f" nNsVth   = {nnsvth:.6g} V")
    summary_lines.append("")
    summary_lines.append("-- SDM simulation vs PV module --")
    summary_lines.append(f" Voc(sim) = {float(res['v_oc']):.3f} V  |  Voc* = {nd.voc:.3f} V")
    summary_lines.append(f" Isc(sim) = {float(res['i_sc']):.3f} A  |  Isc* = {nd.isc:.3f} A")
    summary_lines.append(f" Vmp(sim) = {float(res['v_mp']):.3f} V  |  Vmp* = {nd.vmp:.3f} V")
    summary_lines.append(f" Imp(sim) = {float(res['i_mp']):.3f} A  |  Imp* = {nd.imp:.3f} A")
    summary_lines.append(f" Pmp(sim) = {float(res['p_mp']):.3f} W  |  Pmp* = {nd.pmax:.3f} W")
    summary_lines.append("")
    summary_lines.append(f"Total relative error (objective E): {final_error:.3e}")
    summary_lines.append("===============================================")

    summary_string = "\n".join(summary_lines)

    # Print summary to console
    print(summary_string)

    results_dict = {
        "best_individual": {
            "alpha_sc_fixed": alpha_fixed, "a_ref": a_ref, "I_L_ref": I_L_ref,
            "I_o_ref": I_o_ref, "R_s": R_s, "R_sh": R_sh, "nNsVth": nnsvth
        },
        "final_error": final_error,
        "result_sdm": {
            "v_oc": float(res["v_oc"]),
            "i_sc": float(res["i_sc"]),
            "v_mp": float(res["v_mp"]),
            "i_mp": float(res["i_mp"]),
            "p_mp": float(res["p_mp"]),
        },
        "individual_errors": individual_errors,
        "runtime_seconds": runtime,
        "generations_done": gen,
        "early_stop": early_stop,
        "n_processes": nproc,
    }

    return results_dict, summary_string

# --------------------------- Entry point ------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    p = argparse.ArgumentParser(
        prog="pvfit5",
        description=(
            "Estimate the five single-diode model parameters of a PV module "
            "from datasheet values at STC."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  pvfit5 --voc 36.3 --isc 8.19 --pmax 218.95 --vmp 29.0 --imp 7.55\n"
            "  pvfit5 --voc 36.3 --isc 8.19 --pmax 218.95 --vmp 29.0 --imp 7.55 "
            "--alpha-sc 0.05 --name MyModule"
        ),
    )
    # Required PV module parameters
    p.add_argument("--voc", type=float, default=None,
                   help="Open-circuit voltage at STC (V)")
    p.add_argument("--isc", type=float, default=None,
                   help="Short-circuit current at STC (A)")
    p.add_argument("--pmax", type=float, default=None,
                   help="Maximum power at STC (W)")
    p.add_argument("--vmp", type=float, default=None,
                   help="Voltage at maximum power point (V)")
    p.add_argument("--imp", type=float, default=None,
                   help="Current at maximum power point (A)")
    # Optional parameters
    p.add_argument("--name", type=str, default=None,
                   help="Module name (used for output files, default: PVModule)")
    p.add_argument("--error-target", type=float, default=None,
                   help="GA early-stop threshold (default: 1e-2)")
    # Temperature coefficient
    p.add_argument("--alpha-sc", type=float, default=None,
                   help="Short-circuit current temperature coefficient in A/°C "
                        "(absolute mode, default: 0.05)")
    p.add_argument("--alpha-sc-rel", type=float, default=None,
                   help="Short-circuit current temperature coefficient in 1/°C "
                        "(relative mode, e.g. 0.005). Overrides --alpha-sc")
    # Material parameters
    p.add_argument("--egref", type=float, default=None,
                   help="Band gap energy at reference conditions in eV "
                        "(default: 1.121 for crystalline Si)")
    p.add_argument("--degdt", type=float, default=None,
                   help="Temperature coefficient of band gap in eV/K "
                        "(default: -0.000267 for crystalline Si)")
    # Flags
    p.add_argument("--no-plot", action="store_true",
                   help="Disable live and final plots")
    return p


def main() -> None:
    """Main entry point — supports both CLI arguments and USER INPUT defaults."""
    global MODULE_NAME, VOC, ISC, PMAX, VMP, IMP, ERROR_TARGET
    global ALPHA_SC_MODE, ALPHA_SC, REL_ALPHA_SC

    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    # Determine if any datasheet argument was given on the CLI
    cli_datasheet = any(v is not None for v in [args.voc, args.isc, args.pmax, args.vmp, args.imp])

    if cli_datasheet:
        # When using CLI, all five datasheet values are required
        missing = []
        for name in ("voc", "isc", "pmax", "vmp", "imp"):
            if getattr(args, name) is None:
                missing.append(f"--{name}")
        if missing:
            parser.error(f"When using CLI arguments, all five datasheet values "
                         f"are required. Missing: {', '.join(missing)}")

        VOC = args.voc
        ISC = args.isc
        PMAX = args.pmax
        VMP = args.vmp
        IMP = args.imp
        MODULE_NAME = args.name if args.name else "PVModule"
    else:
        # Use the module-level USER INPUT defaults
        if args.name:
            MODULE_NAME = args.name

    # Override optional parameters if provided
    if args.error_target is not None:
        ERROR_TARGET = args.error_target

    if args.alpha_sc_rel is not None:
        ALPHA_SC_MODE = "relative"
        REL_ALPHA_SC = args.alpha_sc_rel
    elif args.alpha_sc is not None:
        ALPHA_SC_MODE = "absolute"
        ALPHA_SC = args.alpha_sc

    # Build data objects
    nd = PVModuleData(voc=VOC, isc=ISC, pmax=PMAX, vmp=VMP, imp=IMP)
    stc_kwargs = {}
    if args.egref is not None:
        stc_kwargs["EgRef"] = args.egref
    if args.degdt is not None:
        stc_kwargs["dEgdT"] = args.degdt
    stc = STC(**stc_kwargs)
    ga = GAConfig(error_target=ERROR_TARGET)

    logging.info("Target PV module: %s", MODULE_NAME)
    logging.info("PV parameters: Voc=%.2f V, Isc=%.2f A, Pmp=%.2f W, Vmp=%.2f V, Imp=%.2f A",
                 nd.voc, nd.isc, nd.pmax, nd.vmp, nd.imp)
    logging.info("STC: EgRef=%.4f eV, dEgdT=%.6f eV/K", stc.EgRef, stc.dEgdT)
    logging.info("Alpha_sc mode: %s", ALPHA_SC_MODE)

    results, summary_string = fit_parameters(
        nd, stc, ga, MODULE_NAME, seed=None, live_plot=not args.no_plot
    )

    # Save summary to file
    save_summary_to_file(summary_string, MODULE_NAME)

    logging.info("Final error: %.2e | Runtime: %s (mm:ss)",
                 results["final_error"], _format_mm_ss(results["runtime_seconds"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        sys.exit(1)

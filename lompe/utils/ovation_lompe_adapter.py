"""OvationPyme conductance adapter for use as Lompe callables.

This module intentionally lives outside Lompe.  It reads local OMNI 1-minute
CDF files for the Newell coupling input, calls OvationPyme's
``ConductanceEstimator``, and interpolates its Hall and Pedersen conductances
to requested geographic coordinates.
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import apexpy
import cdflib
import numpy as np
import pandas as pd
from .download_solarwind_and_indices import get_f107

from ovationpyme import ovation_utilities
from ovationpyme.ovation_prime import (
    ConductanceEstimator,
    BinCorrector,
    FluxEstimator,
    LatLocaltimeInterpolator,
)


DEFAULT_OMNI_CDF_DIR = Path(__file__).resolve().parent.parent / 'data' / 'omni'
ELECTRON_AURORA_TYPES = ('diff', 'mono', 'wave')


def _newell_coupling(bx: np.ndarray, by: np.ndarray, bz: np.ndarray,
                     speed: np.ndarray) -> np.ndarray:
    """Calculate the Newell coupling function used by OvationPyme."""
    bt = np.sqrt(by**2 + bz**2)
    nonzero_bz = np.where(bz == 0, 0.001, bz)
    clock_angle = np.arctan2(by, nonzero_bz)
    clock_angle[bt * np.cos(clock_angle) * bz < 0] += np.pi
    return speed**1.33333 * np.abs(np.sin(clock_angle / 2.0))**2.66667 * bt**0.66667


@lru_cache(maxsize=12)
def _read_omni_month(cdf_path: str) -> pd.DataFrame:
    """Load the OMNI quantities needed for Newell coupling from one CDF."""
    cdf = cdflib.CDF(cdf_path)
    frame = pd.DataFrame(
        {
            'bx': np.asarray(cdf.varget('BX_GSE'), dtype=float),
            'by': np.asarray(cdf.varget('BY_GSM'), dtype=float),
            'bz': np.asarray(cdf.varget('BZ_GSM'), dtype=float),
            'speed': np.asarray(cdf.varget('flow_speed'), dtype=float),
        },
        index=pd.to_datetime(cdflib.cdfepoch.to_datetime(cdf.varget('Epoch'))),
    ).sort_index()
    # OMNI CDF fill values (for example 9999.99 nT and 99999.9 km/s) are
    # finite, so remove them explicitly before calculating the coupling.
    valid = (
        frame['bx'].abs().lt(1_000)
        & frame['by'].abs().lt(1_000)
        & frame['bz'].abs().lt(1_000)
        & frame['speed'].between(100, 3_000)
    )
    frame.loc[~valid, ['bx', 'by', 'bz', 'speed']] = np.nan
    frame['newell'] = _newell_coupling(
        frame['bx'].to_numpy(), frame['by'].to_numpy(), frame['bz'].to_numpy(),
        frame['speed'].to_numpy(),
    )
    return frame


def _cdf_for_time(when: dt.datetime, omni_cdf_dir: Path) -> Path:
    pattern = f'omni_hro_1min_{when:%Y%m}01_v*.cdf'
    matches = sorted(omni_cdf_dir.expanduser().glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f'No local OMNI CDF matches {pattern} in {omni_cdf_dir}. '
            'Download that monthly OMNI 1-minute CDF into this directory first.'
        )
    return matches[-1]


def local_newell_coupling(when: dt.datetime, omni_cdf_dir: str | Path = DEFAULT_OMNI_CDF_DIR) -> float:
    """Return OvationPyme's four-hour, hourly-binned weighted Newell value."""
    timestamp = pd.Timestamp(when)
    frame = _read_omni_month(str(_cdf_for_time(timestamp.to_pydatetime(), Path(omni_cdf_dir))))
    hourly_means = []
    for hours_before in range(3, -1, -1):
        start = timestamp - pd.Timedelta(hours=hours_before + 1)
        end = timestamp - pd.Timedelta(hours=hours_before)
        values = frame.loc[(frame.index >= start) & (frame.index < end), 'newell']
        hourly_means.append(values.mean())
    values = np.asarray(hourly_means, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError(f'No finite OMNI solar-wind values in the four hours before {timestamp}.')
    weights = 0.65**np.arange(3, -1, -1, dtype=float)
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


@contextmanager
def _temporary_space_weather(newell: float, f107: float):
    """Supply local Newell and explicit F10.7 values to OvationPyme."""
    original_newell = ovation_utilities.calc_dF
    original_f107 = ovation_utilities.get_daily_f107
    ovation_utilities.calc_dF = lambda when: newell
    ovation_utilities.get_daily_f107 = lambda when: f107
    try:
        yield
    finally:
        ovation_utilities.calc_dF = original_newell
        ovation_utilities.get_daily_f107 = original_f107


def _geographic_to_magnetic(glat: np.ndarray, glon: np.ndarray,
                            when: dt.datetime, refh_km: float) -> tuple[np.ndarray, np.ndarray]:
    apex = apexpy.Apex(when, refh=refh_km)
    mlat, mlon = apex.geo2apex(glat, glon, refh_km)
    mlt = np.mod(apex.mlon2mlt(mlon, when), 24.0)
    return np.asarray(mlat, dtype=float), np.asarray(mlt, dtype=float)


def _proton_conductance(energy_flux: np.ndarray,
                        number_flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return proton Hall and Pedersen conductance from OVATION ion fluxes.

    The energy conversion, thresholds, and proton formulae match Lompe's
    existing ``ovation_total_conductance.py`` implementation.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_energy = (energy_flux / 1.6e-12) / number_flux / 1000.0
    mean_energy = np.nan_to_num(mean_energy, nan=0.0, posinf=0.0, neginf=0.0)
    mean_energy = np.clip(mean_energy, 0.0, 30.0)
    mean_energy[mean_energy < 0.2] = 0.0
    energy_flux = np.nan_to_num(energy_flux, nan=0.0, posinf=0.0, neginf=0.0)
    energy_flux[energy_flux < 0.0] = 0.0
    pedersen = 5.7 * np.sqrt(energy_flux)
    hall = 2.565 * mean_energy**0.3 * np.sqrt(energy_flux)
    return hall, pedersen


def _correct_ion_fluxes(mlat: np.ndarray, mlt: np.ndarray,
                        energy_flux: np.ndarray, number_flux: np.ndarray,
                        number_flux_threshold: float = 1e8,
                        mean_energy_threshold: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Correct ion inputs with OvationPyme's native-grid bin corrector.

    This mirrors the electron correction in ``ConductanceEstimator``: correct
    number flux and mean energy independently, then reconstruct a mutually
    consistent energy-flux grid for the proton conductance formulae.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_energy = (energy_flux / 1.6e-12) / number_flux / 1000.0
    mean_energy = np.nan_to_num(mean_energy, nan=0.0, posinf=0.0, neginf=0.0)
    number_flux = np.nan_to_num(number_flux, nan=0.0, posinf=0.0, neginf=0.0)
    number_flux[number_flux < 0.0] = 0.0

    corrector = BinCorrector(mlat, mlt)
    corrector.dy_thresh = number_flux_threshold
    corrected_number_flux = corrector.fix(number_flux, label='ion_nflux')
    corrector.dy_thresh = mean_energy_threshold
    corrected_mean_energy = corrector.fix(mean_energy, label='ion_eavg')

    # Match OvationPyme's treatment of the low-latitude artefact.  Rebuilding
    # energy flux from the two corrected fields preserves their relationship.
    corrected_number_flux[np.abs(mlat) < 52.0] = 0.0
    corrected_energy_flux = corrected_number_flux * corrected_mean_energy * 1.6022e-9
    return corrected_energy_flux, corrected_number_flux


def ovation_conductance(
    glat: np.ndarray | float,
    glon: np.ndarray | float,
    when: dt.datetime | pd.Timestamp,
    *,
    f107: float | None = None,
    omni_cdf_dir: str | Path = DEFAULT_OMNI_CDF_DIR,
    aurora_types: tuple[str, ...] = ('diff', 'mono', 'wave'),
    include_ions: bool = True,
    solar: bool = True,
    interp_bad_bins: bool = True,
    refh_km: float = 110.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Hall then Pedersen conductance for geographic locations.

    If ``f107`` is omitted, it is read from the local GFZ history file
    downloaded by ``download_solarwind_and_indices.py``.  The default aurora
    types include diffuse, monoenergetic, and wave electrons by default.
    Ions are included by default through the proton formulae used by Lompe's
    existing Ovation helper; set ``include_ions=False`` for electron-only
    comparisons. ``interp_bad_bins`` corrects anomalous electron and ion
    source-grid bins before the conductance calculation.
    """
    invalid_types = set(aurora_types).difference(ELECTRON_AURORA_TYPES)
    if invalid_types:
        raise ValueError(f'Only electron aurora types are supported: {ELECTRON_AURORA_TYPES}.')
    if not aurora_types:
        raise ValueError('Select at least one aurora type.')

    timestamp = pd.Timestamp(when).to_pydatetime()
    if f107 is None:
        f107 = get_f107(timestamp)
    glat_arr, glon_arr = np.broadcast_arrays(
        np.asarray(glat, dtype=float), np.asarray(glon, dtype=float))
    shape = glat_arr.shape
    mlat, mlt = _geographic_to_magnetic(glat_arr.ravel(), glon_arr.ravel(), timestamp, refh_km)
    hall = np.zeros(mlat.size, dtype=float)
    pedersen = np.zeros(mlat.size, dtype=float)
    newell = local_newell_coupling(timestamp, omni_cdf_dir)

    estimator = ConductanceEstimator(fluxtypes=aurora_types)
    ion_energy_estimator = FluxEstimator('ions', 'energy') if include_ions else None
    ion_number_estimator = FluxEstimator('ions', 'number') if include_ions else None
    kwargs = {
        'solar': solar,
        'background_p': 0.1,
        'background_h': 0.1,
        'conductance_fluxtypes': aurora_types,
        'interp_bad_bins': interp_bad_bins,
        'return_dF': True,
        'return_f107': True,
        'dnflux_bad_thresh': 1e8,
        'deavg_bad_thresh': 0.3,
    }
    with _temporary_space_weather(newell, float(f107)):
        for hemisphere, mask in (('N', mlat >= 0), ('S', mlat < 0)):
            if not mask.any():
                continue
            model_mlat, model_mlt, model_ped, model_hall, _, _ = estimator.get_conductance(
                timestamp, hemi=hemisphere, **kwargs)
            pedersen[mask] = LatLocaltimeInterpolator(
                model_mlat, model_mlt, model_ped).interpolate(mlat[mask], mlt[mask], method='linear')
            hall[mask] = LatLocaltimeInterpolator(
                model_mlat, model_mlt, model_hall).interpolate(mlat[mask], mlt[mask], method='linear')
            if include_ions:
                ion_mlat, ion_mlt, ion_energy_flux = ion_energy_estimator.get_flux_for_time(
                    timestamp, hemi=hemisphere
                )
                _, _, ion_number_flux = ion_number_estimator.get_flux_for_time(
                    timestamp, hemi=hemisphere
                )
                if interp_bad_bins:
                    ion_energy_flux, ion_number_flux = _correct_ion_fluxes(
                        ion_mlat, ion_mlt,
                        np.asarray(ion_energy_flux, dtype=float),
                        np.asarray(ion_number_flux, dtype=float),
                    )
                ion_hall, ion_pedersen = _proton_conductance(
                    np.asarray(ion_energy_flux, dtype=float),
                    np.asarray(ion_number_flux, dtype=float),
                )
                # Retain Lompe's convention: add the proton contribution
                # linearly after the OvationPyme electron/solar result.
                hall[mask] += LatLocaltimeInterpolator(
                    ion_mlat, ion_mlt, ion_hall).interpolate(mlat[mask], mlt[mask], method='linear')
                pedersen[mask] += LatLocaltimeInterpolator(
                    ion_mlat, ion_mlt, ion_pedersen).interpolate(mlat[mask], mlt[mask], method='linear')

    return hall.reshape(shape), pedersen.reshape(shape)

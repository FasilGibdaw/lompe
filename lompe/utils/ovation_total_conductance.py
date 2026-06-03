from __future__ import annotations

import logging
import os
import sys
import types
import warnings
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path

import apexpy
import cdflib
import numpy as np
import pandas as pd
import requests
import lompe
from lompe.utils import sunlight

BASE_DIR = Path('.').resolve()
DEFAULT_OMNI_DOWNLOAD_DIR = Path('./sample_dataset/omni_tempfiles')
DEFAULT_AURORA_TYPES = ('diff', 'mono', 'wave')
DEFAULT_REFH_KM = 110.0

""" 

This module provides a function to calculate the total Pedersen and Hall conductance at a given geographic location and time using the OVATION Prime model, with input from OMNI solar wind data. The main function is `get_total_ovation_conductance`, which takes geographic latitude and longitude, a timestamp, and optional parameters for the OMNI CDF source, OVATIONpyme path, aurora types to include, whether to combine hemispheres, and the reference height for apex coordinates. 

Most help from Codex, but some manual adjustments to ensure proper handling of edge cases and to provide a fallback for missing dependencies."""


def get_total_ovation_conductance_EUV(
    glat,
    glon,
    ts,
    hallOrPed: str = 'hp',
    starlight: float = 0,
    F107: float = 100,
    calibration: str = 'MoenBrekke1993',
):
    glat = np.array(glat, ndmin=1)
    glon = np.array(glon, ndmin=1)
    shape = np.broadcast(glat, glon).shape
    glat = glat.flatten()
    glon = glon.flatten()
    sza = sunlight.sza(glat, glon, ts)
    """takes the total conductance (electrons and protons) from OVATION prime model (get_total_ovation_conductance function) and add EUV based on the solar zenith angle and F107

    Returns:
    hall_coductance and pedersen_conductance if hallOrPed is 'hp', otherwise returns the requested conductance type as a single array
    """

    hop = hallOrPed.lower()
    if hop in ('hall', 'h'):
        hop = 'h'
    elif hop in ('pedersen', 'p'):
        hop = 'p'
    elif hop in ('hp', 'hallandpedersen'):
        hop = 'hp'
    else:
        raise ValueError("hallOrPed must be one of 'h', 'p', or 'hp'")

    if hop == 'hp':
        EUVh, EUVp = lompe.conductance.EUV_conductance(
            sza, F107=F107, hallOrPed='hp', calibration=calibration
        )
        hc_hall, hc_pedersen = get_total_ovation_conductance(
            glat, glon, ts, hallOrPed='hp')
        return (
            np.sqrt(hc_hall**2 + EUVh**2 + starlight**2).reshape(shape),
            np.sqrt(hc_pedersen**2 + EUVp**2 + starlight**2).reshape(shape),
        )

    EUV = lompe.conductance.EUV_conductance(
        sza, F107=F107, hallOrPed=hop, calibration=calibration
    )
    hc = get_total_ovation_conductance(glat, glon, ts, hallOrPed=hop)
    return np.sqrt(hc**2 + EUV**2 + starlight**2).reshape(shape)


def download_omni_file(url: str, filename: str, basepath: str = DEFAULT_OMNI_DOWNLOAD_DIR) -> None:
    base = Path(basepath).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    target = base / filename
    if target.exists():
        print(f'{filename} already exists.')
        return None
    response = requests.get(url, allow_redirects=True, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)
    return None


def _import_ovationpyme(ovationpyme_path: str = ''):
    if ovationpyme_path:
        root = Path(ovationpyme_path).expanduser().resolve()
        search_roots = [root]
        if (root / 'ovationpyme').exists():
            search_roots.insert(0, root.parent)
        for candidate in search_roots:
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

    runtime_dir = BASE_DIR / '_runtime'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('SPACEPY', str(runtime_dir))

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='pkg_resources is deprecated as an API.*')
            import pkg_resources
        import appdirs
        if not hasattr(pkg_resources, 'appdirs'):
            pkg_resources.appdirs = appdirs
    except Exception:
        pass

    logbook_missing = False
    if 'logbook' not in sys.modules:
        try:
            logbook_missing = find_spec('logbook') is None
        except ValueError:
            logbook_missing = True

    if logbook_missing and 'logbook' not in sys.modules:
        class _CompatLogger:
            def __init__(self, name: str):
                self._logger = logging.getLogger(name)

            def debug(self, *args, **kwargs):
                self._logger.debug(*args, **kwargs)

            def info(self, *args, **kwargs):
                self._logger.info(*args, **kwargs)

            def notice(self, *args, **kwargs):
                self._logger.info(*args, **kwargs)

            def warning(self, *args, **kwargs):
                self._logger.warning(*args, **kwargs)

            warn = warning

            def error(self, *args, **kwargs):
                self._logger.error(*args, **kwargs)

        shim = types.ModuleType('logbook')
        shim.Logger = _CompatLogger
        sys.modules['logbook'] = shim

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='pkg_resources is deprecated as an API.*')
        from ovationpyme.ovation_prime import FluxEstimator, LatLocaltimeInterpolator
        from ovationpyme import ovation_utilities
    return FluxEstimator, LatLocaltimeInterpolator, ovation_utilities


def _electron_pedersen(emean: np.ndarray, eflux: np.ndarray) -> np.ndarray:
    return ((40.0 * emean) / (16.0 + emean**2)) * np.sqrt(eflux)


def _electron_hall(emean: np.ndarray, eflux: np.ndarray) -> np.ndarray:
    pedersen_conductance = _electron_pedersen(emean, eflux)
    return 0.45 * (emean**0.85) * pedersen_conductance


def _proton_pedersen(eflux: np.ndarray) -> np.ndarray:
    return 5.7 * np.sqrt(eflux)


def _proton_hall(emean: np.ndarray, eflux: np.ndarray) -> np.ndarray:
    return 2.565 * (emean**0.3) * np.sqrt(eflux)


def _calc_newell_coupling(bx: np.ndarray, by: np.ndarray, bz: np.ndarray, v: np.ndarray) -> np.ndarray:
    bt = np.sqrt(by**2 + bz**2)
    bztemp = np.asarray(bz, dtype=float).copy()
    bztemp[bztemp == 0] = 0.001
    tc = np.arctan2(by, bztemp)
    neg_tc = bt * np.cos(tc) * bz < 0
    tc[neg_tc] = tc[neg_tc] + np.pi
    sintc = np.abs(np.sin(tc / 2.0))
    return (v ** 1.33333) * (sintc ** 2.66667) * (bt ** 0.66667)


def _cdf_epoch_to_datetime(epoch_values: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(cdflib.cdfepoch.to_datetime(epoch_values))


def _monthly_omni_filename(when: datetime | pd.Timestamp) -> str:
    ts = pd.Timestamp(when)
    return f'omni_hro_1min_{ts.year:04d}{ts.month:02d}01_v01.cdf'


def _monthly_omni_url(when: datetime | pd.Timestamp) -> str:
    ts = pd.Timestamp(when)
    filename = _monthly_omni_filename(ts)
    return f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/omni/hro_1min/{ts.year:04d}/{filename}'


def _ensure_omni_cdf_available(when: datetime | pd.Timestamp, cdf_source: str | Path) -> Path:
    source = Path(cdf_source).expanduser()
    filename = _monthly_omni_filename(when)

    if source.suffix.lower() == '.cdf':
        target = source.resolve()
        if target.exists():
            return target
        download_omni_file(_monthly_omni_url(
            when), target.name, str(target.parent))
        return target

    base = source.resolve()
    base.mkdir(parents=True, exist_ok=True)
    target = base / filename
    if not target.exists():
        download_omni_file(_monthly_omni_url(when), filename, str(base))
    return target


def _load_omni_cdf_getter(cdf_source: str | Path):
    base = Path(cdf_source).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f'OMNI CDF source not found: {base}')
    if base.is_file() and base.suffix.lower() == '.cdf':
        files = [base]
    else:
        files = sorted(base.glob('*.cdf'))
    if not files:
        raise FileNotFoundError(f'No OMNI CDF files found in {base}')

    frames: list[pd.DataFrame] = []
    for path in files:
        cdf = cdflib.CDF(str(path))
        frame = pd.DataFrame(
            {
                'BX_GSE': np.asarray(cdf.varget('BX_GSE'), dtype=float),
                'BY_GSM': np.asarray(cdf.varget('BY_GSM'), dtype=float),
                'BZ_GSM': np.asarray(cdf.varget('BZ_GSM'), dtype=float),
                'flow_speed': np.asarray(cdf.varget('flow_speed'), dtype=float),
            },
            index=_cdf_epoch_to_datetime(cdf.varget('Epoch')),
        )
        frames.append(frame)

    omni = pd.concat(frames).sort_index()
    omni = omni[~omni.index.duplicated(keep='first')]
    omni['Ec'] = _calc_newell_coupling(
        omni['BX_GSE'].values,
        omni['BY_GSM'].values,
        omni['BZ_GSM'].values,
        omni['flow_speed'].values,
    )

    cache: dict[pd.Timestamp, float] = {}

    def get_df(timestamp: pd.Timestamp) -> float:
        ts = pd.Timestamp(timestamp)
        if ts in cache:
            return cache[ts]
        window = omni.loc[ts - pd.Timedelta(hours=4): ts]
        if window.empty:
            raise KeyError(f'No local OMNI CDF data available near {ts}')
        weights = 0.65 ** np.arange(len(window) - 1, -1, -1, dtype=float)
        ec = window['Ec'].to_numpy(dtype=float)
        valid = np.isfinite(ec)
        if not np.any(valid):
            raise ValueError(
                f'No finite OMNI-derived Newell coupling values near {ts}')
        value = float(np.sum(ec[valid] * weights[valid]
                             ) / np.sum(weights[valid]))
        cache[ts] = value
        return value

    return get_df


def geographic_to_ovation_coords(
    glat: np.ndarray | float,
    glon: np.ndarray | float,
    when: datetime | pd.Timestamp,
    refh_km: float = DEFAULT_REFH_KM,
) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.Timestamp(when).to_pydatetime()
    glat_arr = np.asarray(glat, dtype=float)
    glon_arr = np.asarray(glon, dtype=float)
    apex = apexpy.Apex(ts, refh=refh_km)
    mlat, mlon = apex.geo2apex(glat_arr, glon_arr, refh_km)
    mlt = np.mod(apex.mlon2mlt(mlon, ts), 24.0)
    return np.asarray(mlat, dtype=float), np.asarray(mlt, dtype=float)


def get_total_ovation_conductance(
    glat: np.ndarray | float,
    glon: np.ndarray | float,
    when: datetime | pd.Timestamp,
    omni_cdf_dir: str | Path = DEFAULT_OMNI_DOWNLOAD_DIR,
    ovationpyme_path: str = '',
    aurora_types: tuple[str, ...] = DEFAULT_AURORA_TYPES,
    combine_hemispheres: bool = True,
    refh_km: float = DEFAULT_REFH_KM,
    hallOrPed: str = 'hp',
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    ts = pd.Timestamp(when)
    omni_cdf_path = _ensure_omni_cdf_available(ts, omni_cdf_dir)
    FluxEstimator, LatLocaltimeInterpolator, ovation_utilities = _import_ovationpyme(
        ovationpyme_path)
    omni_getter = _load_omni_cdf_getter(omni_cdf_path)
    ovation_utilities.calc_dF = lambda dt: float(omni_getter(pd.Timestamp(dt)))

    glat_arr = np.asarray(glat, dtype=float)
    glon_arr = np.asarray(glon, dtype=float)
    mlat, mlt = geographic_to_ovation_coords(
        glat_arr, glon_arr, ts, refh_km=refh_km)

    electron_energy_estimators = [FluxEstimator(
        atype, 'energy') for atype in aurora_types]
    electron_number_estimators = [FluxEstimator(
        atype, 'number') for atype in aurora_types]
    ion_energy_estimator = FluxEstimator('ions', 'energy')
    ion_number_estimator = FluxEstimator('ions', 'number')

    total_pedersen_e_sq = np.zeros_like(mlat, dtype=float)
    total_hall_e_sq = np.zeros_like(mlat, dtype=float)
    electron_contribs: list[dict[str, np.ndarray]] = []

    for energy_est, number_est in zip(electron_energy_estimators, electron_number_estimators):
        model_mlat, model_mlt, energy_flux = energy_est.get_flux_for_time(
            ts.to_pydatetime(), hemi='N', combine_hemispheres=combine_hemispheres
        )
        _, _, number_flux = number_est.get_flux_for_time(
            ts.to_pydatetime(), hemi='N', combine_hemispheres=combine_hemispheres
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            electron_eavg = (energy_flux / 1.6e-12) / number_flux / 1000.0
        electron_eavg = np.asarray(electron_eavg, dtype=float)
        electron_eavg[~np.isfinite(electron_eavg)] = 0.0
        electron_eavg[electron_eavg > 30.0] = 30.0
        electron_eavg[electron_eavg < 0.2] = 0.0
        energy_flux = np.asarray(energy_flux, dtype=float)
        energy_flux[~np.isfinite(energy_flux)] = 0.0
        energy_flux[energy_flux < 0] = 0.0

        electron_eavg_interp = LatLocaltimeInterpolator(
            model_mlat, model_mlt, electron_eavg)
        electron_eflux_interp = LatLocaltimeInterpolator(
            model_mlat, model_mlt, energy_flux)
        electron_eavg_native = np.asarray(
            electron_eavg_interp.interpolate(mlat, mlt, method='linear'),
            dtype=float,
        )
        electron_eflux_native = np.asarray(
            electron_eflux_interp.interpolate(mlat, mlt, method='linear'),
            dtype=float,
        )
        electron_eavg_native[~np.isfinite(electron_eavg_native)] = 0.0
        electron_eflux_native[~np.isfinite(electron_eflux_native)] = 0.0
        electron_eflux_native[electron_eflux_native < 0] = 0.0

        ped_type = _electron_pedersen(
            electron_eavg_native, electron_eflux_native)
        hall_type = _electron_hall(electron_eavg_native, electron_eflux_native)
        total_pedersen_e_sq += ped_type**2
        total_hall_e_sq += hall_type**2
        electron_contribs.append(
            {
                'emean': electron_eavg_native,
                'eflux': electron_eflux_native,
                'pedersen': ped_type,
                'hall': hall_type,
            }
        )

    ion_mlat, ion_mlt, ion_energy_flux = ion_energy_estimator.get_flux_for_time(
        ts.to_pydatetime(), hemi='N', combine_hemispheres=combine_hemispheres
    )
    _, _, ion_number_flux = ion_number_estimator.get_flux_for_time(
        ts.to_pydatetime(), hemi='N', combine_hemispheres=combine_hemispheres
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        ion_eavg = (ion_energy_flux / 1.6e-12) / ion_number_flux / 1000.0
    ion_eavg = np.asarray(ion_eavg, dtype=float)
    ion_eavg[~np.isfinite(ion_eavg)] = 0.0
    ion_eavg[ion_eavg > 30.0] = 30.0
    ion_eavg[ion_eavg < 0.2] = 0.0
    ion_energy_flux = np.asarray(ion_energy_flux, dtype=float)
    ion_energy_flux[~np.isfinite(ion_energy_flux)] = 0.0
    ion_energy_flux[ion_energy_flux < 0] = 0.0

    ion_eavg_interp = LatLocaltimeInterpolator(ion_mlat, ion_mlt, ion_eavg)
    ion_eflux_interp = LatLocaltimeInterpolator(
        ion_mlat, ion_mlt, ion_energy_flux)
    ion_eavg_native = np.asarray(ion_eavg_interp.interpolate(
        mlat, mlt, method='linear'), dtype=float)
    ion_eflux_native = np.asarray(ion_eflux_interp.interpolate(
        mlat, mlt, method='linear'), dtype=float)
    ion_eavg_native[~np.isfinite(ion_eavg_native)] = 0.0
    ion_eflux_native[~np.isfinite(ion_eflux_native)] = 0.0
    ion_eflux_native[ion_eflux_native < 0] = 0.0

    pedersen_electron = np.sqrt(total_pedersen_e_sq)
    hall_electron = np.sqrt(total_hall_e_sq)
    pedersen_proton = _proton_pedersen(ion_eflux_native)
    hall_proton = _proton_hall(ion_eavg_native, ion_eflux_native)

    total_hall = np.asarray(hall_electron + hall_proton, dtype=float)
    total_pedersen = np.asarray(
        pedersen_electron + pedersen_proton, dtype=float)

    hop = hallOrPed.lower()
    if hop in ('h', 'hall'):
        return total_hall
    if hop in ('p', 'pedersen'):
        return total_pedersen
    return total_hall, total_pedersen


if __name__ == '__main__':
    hall_total, pedersen_total = get_total_ovation_conductance(
        glat=np.array([67.0]),
        glon=np.array([20.0]),
        when=pd.Timestamp('2000-06-23 14:04:42'),
        omni_cdf_dir=DEFAULT_OMNI_DOWNLOAD_DIR,
    )
    print('Pedersen total:', pedersen_total)
    print('Hall total:', hall_total)

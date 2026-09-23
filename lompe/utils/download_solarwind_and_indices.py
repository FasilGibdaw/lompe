"""Download local OMNI CDF and GFZ F10.7 inputs for conductance models."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
DEFAULT_OMNI_DIR = DATA_DIR / 'omni'
F107_FILENAME = 'Kp_ap_Ap_SN_F107_since_1932.txt'
F107_PATH = DATA_DIR / F107_FILENAME
F107_URL = (
    'ftp://ftp.gfz-potsdam.de/pub/home/obs/Kp_ap_Ap_SN_F107/'
    + F107_FILENAME
)


def get_f107(when, f107_file: str | Path = F107_PATH,
             adjusted: bool = False) -> float:
    """Return the daily GFZ F10.7 value for a UTC date.

    The observed noon-time F10.7 value is used by default, following the GFZ
    file's recommendation for ionospheric and atmospheric studies.  Set
    ``adjusted=True`` to select the adjusted F10.7 column instead.
    """
    path = Path(f107_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f'F10.7 file not found: {path}. Run this script without --skip-f107 first.'
        )
    try:
        target_date = np.datetime64(when, 'D')
    except ValueError as exc:
        raise ValueError(f'Cannot interpret {when!r} as a date.') from exc
    data = np.loadtxt(path, comments='#', usecols=(0, 1, 2, 25, 26))
    dates = np.array(
        [f'{int(year):04d}-{int(month):02d}-{int(day):02d}'
         for year, month, day in data[:, :3]],
        dtype='datetime64[D]',
    )
    matches = np.flatnonzero(dates == target_date)
    if not matches.size:
        raise KeyError(f'No F10.7 value found for {target_date} in {path}.')
    value = float(data[matches[0], 4 if adjusted else 3])
    if value <= 0:
        kind = 'adjusted' if adjusted else 'observed'
        raise ValueError(f'GFZ {kind} F10.7 is missing for {target_date}.')
    return value


def latest_f107_date(f107_file: str | Path = F107_PATH) -> np.datetime64:
    """Return the most recent daily record in the downloaded GFZ history."""
    path = Path(f107_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'F10.7 file not found: {path}')
    for line in reversed(path.read_text().splitlines()):
        if line and not line.startswith('#'):
            year, month, day = line.split()[:3]
            return np.datetime64(f'{year}-{month}-{day}', 'D')
    raise ValueError(f'No daily F10.7 records found in {path}.')


def omni_filename(year_month: str) -> str:
    try:
        year, month = (int(value) for value in year_month.split('-'))
        if not 1 <= month <= 12:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Month must use YYYY-MM format.') from exc
    return f'omni_hro_1min_{year:04d}{month:02d}01_v01.cdf'


def download(url: str, destination: Path, overwrite: bool = False) -> Path:
    """Download atomically, retaining an existing complete file by default."""
    if destination.exists() and not overwrite:
        print(f'Using existing {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix='.download-', dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        print(f'Downloading {url}')
        urlretrieve(url, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f'Saved {destination}')
    return destination


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Download local data files used by Ovation and Lompe conductance models.')
    parser.add_argument('--month', action='append', default=[], metavar='YYYY-MM',
                        help='OMNI 1-minute month to download; may be repeated.')
    parser.add_argument('--skip-f107', action='store_true',
                        help='Do not download the GFZ F10.7 history file.')
    parser.add_argument('--f107-date', metavar='YYYY-MM-DD',
                        help='Refresh F10.7 only if the local file does not cover this UTC date.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing files.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.month and args.skip_f107:
        raise SystemExit('Specify at least one --month or omit --skip-f107.')

    for month in args.month:
        filename = omni_filename(month)
        year = month[:4]
        url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/omni/hro_1min/{year}/{filename}'
        download(url, DEFAULT_OMNI_DIR / filename, args.overwrite)

    if not args.skip_f107:
        f107_path = F107_PATH
        required_date = None
        if args.f107_date:
            try:
                required_date = np.datetime64(args.f107_date, 'D')
            except ValueError as exc:
                raise SystemExit('--f107-date must use YYYY-MM-DD format.') from exc

        needs_download = args.overwrite or not f107_path.exists()
        if required_date is not None and not needs_download:
            latest_date = latest_f107_date(f107_path)
            needs_download = latest_date < required_date
            if not needs_download:
                print(f'Using {f107_path}; it covers through {latest_date}.')
        if needs_download:
            download(F107_URL, f107_path, overwrite=True)
            if required_date is not None and latest_f107_date(f107_path) < required_date:
                raise SystemExit(
                    f'Downloaded F10.7 history does not yet cover {required_date}.'
                )
        elif required_date is None:
            print(f'Using existing {f107_path}')


if __name__ == '__main__':
    main()

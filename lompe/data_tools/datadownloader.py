from lompe.data_tools.dataloader import cross_track_los, radar_losvec_from_mag
from scipy import interpolate
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import requests
from requests_ftp import ftp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import datetime as dt
import os
import cdflib
import ppigrf
import certifi
from lompe.data_tools.supermag_api import SuperMAGGetData, sm_GetUrl, sm_coreurl
from lompe.data_tools.supermag_direct import download_data_for_event as ddfe
import glob
import shutil
import itertools
from concurrent.futures import ProcessPoolExecutor
from lompe.utils.time import date2doy

warnings.filterwarnings("ignore")

# extensive list of functions (including helper functions) to download data from different sources for a given event date
# the functions are designed to be used in the lompe package for data loading and processing
# this has to be integrated with the dataloader.py in the lompe package at some level to get lompe data format

# the idea is to merge the two files (datadownloader.py and dataloader.py) into one file in the lompe package


def fetch_sussi_urls(sat, year, doy, source):
    doy_str = f"{doy:03d}"  # Zero-pad the day of year to three digits
    if source == 'jhuapl':
        url = f"https://ssusi.jhuapl.edu/data_retriver?spc=f{sat}&type=edr-aur&year={year}&Doy={doy_str}"
    elif source == 'cdaweb':
        url = f'https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmspf{sat}/ssusi/data/edr-aurora/{year}/{doy_str}/'
    else:
        print(f"Unsupported source: {source}")
        return []

    try:
        response = requests.get(url)
        response.raise_for_status()  # raise an HTTPError on bad response
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch URL probably no data : \n{e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    urls = [urljoin(url, link.get('href')) for link in soup.find_all(
        'a', href=True) if link.get('href').lower().endswith('.nc')]

    if not urls:
        print(
            f"No .nc files found for satellite F{sat} on day {doy_str} of year {year} from source {source}")

    return urls


def download_sussi_file(file_url, destination):
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  # raise an HTTPError on bad response
        filename = os.path.join(destination, os.path.basename(file_url))

        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:  # Filter out keep-alive new chunks
                    file.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file_url}: {e}")


def download_sussi(event, source='cdaweb'):
    """Dowloading data from the SSUSI instrument onboard the DMSP satellites for a given event date.

    Args:
        event strin: format 'YYYY-MM-DD'
        source (str, optional): data source. Defaults to 'cdaweb'. Another option is 'jhuapl'.
    """
    year = int(event[0:4])
    doy = date2doy(event)
    destination = 'sussi_tempfiles'
    os.makedirs(destination, exist_ok=True)

    # Use ProcessPoolExecutor to fetch URLs concurrently
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(fetch_sussi_urls, [16, 17, 18, 19], [
                               year] * 4, [doy] * 4, [source] * 4)
        all_urls = list(itertools.chain.from_iterable(results))

    # Use ProcessPoolExecutor to download files concurrently
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(download_sussi_file, all_urls,
                     [destination] * len(all_urls))


def download_smag(event, tempfile_path='./', hemi='all'):
    """Download SuperMAG data for a given event date (this can entirely substitute the read_smag function in the lompe package (data_tools))
    Example usage:
        event = '2012-04-05'
        tempfile_path = 'downloads'
        hemi = 'all'
        download_smag(event, tempfile_path, hemi)

    Args:
        event (str): format YYYY-MM-DD
        tempfile_path (str, optional): path to save processed file or check if exists already. Defaults to './'.
        hemi (str, optional): filtering magnetometer stations based on hemsiphere or all the stations. Defaults to 'all'.

    Raises:
        ValueError: throwing error if the download fails

    Returns:
        saved file path if successful: str

    Note: not sure this is the one which is daily basedlined: need to be checked
    """

    start = event + 'T00:00:00'
    duration = 86400  # Duration in seconds (one day)
    # lompe username is already registered in the API
    urlstr = sm_coreurl('inventory.php', 'lompe', start, duration)
    success, stations = sm_GetUrl(urlstr, 'raw')
    stations = stations[1:-1]
    savefile = tempfile_path + event.replace('-', '') + '_supermag.h5'
    if os.path.isfile(savefile):
        return savefile

    elif success and stations:
        basepath = os.path.dirname(__file__)
        file = basepath + '/../data/supermag_stations.csv'
        data_temp = pd.read_csv(file, sep=',', nrows=0)
        df = pd.read_csv(file, sep=',', skiprows=1, header=None,
                         names=data_temp.columns, usecols=range(len(data_temp.columns)))
        if hemi == 'north':
            sta2 = df[df['GEOLAT'] > 40].IAGA.values
            intersection = set(stations).intersection(sta2)
            stations = list(intersection)
        elif hemi == 'south':
            sta2 = df[df['GEOLAT'] < -40].IAGA.values
            intersection = set(stations).intersection(sta2)
            stations = list(intersection)
        elif hemi == 'all':
            stations = stations
        print(
            f"Number of stations available for the selected date is: {len(stations)}")

        # checking the stations and sucess in the geturl inquiry
        if not success or not stations:
            raise ValueError(
                "Failed to fetch stations. Please check the input parameters or API availability.")
        results = []

        # Function to download data for a given station
        def download_data(station):
            success, df = SuperMAGGetData(
                'lompe', start, duration, 'geo', station, BASELINE='yearly')
            if success:
                return df
            else:
                return pd.DataFrame()  # Return an empty DataFrame if the download fails

        # Download data serially for each station
        for station in stations:
            # print(f"Downloading data for station {station}...")
            df = download_data(station)
            if not df.empty:
                results.append(df)
            # time.sleep(1)  # Optional: Sleep between requests to avoid overwhelming the server

        # Combine results into a single DataFrame
        if not results:
            raise ValueError(
                "No valid data downloaded. Please check API or parameters.")

        df_combined = pd.concat(results, ignore_index=True)
        # date conversion and cleaning the DataFrame
        df_combined['tval'] = pd.to_datetime(
            df_combined['tval'], unit='s', origin='unix')
        df_combined[['N', 'E', 'Z']] = df_combined[['N', 'E', 'Z']].map(
            lambda x: x['geo'] if isinstance(x, dict) else np.nan)
        df_combined[['N', 'E', 'Z']] = df_combined[[
            'N', 'E', 'Z']].replace(999999.000000, np.nan)

        # Final DataFrame to save as hdf
        df_combined.set_index('tval', inplace=True)
        df_combined.rename(columns={
            'glat': 'lat', 'glon': 'lon', 'N': 'Bn', 'E': 'Be', 'Z': 'Bu'}, inplace=True)
        df_combined['Bu'] = -df_combined['Bu']
        df_final = df_combined[['Be', 'Bn', 'Bu',
                                'lat', 'lon']].dropna().sort_index()

        # df_final.to_hdf('20120405_supermag_data.h5', key='df_final', mode='w')
        finishedfile = tempfile_path + \
            event.replace('-', '') + '_supermag.h5'

        df_final.to_hdf(finishedfile, key='df_final', mode='w')

        # print("Data processing complete.")
        return finishedfile
    else:
        raise ValueError(
            'Something went wrong, check inputs, API availability etc.')


def ampere_parsestart(start):
    # internal helper function from supermag_api.py

    # takes either list of [yyyy, mo, dd, hh, mm, opt_ss]
    # or string of a normal datetime 'YYYY-MM-DD hh-mm' (optional ss)
    # or the SuperMAG-ready 'YYYY-MM-DDThh-mm-ss'

    if isinstance(start, list):
        timestring = "%4.4d-%2.2d-%2.2dT%2.2d:%2.2d" % tuple(start[0:5])
    elif isinstance(start, dt.date):
        # good to go, TBD
        timestring = start.strftime("%Y-%m-%dT%H:%M")
    else:
        # is a string, reparse, TBD
        timestring = start

    return (timestring)


def ampere_coreurl(page, logon, start, extent):
    # internal helper function from supermag_api.py
    baseurl = "https://ampere.jhuapl.edu/"

    mytime = ampere_parsestart(start)
    urlstr = baseurl + 'services/' + page + '?'
    urlstr += '&logon=' + logon
    urlstr += '&start=' + mytime

    urlstr += '&extent=' + ("%12.12d" % extent)

    return (urlstr)


def download_iridium(event, basepath='./', tempfile_path='./', file_name=''):
    """Download netcdf (dB raw) data to be used by lompe from the AMPERE database (jhuapl) for a given event
    returns an input for the lompe read_iridium script in dataloader.py in data_tools
    Example usage:
        event = '2012-04-05'
        basepath = 'downloads'
        tempfile_path = 'downloads'
        file_name = '20120405_iridium.h5'
        download_iridium(event, basepath, tempfile_path, file_name)

    Args:
        event (str): fromat YYYY-MM-DD
        basepath (str, optional): path to . Defaults to './'.
        tempfile_path (str, optional): path to. Defaults to './'.
        file_name (str, optional):name of the file to write the netcdf file. Defaults to ''.

    Returns:
        saved file: to be used by the lompe read_iridium function in data_tools

    Note: 
        functions "ampere_parsestart" and "ampere_coreurl" are internal helper functions adapted from supermag_api.py
        credit to the original author of the functions.
    """

    start = event + 'T00:00:00'
    duration = 86400  # Duration in seconds (one day)
    # check if the processed file exists
    savefile = tempfile_path + event.replace('-', '') + '_iridium.nc'

    if os.path.isfile(savefile):  # checks if file already exists
        return savefile
    else:

        # URL to download data from (lompe username is already registered in the API)
        urlstr = ampere_coreurl('data-rawdB.php', 'lompe', start, duration)

        response = requests.get(urlstr, verify=certifi.where())

        # Check if the request was successful
        if response.status_code == 200:
            # Save the downloaded data to a file
            with open(savefile, 'wb') as file:
                file.write(response.content)
        else:
            print(f"Failed to retrieve data: {response.status_code}")
        return savefile


def download_supermag(event, tempfile_path='./'):
    """This downlaods data from superMAG for a given event and returns the hdf file suitable to lompe
       This is a faster way i can think of to download data from superMAG, the download_smag.py is slow
       since it uses serial processing to download data for each station. This uses multiprocessing to download (see the supermag_direct.py for the multiprocessing implemetation)

    Args:
        event (str): format 'YYYY-MM-DD'

    Returns:
        hdf file: this returns the hdf file with the data for the event, no need of using read_smag in the lompe dataloader.py script 

    Note: not sure the data we get here is the one which is daily basedlined: need to be checked
    """
    # event = '2012-04-05'
    start = event + 'T00:00:00'
    savefile = tempfile_path + event.replace('-', '') + '_supermag.h5'

    if os.path.isfile(savefile):  # checks if file already exists
        return savefile
    else:
        # run the function to download the data in the tempfiles folder (later to be deleted if successful)
        ddfe(start)

        files = glob.glob('./tempfiles/*.txt')
        df_combined = pd.DataFrame()
        for file in files:
            data = pd.read_json(file)
            df_combined = pd.concat([df_combined, data], axis=0)
        # date conversion and cleaning the DataFrame
        df_combined['tval'] = pd.to_datetime(
            df_combined['tval'], unit='s', origin='unix')
        df_combined[['N', 'E', 'Z']] = df_combined[['N', 'E', 'Z']].map(
            lambda x: x['geo'] if isinstance(x, dict) else np.nan)
        df_combined[['N', 'E', 'Z']] = df_combined[[
            'N', 'E', 'Z']].replace(999999.000000, np.nan)

        # Final DataFrame to save as hdf
        df_combined.set_index('tval', inplace=True)
        df_combined.rename(columns={
            'glat': 'lat', 'glon': 'lon', 'N': 'Bn', 'E': 'Be', 'Z': 'Bu'}, inplace=True)
        df_combined['Bu'] = -df_combined['Bu']
        df_final = df_combined[['Be', 'Bn', 'Bu',
                                'lat', 'lon']].dropna().sort_index()

        # savefile = event.replace('-', '') + '_supermag.h5'

        df_final.to_hdf(savefile, key='df_final', mode='w')
        # remove the tempfiles folder after the hdf file is created
        shutil.rmtree('./tempfiles/')
        return savefile


def download_champ(event, basepath='./', tempfile_path='./'):
    """
    Download CHAMP data from the FTP server and process it in lompe data format.
    Note that CHAMP data is only available for the year between 2000 and 2010.

    Args:
        event (str): format 'YYYY-MM-DD'
        basepath (str, optional): path. Defaults to './'.
        tempfile_path (str, optional): path. Defaults to './'.

    Returns:
        savedfile: file name of the processed file if successful
    """
    event_date = event.replace('-', '')
    year = event[:4]
    savefile = tempfile_path + f'CH_ME_MAG_LR_3_{event_date}_0102.cdf'
    processed_file = tempfile_path + f'{event_date}_champ.h5'

    # Check if the processed file already exists
    if os.path.isfile(processed_file):
        return processed_file

    # Check if the raw file already exists
    if not os.path.isfile(savefile):
        session = requests.Session()
        session.mount('ftp://', ftp.FTPAdapter())
        ftp_url = f"ftp://isdcftp.gfz-potsdam.de/champ/ME/Level3/MAG/V0102/{year}/CH_ME_MAG_LR_3_{event_date}_0102.cdf"
        try:
            # Downloading the file and checking if it was successful
            response = session.get(ftp_url)
            if response.status_code == 200:
                with open(savefile, "wb") as file:
                    file.write(response.content)
                print(f"Downloading {savefile} is successful!")
            else:
                print(
                    f"Failed to download the file. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(e)
            return None

    # Process the downloaded CDF file ot get the magnetic disturbance
    try:
        cdf_file = cdflib.CDF(savefile)
        mag = cdf_file.varget('B_NEC')  # space magnetometer data

        # geocentric coords of CHAMP orbit
        theta = 90 - cdf_file.varget('Latitude')
        phi = cdf_file.varget('Longitude')
        r = cdf_file.varget('Radius') / 1000

        time = cdflib.cdfepoch.to_datetime(cdf_file.varget('Timestamp'))

        # using IGRF to calculate magnetic disturbance (dB) registered by CHAMP
        Br, Btheta, Bphi = ppigrf.igrf_gc(r, theta, phi, time[0])
        B0 = np.vstack((-Btheta.flatten(), Bphi.flatten(), -Br.flatten()))
        dB = mag.T - B0

        champ_df = pd.DataFrame({
            'Be': dB[1],
            'Bn': dB[0],
            'Br': -dB[2],
            'lon': phi,
            'lat': 90 - theta,
            'r': r
        }, index=time)
        champ_df.to_hdf(processed_file, key='df', mode='w')
        return processed_file
    except Exception as e:
        print(f"Failed to process the file: {e}")
        return None


def download_sdarn_file(url, save_path):
    # function to download a file from a URL and save it to a specific path
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        # print(f"Downloaded successfully: {save_path}")
    else:
        return None


def download_sdarn_files(event, basepath='./'):
    filepath = os.path.dirname(__file__)
    # file containing the URLs of the SuperDARN files (zenodo records)
    file_loc = pd.read_csv(filepath + '/../data/sdarn_2010_to_2021.csv')
    # Month mapping
    month_map = {
        'Jan': '01',
        'Feb': '02',
        'Mar': '03',
        'Apr': '04',
        'May': '05',
        'Jun': '06',
        'Jul': '07',
        'Aug': '08',
        'Sep': '09',
        'Oct': '10',
        'Nov': '11',
        'Dec': '12'
    }

    # Add numerical month to DataFrame (string format)
    file_loc['Month_Num'] = file_loc['MM'].map(month_map)
    # event_date = "2019-02-25"

    # Filter the DataFrame based on the year and month of the event date
    year = event[:4]
    month = event[5:7]

    # Filter the DataFrame
    filtered_df = file_loc[(file_loc['year'].astype(
        str) == year) & (file_loc['Month_Num'] == month)]

    # Apply function and add to DataFrame
    event_date_str = event.replace('-', '')

    # URL of the Zenodo record
    url = filtered_df['url'].tolist()[0]

    # Send a GET request to the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Check if the request was successful (status code 200)
    if response.status_code == 200:

        # Find all <a> tags with class "download-file"
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if 'grid.nc' in href and event_date_str in href:
                url_to_download = 'https://zenodo.org' + href
                # print(url_to_download)
                save_path = basepath + \
                    url_to_download.split('/')[-1].split('?')[0]
                download_sdarn_file(url_to_download, save_path)
            # else:
            #     print('No file found')
    else:
        print('Failed to download the file')
    return None


def download_sdarn(event, basepath='./', tempfile_path='./'):
    # tempfile_path = '/Users/fasilkebede/Documents/LOMPE/Data/SuperDARN/'
    # event = '2019-02-25'

    savefile = tempfile_path + event.replace('-', '') + '_superdarn_grdmap.h5'
    if os.path.isfile(savefile):
        return savefile
    else:
        os.makedirs(basepath + 'sdarn_files', exist_ok=True)
        download_sdarn_files(event, basepath + '/sdarn_files/')
        # looking for the .nc files for the event
        files = glob.glob(
            f"{basepath}/sdarn_files/*{event.replace('-', '')}*.nc")
        files.sort()
        ddd = pd.DataFrame()
        for file in files:
            sm = xr.load_dataset(file)
            st_abbrev = file.split('/')[-1].split('.')[1]
            # mjd conversion
            mjd_epoch = pd.Timestamp('1858-11-17')
            duration = (sm['mjd_end'] + mjd_epoch) - \
                (sm['mjd_start'] + mjd_epoch)

            time = (sm['mjd_start'] + mjd_epoch) + duration

            # dff['date'] = unix_epoch + dff.mjd_start

            temp = pd.DataFrame()

            # in degrees AACGM
            temp.loc[:, 'mlat'] = sm['vector.mlat'].values
            # in degrees AACGM
            temp.loc[:, 'mlon'] = sm['vector.mlon'].values
            # glat, glon from lompe "radar_losvec_from_mag" is a bit different from the vector.glat and vector.glon from the data??
            temp.loc[:, 'vector.glat'] = sm['vector.glat'].values  # in degrees
            temp.loc[:, 'vector.glon'] = sm['vector.glon'].values  # in degrees
            # in degrees, the angle between los and magnetic north
            temp.loc[:, 'azimuth'] = sm['vector.kvect'].values
            temp.loc[:, 'vlos'] = sm['vector.vel.median'].values       # in m/s
            temp.loc[:, 'vlos_sd'] = sm['vector.vel.sd'].values         # in m/s
            temp.loc[:, 'range'] = sm['vector.pwr.median'].values       # in km
            # spectral width in m/s
            temp.loc[:, 'wdt'] = sm['vector.wdt.median'].values
            temp.loc[:, 'time'] = pd.to_datetime(time.values).round('s')
            temp.loc[:, 'radar'] = st_abbrev
            # temp.set_index = pd.to_datetime(temp['time'], unit='s')
            # dff['datetime'] = mjd_epoch + dff['mjd_start']
            ddd = pd.concat([ddd, temp], ignore_index=True)
        ddd.set_index('time', inplace=True)
        ddd['glat'], ddd['glon'], ddd['le'], ddd['ln'], ddd['le_m'], ddd['ln_m'] = radar_losvec_from_mag(ddd['mlat'].values,
                                                                                                         ddd['mlon'].values, ddd['azimuth'].values, ddd.index[0])
        dd = ddd[ddd['glat'] > 0]  # restrict to northern hemisphere
        df_final = dd.sort_values(by='time')
        df_final.to_hdf(savefile, key='df', mode='w')
        shutil.rmtree('./sdarn_files/')

        return savefile


def download_swarm(event, tempfile_path='./'):
    """Download Swarm data for a given event date.

    Args:
        event (str): Event date in 'YYYY-MM-DD' format.
        tempfile_path (str, optional): Path to save the file or check if it already exists. Defaults to './'.

    Returns:
        str: File name of the downloaded file if successful.

    Note:
        1. Install viresclient using "pip install viresclient" or make sure that it is installed.
        2. The request needs an access token from the Swarm website. Visit https://viresclient.readthedocs.io/en/latest/config_details.html.
    """

    savefile = tempfile_path + event.replace('-', '') + '_swarm.h5'

    if os.path.isfile(savefile):
        return savefile

    try:
        from viresclient import SwarmRequest
    except ModuleNotFoundError:
        print('Please install viresclient using "pip install viresclient"')
        return
    # checking if token is present if not directing to the website to configure it
    try:
        with open(os.path.expanduser('~/.viresclient.ini'), 'r') as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith("token ="):
                token_value = line.split('=', 1)[1].strip()
                if token_value:
                    print("Swarm token is present:", token_value)
    except:
        print("Token is missing or empty. \nPlease visit https://viresclient.readthedocs.io/en/latest/config_details.html to configure it")
        return
    try:
        request = SwarmRequest()

        event_start = dt.datetime.strptime(event, '%Y-%m-%d')
        event_end = event_start + \
            dt.timedelta(hours=23, minutes=59, seconds=59)
        df = pd.DataFrame()

        for swarm_satellite in ['A', 'B', 'C']:
            request.set_collection(f"SW_OPER_MAG{swarm_satellite}_LR_1B")
            request.set_products(
                measurements=["F", "B_NEC"],
                models=["IGRF", "MCO_SHA_2D"],
                sampling_step="PT10S",  # 10 seconds if needed PT1M for 1 minute
                # quasi-dipole latitude and longitude
                auxiliaries=["MLT", "OrbitNumber", 'QDLat', 'QDLon']
            )
            data = request.get_between(
                start_time=event_start,
                end_time=event_end,
                show_progress=False
            )
            df = pd.concat([df, data.as_dataframe(expand=True)])

        # Removing the background field from IGRF
        df['B_n'] = df['B_NEC_N'] - df['B_NEC_IGRF_N']
        df['B_e'] = df['B_NEC_E'] - df['B_NEC_IGRF_E']
        # Upward is negative in the NEC systema
        df['B_u'] = -(df['B_NEC_C'] - df['B_NEC_IGRF_C'])

        df.sort_values(by='Timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_hdf(savefile, key='df', mode='w')
        return savefile

    except Exception as e:
        print(f"An error occurred while processing the Swarm data: {e}")


def download_dmsp_ssies(event, sat, tempfile_path='./', **madrigal_kwargs):
    """ Download DMSP SSIES ion drift meter data for a full day
        using FTP-like acess at http://cedar.openmadrigal.org/ftp/ and get relevant parameters for Lompe. 
        Saves hdf file in tempfile_path and returns path

    Args:
        event (str): 
            string on format 'yyyy-mm-dd' to specify time of event for model.
        sat (int): 
            Satellite ID for DMSP Block 5D satellite (16-19)
        tempfile_path (str, optional): 
            Path to dir where processed hdf files are placed. Default: './'
        **madrigalkwargs (dict): 
            needed to download data from MadrigalWeb (Madrigal needs user specifications through **madrigal_kwargs.)
    Example usage:
        event = '2014-07-13'
        sat = 17
        tempfile_path = '/Users/fasilkebede/Documents/'
        madrigal_kwargs = {'user_fullname': 'First','user_email': 'name@host.com', 'user_affiliation': 'University'}

    Returns:
        savefile(str):
            Path to hdf-file containing SSIES data for Lompe.
    """

    savefile = tempfile_path + \
        event.replace('-', '') + '_ssies_f' + str(sat) + '.h5'
    if os.path.exists(savefile):
        print('File already exists')
        return savefile

    date_str = event.replace('-', '')
    year = event[:4]
    url_base = "http://cedar.openmadrigal.org"
    url = url_base + \
        f"/ftp/fullname/{madrigal_kwargs['user_fullname']}/email/{madrigal_kwargs['user_email']}/affiliation/{madrigal_kwargs['user_affiliation']}/kinst/8100/year/{year}/"

    response2 = requests.get(url)
    soup2 = BeautifulSoup(response2.content, 'html.parser')
    urls = [link.get('href') for link in soup2.find_all(
        'a', href=True) if '/format/hdf5/' in link.get('href')]

    url_ion_drift = url_base + [a['href'] for a in soup2.find_all(
        'a', href=True) if 'ion drift' in a.text and f'F{sat}' in a.text][0]
    url_plasma_temp = url_base + [a['href'] for a in soup2.find_all(
        'a', href=True) if 'plasma temp' in a.text and f'F{sat}' in a.text][0]
    # url_flux_energy = url_base + [a['href'] for a in soup2.find_all(
    #     'a', href=True) if 'flux/energy' in a.text and f'F{sat}' in a.text][0]

    # downloading the ion drift file
    ion_drift_hdf5_file_url = url_ion_drift + 'format/hdf5/'
    response2 = requests.get(ion_drift_hdf5_file_url)
    soup2 = BeautifulSoup(response2.content, 'html.parser')
    url_target_file = url_base + [link.get('href') for link in soup2.find_all(
        'a', href=True) if '/format/hdf5/fullFilename/' in link.get('href') and date_str in link.get('href')][0]

    response = requests.get(url_target_file, stream=True)
    # tempfile_path = '/Users/fasilkebede/Documents/'
    filename = tempfile_path + url_target_file[-27:-1]
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:  # Filter out keep-alive new chunks
                file.write(chunk)

    # downloading the plasma temperature file
    plasma_hdf5_file_url = url_plasma_temp + 'format/hdf5/'
    response2 = requests.get(plasma_hdf5_file_url)
    soup2 = BeautifulSoup(response2.content, 'html.parser')
    url_target_file = url_base + [link.get('href') for link in soup2.find_all(
        'a', href=True) if '/format/hdf5/fullFilename/' in link.get('href') and date_str in link.get('href')][0]

    response = requests.get(url_target_file, stream=True)
    # tempfile_path = '/Users/fasilkebede/Documents/'
    filename2 = tempfile_path + url_target_file[-27:-1]
    with open(filename2, 'wb') as file:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:  # Filter out keep-alive new chunks
                file.write(chunk)

    dmsp = pd.read_hdf(filename, mode='r', key='Data/Table Layout')
    dmsp2 = pd.read_hdf(filename2, mode='r', key='Data/Table Layout')

    dmsp.index = np.arange(len(dmsp))
    dmsp2.index = np.arange(len(dmsp2))

    # set datetime as index
    times = []
    for i in range(len(dmsp)):
        times.append(dt.datetime.fromtimestamp(
            int(dmsp['ut1_unix'][i]), tz=dt.timezone.utc))
    dmsp.index = times

    # reindexing due to lower cadence measurements
    times2 = []
    for i in range(len(dmsp2)):
        times2.append(dt.datetime.fromtimestamp(
            int(dmsp2['ut1_unix'][i]), tz=dt.timezone.utc))
    dmsp2.index = times2
    dmsp2 = dmsp2.reindex(index=dmsp.index, method='nearest', tolerance='2sec')

    dmsp.loc[:, 'po+'] = dmsp2['po+']
    dmsp.loc[:, 'te'] = dmsp2['te']
    dmsp.loc[:, 'ti'] = dmsp2['ti']

    # Smooth the orbit
    import matplotlib

    # latitude (geodetic)
    tck = interpolate.splrep(matplotlib.dates.date2num(dmsp.index[::60].append(dmsp.index[-1:])),
                             pd.concat([dmsp.gdlat[::60], (dmsp.gdlat[-1:])]).values, k=2)
    gdlat = interpolate.splev(matplotlib.dates.date2num(dmsp.index), tck)

    # longitude (geodetic)
    negs = dmsp.glon < 0
    dmsp.loc[negs, 'glon'] = dmsp.glon[negs] + 360
    tck = interpolate.splrep(matplotlib.dates.date2num(dmsp.index[::60].append(dmsp.index[-1:])),
                             pd.concat([dmsp.glon[::60], (dmsp.glon[-1:])]).values, k=2)
    glon = interpolate.splev(matplotlib.dates.date2num(dmsp.index), tck)

    # altitude (geodetic)
    tck = interpolate.splrep(matplotlib.dates.date2num(dmsp.index[::60].append(dmsp.index[-1:])),
                             pd.concat([dmsp.gdalt[::60], (dmsp.gdalt[-1:])]).values, k=2)
    gdalt = interpolate.splev(matplotlib.dates.date2num(dmsp.index), tck)

    # get eastward and northward component of cross track direction
    le, ln, bearing = cross_track_los(dmsp['hor_ion_v'], gdlat, glon)

    # put together the relevant data
    ddd = pd.DataFrame()      # dataframe to return
    ddd.index = dmsp.index
    ddd.loc[:, 'gdlat'] = gdlat
    ddd.loc[:, 'glon'] = glon
    ddd.loc[:, 'gdalt'] = gdalt
    ddd.loc[:, 'hor_ion_v'] = np.abs(dmsp.hor_ion_v)
    ddd.loc[:, 'vert_ion_v'] = dmsp.vert_ion_v
    ddd.loc[:, 'bearing'] = bearing
    ddd.loc[:, 'le'] = le
    ddd.loc[:, 'ln'] = ln

    # quality flag from Zhu et al 2020 page 8 https://doi.org/10.1029/2019JA027270
    # flag1 is good, flag2 is also usually acceptable

    flag1 = (dmsp['ne'] > 1e9) & (dmsp['po+'] > 0.85)
    flag2 = ((dmsp['po+'] > 0.85) & (dmsp['ne'] > 1e8) & (dmsp['ne'] < 1e9)) | ((dmsp['po+'] > 0.75)
                                                                                & (dmsp['po+'] < 0.85) & (dmsp['ne'] > 1e8))
    flag3 = (dmsp['ne'] < 1e8) | (dmsp['po+'] < 0.75)
    flag4 = dmsp['po+'].isna()

    ddd.loc[:, 'quality'] = np.zeros(ddd.shape[0]) + np.nan
    ddd.loc[flag1, 'quality'] = 1
    ddd.loc[flag2, 'quality'] = 2
    ddd.loc[flag3, 'quality'] = 3
    ddd.loc[flag4, 'quality'] = 4

    ddd.to_hdf(savefile, key='df', mode='w')
    os.remove(filename)
    os.remove(filename2)
    return savefile

def download_eiscat():
    pass 
# def date2doy(date_str):
#     date = dt.datetime.strptime(date_str, "%Y-%m-%d")
#     return date.timetuple().tm_yday

if __name__ == '__main__':
    print("This is a module to download data from different sources for a given event date.")

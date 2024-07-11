from lompe.data_tools.dataloader import radar_losvec_from_mag
import xarray as xr
import numpy as np
import pandas as pd
import requests
from requests_ftp import ftp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import datetime as dt
import os
import certifi
from lompe.data_tools.supermag_api import SuperMAGGetData, sm_GetUrl, sm_coreurl
from lompe.data_tools.supermag_direct import download_data_for_event as ddfe
import glob
import shutil


def download_sussi(event, destination='downloads', source='jhuapl'):
    """function to download SSUSI data for a given event date
    Example usage:
        event = '2014-08-20'
        detination = 'downloads'
        source = 'cdaweb'
        download_sussi(event, destination, source)

    Args:
        event (str): format YYYY-MM-DD
        destination (str, optional): where to save the data. Defaults to 'downloads'.
        source (str, optional): Defaults to 'jhuapl'. cdaweb is the other option.

    Note: I (Fasil) prefer the cdaweb because it is more faster to downlaod.
          but the read_sussi function in lompe package is tailored to the jhuapl data.
    """

    year = int(event[0:4])
    doy = date2doy(event)
    os.makedirs(destination, exist_ok=True)
    # iterate over the satellites with SSUSI data
    for sat in [16, 17, 18, 19]:
        if source == 'jhuapl':
            url = f"https://ssusi.jhuapl.edu/data_retriver?spc=f{sat}&type=edr-aur&year={year}&Doy={doy}"
        elif source == 'cdaweb':
            url = f'https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmspf{sat}/ssusi/data/edr-aurora/{year}/{doy}/'
        else:
            print(f"Unsupported source: {source}")
            continue
        # content of the webpage
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href.lower().endswith('.nc'):  # looking for .NC (jhuapl) and .nc (cdaweb) files
                file_url = urljoin(url, href)
                try:
                    # Download the file
                    # probably too much print uncommented the print statement if needed
                    # print(f"Downloading {file_url}...")
                    response = requests.get(file_url, stream=True)
                    response.raise_for_status()  # raise an HTTPError on bad response
                    filename = os.path.join(
                        destination, os.path.basename(file_url))

                    with open(filename, 'wb') as file:
                        # 64KB chuck size
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:  # Filter out keep-alive new chunks
                                file.write(chunk)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download {file_url}: {e}")

    print("Download complete!")
    return filename


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
    urlstr = baseurl + 'services/'+page+'?'
    urlstr += '&logon='+logon
    urlstr += '&start='+mytime

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
    """Download CHAMP data from the FTP server (source: https://www.gfz-potsdam.de). 
       Note that the data is only available for the year between 2000 and 2010.

    Args:
        event (str): format 'YYYY-MM-DD'
        basepath (str, optional): path. Defaults to './'.
        tempfile_path (str, optional): path. Defaults to './'.

    Returns:
        savedfile: file name of the downloaded file if successful
    """
    event_date = event.replace('-', '')
    year = event[:4]
    savefile = tempfile_path + event.replace('-', '') + '_champ.cdf'
    savefile = tempfile_path + f'CH_ME_MAG_LR_3_{event_date}_0102.cdf'
    if os.path.isfile(savefile):  # checks if file already exists
        return savefile
    else:
        # Register the FTP adapter
        session = requests.Session()
        session.mount('ftp://', ftp.FTPAdapter())
        # URL of the file
        try:
            ftp_url = f"ftp://isdcftp.gfz-potsdam.de/champ/ME/Level3/MAG/V0102/{year}/CH_ME_MAG_LR_3_{event_date}_0102.cdf"

            # Download the file
            response = session.get(ftp_url)

            # Check if the download was successful
            if response.status_code == 200:
                # Write the content to a file
                # savefile = ftp_url.split('/')[-1]
                with open(savefile, "wb") as file:  # downloading the file
                    file.write(response.content)
                print("Download successful!")
            else:
                print(
                    f"Failed to download the file. Status code: {response.status_code}")
        except Exception as e:
            print(e)
        return savefile


def download_file(url, save_path):
    # function to download a file from a URL and save it to a specific path
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        # print(f"Downloaded successfully: {save_path}")
    else:
        return None


def download_sdarn_files(event, basepath='./'):
    basepath = os.path.dirname(__file__)
    # file containing the URLs of the SuperDARN files (zenodo records)
    file_loc = pd.read_csv(basepath + '/../data/sdarn_2010_to_2021.csv')
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
                download_file(url_to_download, save_path)
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
        os.makedirs('sdarn_files', exist_ok=True)
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


def download_dmsp():
    pass


def download_swarm():
    pass


def date2doy(date_str):
    date = dt.datetime.strptime(date_str, "%Y-%m-%d")
    return date.timetuple().tm_yday


if __name__ == '__main__':
    print("This is a module to download data from different sources for a given event date.")

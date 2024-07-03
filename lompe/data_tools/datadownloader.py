import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import datetime as dt
import os
from lompe.data_tools.supermag_api import SuperMAGGetData, sm_GetUrl, sm_coreurl


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
    return None


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
    """

    start = event + 'T00:00:00'
    duration = 86400  # Duration in seconds
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


def download_iridium():
    pass


def download_dmsp():
    pass


def download_swarm():
    pass


def date2doy(date_str):
    date = dt.datetime.strptime(date_str, "%Y-%m-%d")
    return date.timetuple().tm_yday


if __name__ == '__main__':
    print("This is a module to download SSUSI data for a given event date.")

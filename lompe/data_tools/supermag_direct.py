from multiprocessing import Pool
from lompe.data_tools.supermag_api import sm_GetUrl, sm_coreurl, sm_keycheck_data
import requests
import certifi
import os
from requests.exceptions import RequestException
import time

# this is a set of helper functions to download data from SuperMAG
# it is adjusted in a way that it downlaods the json temporarily in the tempfiles folder
# read these json files, create a dataframe and write to hdf file and then delete the json files

# is there any better way to do this?
# I think this is better than going to the jhupl webapge and doing it manual clicking and downloading

# it also uses multiprocessing to download data for multiple stations at the same time
# i would like to see how this can be done in a better way

# due to the multiprocessing issues, this should be imported as a module in the datadownloader script

# Note: I have added a retry mechanism to handle the zero byte response from the requests (06 August 2024)
        # using retries and backoff_factor
        # it will retry the request for a number of times before giving up, but i am not sure this is the best way to handle this

def data_download_for_station(args, retries=5, backoff_factor=0.5):
    # DO NOT EDIT THIS FUNCTION
    urlstr, station = args
    url = urlstr + '&station=' + station.upper()
    
    for i in range(retries):
        try:
            response = requests.get(url, verify=certifi.where())
            if response.status_code == 200:
                if response.content:  # Check if the response content is not zero bytes
                    with open(f'./tempfiles/{station}_data.txt', 'wb') as file:
                        file.write(response.content)
                    return None
                else:
                    print(f"Received zero bytes for station {station}")
                    raise RequestException("Received zero bytes.")
            else:
                print(f"Failed to retrieve data for station {station}: {response.status_code}")
                raise RequestException(f"Bad status code: {response.status_code}")
        except RequestException as e:
            print(f"Attempt {i + 1} for station {station} failed with error: {e}")
            time.sleep(backoff_factor * (2 ** i))
    
    print(f"Failed to download data for station {station} after {retries} attempts.")
    return None


# def data_download_for_station(args):
#     # DO NOT EDIT THIS FUNCTION
#     urlstr, station = args
#     url = urlstr + '&station=' + station.upper()
#     response = requests.get(url, verify=certifi.where())

#     if response.status_code == 200:
#         with open(f'./tempfiles/{station}_data.txt', 'wb') as file:
#             file.write(response.content)
#         return None
#     else:
#         print(
#             f"Failed to retrieve data for station {station}: {response.status_code}")
#         return None


def download_data_for_event(start):
    # DONOT EDIT THIS FUNCTION
    os.makedirs('./tempfiles', exist_ok=True)
    duration = 86400  # Duration in seconds (one day)
    # lompe is already registered in API
    urlstr = sm_coreurl('data-api.php', 'lompe', start, 86400)
    # this is hard coded to 'geo', see supermag_api.py to change it according to your needs
    indices = sm_keycheck_data('geo')
    urlstr += indices
    indices = sm_keycheck_data('baseline=daily')
    urlstr += indices

    urlstr_inv = sm_coreurl('inventory.php', 'lompe', start, duration)
    success, stations = sm_GetUrl(urlstr_inv, 'raw')
    stations = stations[1:-1]

    # Create a list of arguments to pass to data_download_for_station
    args_list = [(urlstr, station) for station in stations]

    with Pool() as pool:
        pool.map(data_download_for_station, args_list)


if __name__ == "__main__":
    start_date = '2012-04-05T00:00:00'
    download_data_for_event(start_date)

import os
import gc
import io
import sys
import anvil.server
import anvil.media
import sounderpy as spy
import pandas as pd
import numpy as np
from datetime import datetime as dt
from urllib.request import urlopen
import threading
import time
import requests
import matplotlib.pyplot as plt


# --- SETUP AND CONFIGURATION ---
# fetch api key and uplink to anvil 
UPLLINK_KEY = os.environ.get("ANVIL_API_KEY", None)

if UPLLINK_KEY is None:
    print("FATAL ERROR: ANVIL_API_KEY environment variable not found.")
    sys.exit(1)

anvil.server.connect(UPLLINK_KEY)



# --- DEFINE LOCKING ---
FILE_I_O_LOCK = threading.Lock()


#-----------------------------------------------------------------------------
# RAOB FUNCTION
#-----------------------------------------------------------------------------
@anvil.server.callable
def get_raob_sounding(site_id, year, month, day, hour, color_blind, dark_mode, hodo, storm_motion, modify_sfc, special_parcels, map_zoom, radar, radar_time, hodo_boundary, file, file_type):
    # Lock is critical here because all sounderpy functions write to the same 'sounderpy_sounding.png'
    with FILE_I_O_LOCK: 
        clean_data = spy.get_obs_data(str(site_id), str(year), str(month), str(day), str(hour), hush=True)

        label_txt = (
            f"{clean_data['site_info']['valid-time'][3]}Z RAOB for "
            f"{clean_data['site_info']['site-id']}, {clean_data['site_info']['site-name']} at "
            f"{clean_data['site_info']['valid-time'][1]}-{clean_data['site_info']['valid-time'][2]}-"
            f"{clean_data['site_info']['valid-time'][0]}"
        )

        if file:
            spy.to_file(file_type, clean_data)
            file = anvil.media.from_file("sounderpy_data", "text/plain", name=f'sounderpy_data_{file_type}')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return file

        elif hodo:
            spy.build_hodograph(clean_data, dark_mode=dark_mode, storm_motion=storm_motion, radar=radar, radar_time=radar_time, hodo_boundary=hodo_boundary, save=True, filename='sounderpy_sounding')
            image =anvil.media.from_file('sounderpy_sounding.png', 'image/jpeg')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return image, label_txt

        else:
            spy.build_sounding(clean_data, dark_mode=dark_mode, storm_motion=storm_motion,
                               special_parcels=special_parcels, color_blind=color_blind,
                               save=True, radar=radar, radar_time=radar_time, hodo_boundary=hodo_boundary, map_zoom=map_zoom, modify_sfc=modify_sfc)

            image =anvil.media.from_file('sounderpy_sounding.png', 'image/jpeg')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return image, label_txt

            plt.close('all')



#-----------------------------------------------------------------------------
# ACARS ALL LIST FUNCTION
#-----------------------------------------------------------------------------
@anvil.server.callable
def get_acars_all_profile_list(year, month, day, hour):
    # Lock is not strictly necessary for this function if it doesn't use the plot file
    with FILE_I_O_LOCK: 
        try:
            list_1 = spy.acars_data(str(year), str(month), str(day), str(hour)).list_profiles()
            list_2 = [profile[0:3] for profile in list_1]
        except Exception as e: 
            return ["No profiles found for this date or time"]

        # Fetch the CSV data once (consider caching this if it's slow/large)
        csv_url = 'https://raw.githubusercontent.com/kylejgillett/sounderpy/main/src/AIRPORTS.csv'
        airports_csv = pd.read_csv(csv_url, skiprows=7, skipinitialspace=True)

        list_3 = []
        for arpt, profile in zip(list_2, list_1):
            # Using .index is generally safer/faster than a full np.where/str.contains chain on every iteration
            try:
                where = airports_csv[airports_csv['IATA'].str.contains(arpt, na=False, case=True)].index[0]
                
                name = airports_csv['Name'].iloc[where]
                city = airports_csv['City'].iloc[where]
                
                list_3.append(f'{profile} | {name}, {city}')
            except IndexError:
                # Handle cases where the airport code isn't found
                list_3.append(f'{profile} | Unknown Airport')
                
        return list_3




#-----------------------------------------------------------------------------
# ACARS AIRPORT LIST FUNCTION
#-----------------------------------------------------------------------------
@anvil.server.callable
def get_acars_airport_profile_list(year, month, day, airport):
    with FILE_I_O_LOCK: 
        profiles_list = []
        dates_list = []

        csv_url = 'https://raw.githubusercontent.com/kylejgillett/sounderpy/main/src/AIRPORTS.csv'
        airports_csv = pd.read_csv(csv_url, skiprows=7, skipinitialspace=True)

        # Iterate over hours (0 to 23)
        for hour in range(24):
            hour_str = f'{hour:02d}' # Format as '00', '01', ... '23'

            try:
                acars_conn = spy.acars_data(year, month, day, hour_str)
                acars_list = acars_conn.list_profiles()
                
                # Filter profiles for the specified airport
                p_list = [item for item in acars_list if airport in item]
                
                if p_list:
                    # Append the date/time info for each found profile
                    d_list = [[year, month, day, hour_str] for _ in p_list] 
                    profiles_list.extend(p_list)
                    dates_list.extend(d_list)

            except Exception: # Catch any sounderpy error for that hour
                pass

        # Get airport name/city info
        try:
            where = airports_csv[airports_csv['IATA'].str.contains(airport, na=False, case=True)].index[0]
            airport_info = f"{airports_csv['Name'].iloc[where]}, {airports_csv['City'].iloc[where]}"
        except IndexError:
            airport_info = f"Unknown Airport: {airport}"

        if profiles_list:
            profile_ids = profiles_list
        else:
            profile_ids = ["No profiles found for given date & airport"]

        return profile_ids, profiles_list, dates_list





#-----------------------------------------------------------------------------
# ACARS FUNCTION
#-----------------------------------------------------------------------------
@anvil.server.callable
def get_acars_sounding(profile_id, year, month, day, hour, color_blind, dark_mode, hodo, storm_motion, modify_sfc, special_parcels, map_zoom, radar, radar_time, hodo_boundary, file, file_type):
    with FILE_I_O_LOCK:
        clean_data = spy.acars_data(str(year), str(month), str(day), str(hour)).get_profile(profile_id)

        label_txt = (
            f"{clean_data['site_info']['valid-time'][3]}Z flight from "
            f"{clean_data['site_info']['site-id']}, {clean_data['site_info']['site-name']} at "
            f"{clean_data['site_info']['valid-time'][1]}-{clean_data['site_info']['valid-time'][2]}-"
            f"{clean_data['site_info']['valid-time'][0]}"
        )

        if file:
            spy.to_file(file_type, clean_data)
            file = anvil.media.from_file("sounderpy_data", "text/plain", name=f'sounderpy_data_{file_type}')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return file

        elif hodo:
            spy.build_hodograph(clean_data, dark_mode=dark_mode, storm_motion=storm_motion, radar=radar, radar_time=radar_time, hodo_boundary=hodo_boundary, save=True, filename='sounderpy_sounding')
            image =anvil.media.from_file('sounderpy_sounding.png', 'image/jpeg')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return image, label_txt

        else:
            spy.build_sounding(clean_data, dark_mode=dark_mode, storm_motion=storm_motion,
                               special_parcels=special_parcels, color_blind=color_blind,
                               save=True, radar=radar, radar_time=radar_time, hodo_boundary=hodo_boundary, map_zoom=map_zoom, modify_sfc=modify_sfc)

            image =anvil.media.from_file('sounderpy_sounding.png', 'image/jpeg')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return image, label_txt






#-----------------------------------------------------------------------------
# BUFKIT FUNCTION (Specific Run)
#-----------------------------------------------------------------------------
@anvil.server.callable
def get_bufkit_sounding(model, bufkit_site, fcst_hr, run_year, run_month, run_day, run_hour, color_blind, dark_mode, hodo, storm_motion, modify_sfc, special_parcels, map_zoom, radar, radar_time, hodo_boundary, file, file_type):
    with FILE_I_O_LOCK:
        clean_data = spy.get_bufkit_data(str(model), str(bufkit_site), int(fcst_hr), str(run_year), str(run_month), str(run_day), str(run_hour), hush=True)

        label_txt = (
            f"{clean_data['site_info']['valid-time'][3]}Z forecast for "
            f"{clean_data['site_info']['site-id']}, {clean_data['site_info']['site-name']} at "
            f"{clean_data['site_info']['valid-time'][1]}-{clean_data['site_info']['valid-time'][2]}-"
            f"{clean_data['site_info']['valid-time'][0]}"
        )

        if file:
            spy.to_file(file_type, clean_data)
            file = anvil.media.from_file("sounderpy_data", "text/plain", name=f'sounderpy_data_{file_type}')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return file

        elif hodo:
            spy.build_hodograph(clean_data, dark_mode=dark_mode, storm_motion=storm_motion, radar=radar, radar_time=radar_time, hodo_boundary=hodo_boundary, save=True, filename='sounderpy_sounding')
            image =anvil.media.from_file('sounderpy_sounding.png', 'image/jpeg')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return image, label_txt

        else:
            spy.build_sounding(clean_data, dark_mode=dark_mode, storm_motion=storm_motion,
                               special_parcels=special_parcels, color_blind=color_blind,
                               save=True, radar=radar, radar_time=radar_time, hodo_boundary=hodo_boundary, map_zoom=map_zoom, modify_sfc=modify_sfc)

            image =anvil.media.from_file('sounderpy_sounding.png', 'image/jpeg')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return image, label_txt





#-----------------------------------------------------------------------------
# LATEST BUFKIT FUNCTION
#-----------------------------------------------------------------------------
@anvil.server.callable
def get_latest_bufkit_sounding(model, bufkit_site, fcst_hr, color_blind, dark_mode, hodo, storm_motion, modify_sfc, special_parcels, map_zoom, radar, radar_time, hodo_boundary, file, file_type):
    with FILE_I_O_LOCK:
        clean_data = spy.get_bufkit_data(str(model), str(bufkit_site), int(fcst_hr), hush=True)

        label_txt = (
            f"{clean_data['site_info']['valid-time'][3]}Z forecast for "
            f"{clean_data['site_info']['site-id']}, {clean_data['site_info']['site-name']} at "
            f"{clean_data['site_info']['valid-time'][1]}-{clean_data['site_info']['valid-time'][2]}-"
            f"{clean_data['site_info']['valid-time'][0]}"
        )

    
        if file:
            spy.to_file(file_type, clean_data)
            file = anvil.media.from_file("sounderpy_data", "text/plain", name=f'sounderpy_data_{file_type}')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return file

        elif hodo:
            spy.build_hodograph(clean_data, dark_mode=dark_mode, storm_motion=storm_motion, radar=radar, radar_time=radar_time, hodo_boundary=hodo_boundary, save=True, filename='sounderpy_sounding')

            image = anvil.media.from_file('sounderpy_sounding.png', 'image/jpeg')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return image, label_text


        else:
            spy.build_sounding(clean_data, dark_mode=dark_mode, storm_motion=storm_motion,
                               special_parcels=special_parcels, color_blind=color_blind,
                               save=True, radar=radar, radar_time=radar_time, hodo_boundary=hodo_boundary, map_zoom=map_zoom, modify_sfc=modify_sfc)

            image = anvil.media.from_file('sounderpy_sounding.png', 'image/jpeg')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return image, label_txt






#-----------------------------------------------------------------------------
# REANL FUNCTION
#-----------------------------------------------------------------------------
@anvil.server.callable
def get_reanl_sounding(latlon, year, month, day, hour, color_blind, dark_mode, hodo, storm_motion, modify_sfc, special_parcels, map_zoom, radar, radar_time, hodo_boundary, file, file_type):
    with FILE_I_O_LOCK:
        clean_data = spy.get_model_data('rap-ruc', latlon, str(year), str(month), str(day), str(hour), hush=True)

        label_txt = (
            f"{clean_data['site_info']['valid-time'][3]}Z reanalysis for "
            f"{clean_data['site_info']['site-latlon']} at {clean_data['site_info']['valid-time'][1]}-"
            f"{clean_data['site_info']['valid-time'][2]}-{clean_data['site_info']['valid-time'][0]}"
        )

        if file:
            spy.to_file(file_type, clean_data)
            file = anvil.media.from_file("sounderpy_data", "text/plain", name=f'sounderpy_data_{file_type}')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return file

        elif hodo:
            spy.build_hodograph(clean_data, dark_mode=dark_mode, storm_motion=storm_motion, radar=radar, radar_time=radar_time, hodo_boundary=hodo_boundary, save=True, filename='sounderpy_sounding')
            image =anvil.media.from_file('sounderpy_sounding.png', 'image/jpeg')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return image, label_txt

        else:
            spy.build_sounding(clean_data, dark_mode=dark_mode, storm_motion=storm_motion,
                               special_parcels=special_parcels, color_blind=color_blind,
                               save=True, radar=radar, radar_time=radar_time, hodo_boundary=hodo_boundary, map_zoom=map_zoom, modify_sfc=modify_sfc)

            image =anvil.media.from_file('sounderpy_sounding.png', 'image/jpeg')
            # MEMORY CLEANUP
            plt.close('all')
            del clean_data
            gc.collect()
            return image, label_txt







#-----------------------------------------------------------------------------
# GET MODEL RUNS FUNCTION (Simplified)
#-----------------------------------------------------------------------------
@anvil.server.callable
def get_latest_run():

    # make sure variables are in the correct case
    text_list = []
    station = 'KMOP'

    for model in ['gfs', 'nam', 'namnest', 'rap', 'hrrr', 'sref', 'hiresw']:
        if model == 'gfs':
            model3 = 'gfs3'
        else:
            model3 = model
        data_conn = f'http://www.meteo.psu.edu/bufkit/data/{model.upper()}/{model3}_{station.lower()}.buf'

        # GET BUFKIT FILE
        # CONVERT LINES OF BYTES TO STRINGS
        buf_file = urlopen(data_conn)
        buf_file = [str(line).replace("b'", "").replace("\\r\\n'", "") for line in buf_file]

        # SET UP DATE / TIME OBJECTS FROM THE BUFKIT FILE
        run_time = buf_file[4][buf_file[4].index('TIME') + 7:(buf_file[4].index('TIME')+9)+9]

        text_list.append(f"{str.upper(model)} | Latest run: {run_time[2:4]}/{run_time[4:6]} {run_time[7:9]}Z")

        # final_str = ''
        # for i, string in enumerate(text_list):
        #   final_str += string
        #   if i < len(text_list) - 1:
        #       final_str += "  |  "

    return text_list 

# --- INIT SERVER ---
# sever runs indefinitely, managed by systemd
anvil.server.wait_forever()





# =================================================================
# IMAGE COMPRESSION UTILITY FUNCTION
# =================================================================

# def compressor(file_path="sounderpy_sounding.png", output_name="sounding_sounding.jpeg", quality=75, resize_factor=1.0):

#     print(f"Starting compression for file: {file_path}")

#     try:
#         # 1. Load the original image
#         img = Image.open(file_path)
#         img = img.convert("RGB")

#         # 2. Resize the image (optional, based on resize_factor)
#         if resize_factor < 1.0:
#             width, height = img.size
#             new_size = (int(width * resize_factor), int(height * resize_factor))
#             img = img.resize(new_size, Image.Resampling.LANCZOS)

#         # 3. Save as a compressed JPEG to a memory buffer
#         output_buffer = io.BytesIO()
#         img.save(output_buffer, format='JPEG', quality=quality) 
        
#         # 4. Log successful compression 
#         print("Compression and JPEG conversion completed.")
        
#         # 5. Create the Anvil Media object
#         output_buffer.seek(0)
#         image_media = anvil.media.from_file(output_buffer, 'image/jpeg',name=output_name)

#         return image_media

#     except Exception as e:
#         print(f"An error occurred during image processing: {e}")
#         return None


"""
Freya Glacier Analysis Station Data Loader.

Author: Sonika Shahi
Updated: 25 April 2026

This module provides functions to read and process station data for Freya Glacier analysis, including metadata,
preprocessed station data, and data from RACMO. It also includes a function to read Freya data from PROMICE and
join it with fog classification data for analyzing the change in net radiation during fog events.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

os.chdir('/nas/home/sos/third_paper_work/shahi_et_al_2027_fog/codes/utilities')
from derive_and_correct_aws_variables import calculate_declination, calculate_hour_angle, calculate_zenith

def read_station_metadata(station=None):
    station_metadata_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_fog/data/processed_data/station_data/northeast_stations_metadata.csv'
    stations_data = pd.read_csv(station_metadata_path, encoding='iso-8859-1') # Check encoding if needed on bash file -i northeast_stations_metadata.csv
    stations_data = stations_data.set_index('Stations').T
    stations_data[['Elevation (m a.s.l)', 'lat (N)', 'lon (W)']] = (stations_data[['Elevation (m a.s.l)', 'lat (N)', 'lon (W)']].apply(pd.to_numeric, errors='coerce'))
    stations_data.rename(columns={'Elevation (m a.s.l)': 'elevation', 'lat (N)': 'latitude', 'lon (W)': 'longitude'}, inplace=True)
    if station is None:
        print('Available stations:\n')
        for s in stations_data.index:
            print(s)
        return
    else:
        station_metadata = pd.DataFrame(stations_data.loc[station])
        station_metadata.dropna(inplace=True)
        return station_metadata

def read_preprocessed_station_data(station, variable, column_name, time_freq, rename_column=None):
    """Read preprocessed station data.

    """
    print("This function is deprecated. Please use 'read_freya_data_from_promice' for Freya data and 'read_zackenberg_aws_data' for Zackenberg AWS data instead.")
    processed_aws_data_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')

    station_variable_hour = pd.read_csv(f"{processed_aws_data_path}/{station}_{variable}_{time_freq}.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
    station_variable_hour.index = pd.to_datetime(station_variable_hour.index)
    freq = station_variable_hour.index.to_series().diff().mode()[0]
    station_variable_hour = station_variable_hour.resample(freq).mean()
    station_variable_hour = pd.DataFrame(station_variable_hour[column_name])

    if variable == 'q2m':
        # Specific humidity needs to be converted from kg/kg to g/kg
        station_variable_hour[column_name] = station_variable_hour[column_name] * 1000

    if rename_column is not None:
        station_variable_hour.rename(columns={column_name: rename_column}, inplace=True)
    return station_variable_hour

def read_freya_aws_data(variable=None):
    """Read Freya data from PROMICE.

    """
    # Read Freya data from PROMICE
    freya_data_path = Path('/nas/data/raw/gem/freya/freya_from_promice')
    freya_data = pd.read_csv(f"{freya_data_path}/FRE_hour_promice.csv", parse_dates=['time'], index_col='time')
    freya_data.index = pd.to_datetime(freya_data.index)
    freq = freya_data.index.to_series().diff().mode()[0]
    freya_data = freya_data.resample(freq).mean()

    # Select only variables that exist in your dataset
    vars_in_data = freya_data.columns

    # Read metadata for Freya variables from PROMICE
    freya_variable_metadata = pd.read_csv(f"{freya_data_path}/AWS_variables.csv")
    freya_variable_metadata.index = freya_variable_metadata['field']
    freya_variable_metadata = freya_variable_metadata.loc[vars_in_data]
    # Display key metadata fields
    # freya_variable_metadata = freya_variable_metadata[['standard_name', 'long_name', 'units']]
    freya_variable_metadata = freya_variable_metadata[['field', 'standard_name', 'long_name']]
    freya_variable_metadata = pd.DataFrame(freya_variable_metadata)
    # if variable not in freya_data.columns:
    #     print("Available variables in Freya data from PROMICE:\n")
    #     for var, row in freya_variable_metadata.iterrows():
    #         # print(f"{var:5} → {row['standard_name']} ({row['units']})")
    #         print(f"{var:5} → {row['long_name']}")
    #     raise KeyError(f"Variable '{variable}' not found in Freya data from PROMICE")

    # Calculate declination, zenith, and hour angle of the sun
    date = pd.DataFrame(data=freya_data.index.values, index=freya_data.index, columns=['date'])
    doy = date.index.dayofyear
    hour = date.index.hour
    minute = date.index.minute

    Declination_rad = calculate_declination(doy, hour, minute)                      # Calculate declination
    freya_station_metadata = read_station_metadata(station='FRE2 (promice type, tripod)')
    HourAngle_rad =  calculate_hour_angle(hour, minute, freya_station_metadata.loc['longitude'].values[0])
    _, ZenithAngle_deg = calculate_zenith(freya_station_metadata.loc['latitude'].values[0],
                                                        Declination_rad, HourAngle_rad)

    ZenithAngle_deg = pd.DataFrame(ZenithAngle_deg.values, index=freya_data.index, columns=['zenith'])
    freya_data = freya_data.join(ZenithAngle_deg)

    if variable is None:
        print("Available variables:\n")
        for var, row in freya_variable_metadata.iterrows():
            print(f"{var:15} → {row['long_name']}")
        return
    else:
        return pd.DataFrame(freya_data[variable])

def read_zackenberg_aws_data(variable=None):
    """Read preprocessed Zackenberg AWS data.

    The function 'preprocessed_and_combine_zac_variables' in FG0_preprocess_data.py was used to preprocess the Zackenberg AWS data and combine all variables into a single CSV file.
    """
    zac_data_path = Path('/nas/data/processed/people/sos/third_paper_data/processed_aws')
    zac_data = pd.read_csv(f"{zac_data_path}/zac_aws_hour.csv", parse_dates=['Date'], index_col='Date')
    zac_data.index = pd.to_datetime(zac_data.index)

    if variable is None:
        print(f"Available variables:\n {zac_data.columns}")
    else:
        return pd.DataFrame(zac_data[variable])

# def classify_wind_direction_in_degrees_into_cardinal_direction(wind_direction_in_degree, column_name):
#     """Classify wind direction from degrees into cardinal directions.

#     """
#     if column_name not in wind_direction_in_degree.columns:
#         raise KeyError(f"Station '{column_name}' not found in wind_direction_in_degree")

#     wind_direction_in_degree = pd.DataFrame(data=wind_direction_in_degree[column_name])
#     # Normalize to [0, 360)
#     wd = wind_direction_in_degree[column_name] % 360
#     # # Rename for clarity
#     # wind_direction_in_degree = wind_direction_in_degree.rename(
#     #     columns={column_name: 'wd'})
#     # 16-point compass directions (+ wraparound N)
#     directions = np.array(['N', 'NNE', 'NE', 'ENE',
#                             'E', 'ESE', 'SE', 'SSE',
#                             'S', 'SSW', 'SW', 'WSW',
#                             'W', 'WNW', 'NW', 'NNW', 'N'])
#     # Bin edges: 22.5° sectors centered on each direction
#     bins = np.arange(11.25, 360, 22.5)
#     # Classify
#     wind_direction_in_degree[f"{column_name}_class"] = directions[np.digitize(wd, bins)]
#     # Assign NaN where original data was NaN
#     wind_direction_in_degree.loc[wind_direction_in_degree[column_name].isna(), f"{column_name}_class"] = np.nan

#     return wind_direction_in_degree

def read_fog_classification():
    """Read fog covering Freya Glacier AWS classification.

    """
    # freya_variables = []
    # # df = read_freya_aws_data(variable=None) # Print all available variables
    # # for v in ['dsr_cor', 'cc', 't_u']:
    # for v in variables:
    #     df = read_freya_aws_data(variable=v)
    #     freya_variables.append(df)
    # freya_variables = pd.concat(freya_variables, axis=1).sort_index()

    # # Read Freya AWS metadata
    # # freya_metadata = read_station_metadata() Check all available stations and their metadata
    # freya_metadata = read_station_metadata(station='FRE2 (promice type, tripod)')

    # File path for fog classification data
    ifile_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_fog/data/processed_data/fog_timeseries/'
    fog_class_fog_cover_aws_year_all = []
    results_fog_all = []
    for year in [2016, 2019, 2022]:
        fog_class_fog_cover_aws_year = []
        for month in [5, 6, 7, 8, 9]:
            fog_class = pd.read_csv(f"{ifile_path}fog_on_glacier_{year}_{month}.csv")
            fog_class['date'] = pd.to_datetime(fog_class['date'])
            fog_class.index = fog_class['date']
            fog_class.drop('date', axis=1, inplace=True)

            # Select only the fog cover aws variable
            fog_class_fog_cover_aws = fog_class.loc[:, ['fog_cover_aws']]
            fog_class_fog_cover_aws_year.append(fog_class_fog_cover_aws)
        fog_class_fog_cover_aws_year = pd.concat(fog_class_fog_cover_aws_year).sort_index()
        if year == 2016:
            freq = pd.Timedelta('1h')
        elif year in [2019, 2022]:
            freq = pd.Timedelta('2h')
        fog_class_fog_cover_aws_year = fog_class_fog_cover_aws_year.resample(freq).mean()

        # Find the fog events which lasts more than half a day to ensure the reliability of the analysis of the change in net radiation during the fog events.
        # Create group id whenever fog changes
        fog_class_fog_cover_aws_year['group'] = (fog_class_fog_cover_aws_year['fog_cover_aws'] != fog_class_fog_cover_aws_year['fog_cover_aws'].shift()).cumsum()
        # Keep only fog periods
        fog_groups = fog_class_fog_cover_aws_year[fog_class_fog_cover_aws_year['fog_cover_aws'] == 1].groupby('group')
        # Select fog events that last at least half a day which means at least 12 consecutive 1-hour samples or at least 6 consecutive 2-hour samples, to ensure the reliability of the analysis of the change in net radiation during the fog events.
        fog_events = fog_groups.filter(lambda x: len(x) >= 12 if freq == pd.Timedelta('1h') else len(x) >= 6)
        # fog_events = fog_groups.filter(lambda x: len(x) >= 6)

        # Show indices
        fog_event_indices = [list(g.index) for _, g in fog_events.groupby('group')]
        results_fog = []
        for fog_event_index in fog_event_indices:
            if year == 2016:
                duration = len(fog_event_index) # Duration in hours for 1-hour data
            elif year in [2019, 2022]:
                duration = len(fog_event_index) * 2 # Duration in hours for 2-hour data
            results_fog.append({'duration (hours)': duration,
                                'start_time': fog_event_index[0],
                                'end_time': fog_event_index[-1],
                                'year': fog_event_index[0].year,
                                'month': fog_event_index[0].month})
        results_fog = pd.DataFrame(results_fog)

        # Add columns for start and end intervals of fog events to analyze the change in net radiation during the fog events by comparing the net radiation during the fog events with the net radiation before and after the fog events.
        results_fog['start_time_normalized'] = pd.NaT
        results_fog['end_time_normalized'] = pd.NaT
        results_fog['end_time_normalized_plus_one_day'] = pd.NaT
        for fog_index, fog_start_time, fog_end_time in zip(results_fog.index, results_fog.start_time, results_fog.end_time):
            results_fog.loc[fog_index, 'start_time_normalized'] = fog_start_time.normalize()

            results_fog.loc[fog_index, 'end_time_normalized'] = fog_end_time.normalize() + pd.Timedelta(days=1) - freq
            results_fog.loc[fog_index, 'end_time_normalized_plus_one_day'] = fog_end_time.normalize() + pd.Timedelta(days=1)

        # Compare current start with previous end
        new_event = (results_fog['start_time_normalized'] > results_fog['end_time_normalized_plus_one_day'].shift())
        # First row should always start a new event
        new_event.iloc[0] = True
        # Create event IDs
        results_fog['events'] = new_event.cumsum()
        # Add columns for start and end intervals of fog events
        results_fog['start_interval'] = results_fog.groupby('events')['start_time_normalized'].transform('min')
        results_fog['end_interval'] = results_fog.groupby('events')['end_time_normalized'].transform('max')

        # Add buffer of 7 days before and after the start and end intervals of fog events
        # Merged intervals for fog events with buffer even though it overlaps the fog events separated by more than one day
        # This is done to avoid comparing atmospheric variables during fog days with
        # atmospheric variables during non-fog days that are too far away from the fog events,
        # which may not be representative of the conditions during the fog events and
        # may introduce bias in the analysis of the change in net radiation during the fog events
        # due to the seasonal variations.
        interval_buffer_days = 7
        results_fog['start_interval_buffer'] = results_fog['start_interval'] - pd.Timedelta(days=interval_buffer_days)
        results_fog['end_interval_buffer'] = results_fog['end_interval'] + pd.Timedelta(days=interval_buffer_days)

        # # Merge overlapping intervals if there are multiple fog events in the same month to avoid double counting of the same time period in the analysis of the change in net radiation during the fog events.
        # current_start = results_fog.iloc[0]['start_time_normalized']
        # current_end = results_fog.iloc[0]['end_time_normalized_plus_one_day']
        # for i in range(1, len(results_fog)):
        #     row_i = results_fog.iloc[i]
        #     next_start = row_i['start_time_normalized']
        #     next_end = row_i['end_time_normalized_plus_one_day']

        #     # If intervals overlap OR touch (same day)
        #     # Allow small gap → treat as continuous
        #     if year == 2016:
        #         hours = 1
        #     elif year in [2019, 2022]:
        #         hours = 2
        #     if next_start <= current_end + pd.Timedelta(hours=hours):
        #         current_end = max(current_end, next_end)
        #     else:
        #         merged_intervals.append((current_start, current_end))
        #         current_start = next_start
        #         current_end = next_end
        # # Append last interval after the loop
        # merged_intervals.append((current_start, current_end))

        # # Merged intervals for fog events with buffer even though it overlaps the fog events separated by more than one day
        # # This is done to avoid comparing atmospheric variables during fog days with
        # # atmospheric variables during non-fog days that are too far away from the fog events,
        # # which may not be representative of the conditions during the fog events and
        # # may introduce bias in the analysis of the change in net radiation during the fog events
        # # due to the seasonal variations.
        # for merged_interval in merged_intervals:
        #     merged_start = merged_interval[0]
        #     merged_end = merged_interval[1]

        #     # Add buffer of 7 days before and after the merged interval
        #     buffer_before = merged_start - pd.Timedelta(days=7)
        #     buffer_after = merged_end + pd.Timedelta(days=7)

        #     # Update the merged interval with the buffer
        #     merged_intervals_all.append((buffer_before, buffer_after))

        # -------------------------------------------
        # The following code is an alternative way to add buffer to the start and end intervals of fog events without merging the intervals, which may be useful for analyzing the change in net radiation during the fog events while keeping the original intervals of the fog events separate.
        # However, it may introduce bias in the analysis of the change in net radiation during the fog events due to the seasonal variations if there are multiple fog events in the same month that are separated by more than one day, as it may compare atmospheric variables during fog days with atmospheric variables during non-fog days that are too far away from the fog events, which may not be representative of the conditions during the fog events. Therefore, it is recommended to merge overlapping intervals of fog events with a buffer even if they are separated by more than one day to ensure a more reliable analysis of the change in net radiation during the fog events.
        # -------------------------------------------
        # # Add columns for start and end intervals of fog events to analyze the change in net radiation during the fog events by comparing the net radiation during the fog events with the net radiation before and after the fog events.
        # results_fog['start_interval'] = pd.NaT
        # results_fog['end_interval'] = pd.NaT
        # # Calculate the start and end intervals for each fog event by adding two days before the start time and a day after the end time of the fog event, respectively, to analyze the change in net radiation during the fog events.
        # interval_buffer_days = 7
        # for fog_index, fog_start_time, fog_end_time in zip(results_fog.index, results_fog.start_time, results_fog.end_time):
        #     days_before_fog = fog_start_time.normalize() - pd.Timedelta(days=interval_buffer_days)

        #     start_next_day = fog_end_time.normalize() + pd.Timedelta(days=1)
        #     # one_day_after_fog = start_next_day + pd.Timedelta(days=2) - freq
        #     days_after_fog = start_next_day + pd.Timedelta(days=interval_buffer_days) - freq

        #     results_fog.loc[fog_index, 'start_interval'] = days_before_fog
        #     results_fog.loc[fog_index, 'end_interval'] = days_after_fog
        # results_fog = results_fog.sort_values('start_interval')

        # # Merge overlapping intervals if there are multiple fog events in the same month to avoid double counting of the same time period in the analysis of the change in net radiation during the fog events.
        # current_start = results_fog.iloc[0]['start_interval']
        # current_end = results_fog.iloc[0]['end_interval']
        # for i in range(1, len(results_fog)):
        #     row_i = results_fog.iloc[i]
        #     next_start = row_i['start_interval']
        #     next_end = row_i['end_interval']

        #     # If intervals overlap OR touch (same day)
        #     # Allow small gap → treat as continuous
        #     if year == 2016:
        #         hours = 1
        #     elif year in [2019, 2022]:
        #         hours = 2
        #     if next_start <= current_end + pd.Timedelta(hours=hours):
        #         current_end = max(current_end, next_end)
        #     else:
        #         merged_intervals_all.append((current_start, current_end))
        #         current_start = next_start
        #         current_end = next_end
        # # Append last interval after the loop
        # merged_intervals_all.append((current_start, current_end))
        # -------------------------------------------

        # Append results for that year to the list for all years
        fog_class_fog_cover_aws_year_all.append(fog_class_fog_cover_aws_year)
        results_fog_all.append(results_fog)

    # Concatenate results for all years
    fog_class_fog_cover_aws_year_all = pd.concat(fog_class_fog_cover_aws_year_all)

    results_fog_all = pd.concat(results_fog_all)
    # Reset all the event IDs to be unique across all years by adding an offset to the event IDs of each year, so that the event IDs are unique across all years and can be used to analyze the change in net radiation during the fog events across all years without confusion between events from different years.
    offset = 0
    new_events = []
    for i, e in enumerate(results_fog_all['events']):
        # only trigger when a NEW block starts
        if i > 0 and e == 1 and results_fog_all['events'].iloc[i-1] != 1:
            offset = new_events[-1]
        new_events.append(e + offset)

    results_fog_all['events_fixed'] = new_events

    # # Join fog for that year and freya variables
    # fog_cover_aws_variables = fog_class_fog_cover_aws_year_all.join(freya_variables, how='left')

    # # Merge overlapping intervals of fog events with a buffer even if they are separated by more than one day to ensure a more reliable analysis of the change in net radiation during the fog events, as it may compare atmospheric variables during fog days with atmospheric variables during non-fog days that are too far away from the fog events, which may not be representative of the conditions during the fog events due to the seasonal variations.
    # merged_intervals_all = []
    # for start, end in zip(np.unique(results_fog_all.start_interval_buffer), np.unique(results_fog_all.end_interval_buffer)):
    #     merged_intervals_all.append((start, end))

    # Reset index of results_fog_all to have a unique ID for each row, which can be used for plotting and analysis of the change in net radiation during the fog events across all years without confusion between events from different years.
    results_fog_all['id'] = np.arange(len(results_fog_all))
    results_fog_all.set_index('id', inplace=True)

    # return freya_variables, fog_cover_aws_variables, results_fog_all, merged_intervals_all
    # return freya_variables, results_fog_all
    return results_fog_all

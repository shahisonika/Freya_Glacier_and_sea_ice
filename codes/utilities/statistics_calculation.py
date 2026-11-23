"""
Calculate hourly, daily means based on availability of the data.


Author: Sonika Shahi
Date: 29-08-2025
"""
from pathlib import Path
import pandas as pd
import numpy as np

class scalar_statistics():
    """Calculate statistics of scalar variables based on 80% data availability
    input_data = data in pandas dataframe format.

    """
    @staticmethod
    def hourly_statistics(data, method):
        """ NOTE: This method is DEPRECATED and not used in the current analysis.
        Calculate hourly statistics only if 80% (~4.8 values) of minute values exist.
        The temporal resolution is 10-minute, so 80% of 6 values per hour means at least 5 valid values are needed to calculate the hourly statistic.

        input:
            data: dataframe with datetime index
            method: 'mean', 'median', or 'sum'

        output:
            df_hourly: hourly aggregated DataFrame
        """
        print("This method is DEPRECATED and not used in the current analysis. Please use 'hourly_calculation' instead.")
        # %%
        data = pd.DataFrame(data)
        # Create hourly dataframe
        start = pd.Timestamp(data.index[0])
        end = pd.Timestamp(data.index[-1])
        # Create hourly date range
        hourly_index = pd.date_range(start=start.floor('h'), end=end.floor('h'), freq='h')

        # Create empty DataFrame with this index
        data_hourly = pd.DataFrame(index=hourly_index, columns=data.columns)

        for col in data.columns:
            series = data[[col]].dropna()
            print(f'Calculating hourly {method} of {col} based on 80% data availability')

            # create column to count number of minutes in a common '%Y%m%d %H'
            series['days'] = series.index.strftime('%Y%m%d %H')

            data_count = series.groupby('days').count()
            data_count = pd.DataFrame(
                data={'counts': data_count[col].values},
                index=data_count.index
            )
            data_count.index.name = 'date'
            data_count['days'] = data_count.index
            data_count['freq'] = ((data_count['counts'] / data_count['counts'].max()) * 100).round()

            series['dates'] = series.index
            df = pd.merge(left=series, right=data_count, how='left', left_on='days', right_on='days')
            df.set_index('dates', inplace=True)

            if method == 'mean':
                df_hourly = df[col].where(df['freq'] >= 80).resample('h').mean()
            elif method == 'median':
                df_hourly = df[col].where(df['freq'] >= 80).resample('h').median()
            elif method == 'sum':
                df_hourly = df[col].where(df['freq'] >= 80).resample('h').sum()
            else:
                raise ValueError("method must be 'mean', 'median', or 'sum'")

            data_hourly[col] = pd.DataFrame(df_hourly)

        # Combine results from all columns
        return data_hourly

    @staticmethod
    def hourly_calculation(data, method, max_values_per_hour=6):
        """Calculate hourly statistics only if 80% (~4.8 values) of minute values exist.
        The temporal resolution is 10-minute, so 80% of 6 values per hour means at least 5 valid values are needed to calculate the hourly statistic.

        input:
            data: dataframe with datetime index
            method: 'mean', 'median', or 'sum'

        output:
            df_hourly: hourly aggregated DataFrame
        """
        data = pd.DataFrame(data)

        hourly_data = pd.DataFrame(index=data.resample('h').mean().index, columns=data.columns)
        for col in data.columns:
            series = data[[col]].dropna()
            print(f'Calculating hourly {method} of {col} based on 80% of {max_values_per_hour} values per hour data availability')

            series = data[[col]].dropna()
            valid_counts = series[col].resample('h').count()
            if method == 'mean':
                df_hourly = series[col].resample('h').mean()
            elif method == 'median':
                df_hourly = series[col].resample('h').median()
            elif method == 'sum':
                df_hourly = series[col].resample('h').sum()

            # Apply availability mask - set to NaN if less than 80% of hourly values are available
            df_hourly[valid_counts < max_values_per_hour * 0.8] = np.nan

            hourly_data[col] = pd.DataFrame(df_hourly)

        # Combine results from all columns
        return hourly_data

    @classmethod
    def daily_statistics(cls, data, method):
        """NOTE: This method is not used in the current analysis.
        Calculate daily statistics only if 80% (~19 values) of hourly values exist.

        input:
            data: dataframe with datetime index
            method: 'mean', 'median', or 'sum'

        output:
            df_daily: daily aggregated DataFrame
        """
        print("This method is DEPRECATED and not used in the current analysis. Please use 'daily_calculation' instead.")
        # Calculate hourly first
        data = cls.hourly_statistics(data, method)

        # Create hourly dataframe
        start = pd.Timestamp(data.index[0])
        end = pd.Timestamp(data.index[-1])
        # Create hourly date range
        daily_index = pd.date_range(start=start.floor('D'), end=end.floor('D'), freq='D')

        # Create empty DataFrame with this index
        data_daily = pd.DataFrame(index=daily_index, columns=data.columns)

        for col in data.columns:
            series = data[[col]].dropna()
            print(f'Calculating daily {method} of {col} based on 80% data availability')

            series['days'] = series.index.strftime('%Y%m%d')

            data_count = series.groupby('days').count()
            data_count = pd.DataFrame(
                data={'counts': data_count[col].values},
                index=data_count.index
            )
            data_count.index.name = 'date'
            data_count['days'] = data_count.index
            data_count['freq'] = ((data_count['counts'] / data_count['counts'].max()) * 100).round()

            series['dates'] = series.index
            df = pd.merge(series, data_count, on='days', how='left')
            df.set_index('dates', inplace=True)

            if method == 'mean':
                df_daily = df[col].where(df['freq'] >= 80).resample('D').mean()
            elif method == 'median':
                df_daily = df[col].where(df['freq'] >= 80).resample('D').median()
            elif method == 'sum':
                df_daily = df[col].where(df['freq'] >= 80).resample('D').sum()
            else:
                raise ValueError("method must be 'mean', 'median', or 'sum'")

            data_daily[col] = pd.DataFrame(df_daily)

        # Combine results from all columns
        return data_daily

    @staticmethod
    def daily_calculation(data, method, max_values_per_day=24):
        """Calculate daily statistics only if 80% (~19 values) of hourly values exist.

        input:
            data: dataframe with datetime index
            method: 'mean', 'median', or 'sum'

        output:
            daily_data: daily aggregated DataFrame
        """
        # Calculate hourly first
        data = pd.DataFrame(data)

        daily_data = pd.DataFrame(index=data.resample('D').mean().index, columns=data.columns)
        for col in data.columns:
            series = data[[col]].dropna()
            print(f'Calculating daily {method} of {col} based on 80% of {max_values_per_day} hours data availability')

            series = data[[col]].dropna()
            valid_counts = series[col].resample('D').count()
            if method == 'mean':
                df_daily = series[col].resample('D').mean()
            elif method == 'median':
                df_daily = series[col].resample('D').median()
            elif method == 'sum':
                df_daily = series[col].resample('D').sum()

            # Apply availability mask - set to NaN if less than 80% of hourly values are available
            df_daily[valid_counts < max_values_per_day * 0.8] = np.nan

            daily_data[col] = pd.DataFrame(df_daily)

        # Combine results from all columns
        return daily_data

class vector_statistics(scalar_statistics):
    """ Vector averaging of wind speed and direction.

    Reference:
    https://www.researchgate.net/publication/262766424_Technical_note_Averaging_wind_speeds_and_directions

    """
    @classmethod
    def wind_direction_hourly_mean(cls, ws, wd):
        """ NOTE: This method is not used in the current analysis.

        Calculate hourly mean of wind direction only if 80%(~19 values) of the minute values exist

        """
        print("This method is DEPRECATED and not used in the current analysis. Please use 'calculate_wind_direction_hourly_mean' instead.")
        ws = pd.DataFrame(ws)
        wd = pd.DataFrame(wd)

        u_wind = pd.DataFrame(-1 * (ws.iloc[:, 0] * np.sin(np.deg2rad(wd.iloc[:, 0]))))
        #u_wind = pd.DataFrame(-1*(ws*np.sin(np.deg2rad(wd))))
        u_wind.rename(columns={u_wind.columns[0]: 'u'}, inplace=True)
        u_hr = cls.hourly_statistics(u_wind, method='mean')

        v_wind = pd.DataFrame(-1 * (ws.iloc[:, 0] * np.cos(np.deg2rad(wd.iloc[:, 0]))))
        #v_wind = pd.DataFrame(-1*(ws*np.cos(np.deg2rad(wd))))
        v_wind.rename(columns={v_wind.columns[0]: 'v'},inplace=True)
        v_hr = cls.hourly_statistics(v_wind, method='mean')

        print('Calculating hourly mean of wind direction based on 80 percentage of data availability')
        wd_mean = np.asarray(180 + np.rad2deg(np.arctan2(np.asarray(u_hr), np.asarray(v_hr))))
        wd_mean = np.mod(wd_mean, 360)
        wd_mean = pd.DataFrame(wd_mean, index=u_hr.index)
        wd_mean.rename(columns={wd_mean.columns[0]: 'wd'}, inplace=True)
        wd_mean = wd_mean.replace('NAN', np.nan)

        return wd_mean

        # directions = np.array('N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW N NAN'.split())
        # bins = np.arange(11.25, 372, 22.5)
        # wd_avg['class'] = directions[np.digitize(wd_avg['wd'], bins)]

    @classmethod
    def wind_direction_daily_mean(cls, ws, wd):
        """ NOTE: This method is not used in the current analysis.
        Calculate daily mean of wind direction.

        Reference:
        https://www.researchgate.net/publication/262766424_Technical_note_Averaging_wind_speeds_and_directions
        """
        print("This method is DEPRECATED and not used in the current analysis. Please use 'calculate_wind_direction_daily_mean' instead.")

        ws = pd.DataFrame(ws)
        wd = pd.DataFrame(wd)

        u_wind = pd.DataFrame(-1 * (ws.iloc[:, 0] * np.sin(np.deg2rad(wd.iloc[:, 0]))))
        u_wind.rename(columns={u_wind.columns[0]: 'u'}, inplace=True)
        u_daily = cls.daily_statistics(u_wind, method='mean')

        v_wind = pd.DataFrame(-1 * (ws.iloc[:, 0] * np.cos(np.deg2rad(wd.iloc[:, 0]))))
        v_wind.rename(columns={v_wind.columns[0]: 'v'}, inplace=True)
        v_daily = cls.daily_statistics(v_wind, method='mean')

        print('Calculating daily mean of wind direction based on 80 percentage of data availability')
        wd_mean = np.asarray(180 + np.rad2deg(np.arctan2(np.asarray(u_daily), np.asarray(v_daily))))
        wd_mean = np.mod(wd_mean, 360)
        wd_mean = pd.DataFrame(wd_mean,index=u_daily.index)
        wd_mean.rename(columns={wd_mean.columns[0]: 'wd'}, inplace=True)
        wd_mean = wd_mean.replace('NAN', np.nan)
        return wd_mean

    @classmethod
    def calculate_wind_direction_hourly_mean(cls, ws, wd, max_values_per_hour=6):
        """Calculate hourly wind direction only if 80% (~4.8 values) of minute values exist.
        The temporal resolution is 10-minute, so 80% of 6 values per hour means at least 5 valid values are needed to calculate the hourly statistic.

        input:
            ws: dataframe with wind speed values
            wd: dataframe with wind direction values
            max_values_per_hour: maximum number of values per hour (default: 6)

        output:
            df_hourly: hourly aggregated DataFrame
        """
        # Extract first column as Series
        ws = pd.DataFrame(ws)
        ws = ws.iloc[:, 0]

        wd = pd.DataFrame(wd)
        wd = wd.iloc[:, 0]

        # Convert to u, v (meteorological convention)
        u = -ws * np.sin(np.deg2rad(wd))
        v = -ws * np.cos(np.deg2rad(wd))

        # Daily mean of components with 80 % availability check
        u_hourly = cls.hourly_calculation(u, method='mean', max_values_per_hour=max_values_per_hour)
        v_hourly = cls.hourly_calculation(v, method='mean', max_values_per_hour=max_values_per_hour)

        # --- Compute direction ---
        wd_mean = (np.degrees(np.arctan2(-u_hourly, -v_hourly)) + 360) % 360

        # Optional: mask calm winds
        speed_hourly = np.sqrt(u_hourly**2 + v_hourly**2)
        wd_mean[speed_hourly <= 0] = np.nan
        wd_mean = pd.DataFrame(wd_mean, index=u_hourly.index)
        wd_mean.rename(columns={wd_mean.columns[0]: 'wd'}, inplace=True)
        return wd_mean

    @classmethod
    def calculate_wind_direction_daily_mean(cls, ws, wd, max_values_per_day=24):
        """Calculate daily mean wind direction using vector averaging.

        Requires at least 80% (~19/24) valid hourly values per day.

        Reference:
        https://www.researchgate.net/publication/262766424_Technical_note_Averaging_wind_speeds_and_directions
        """
        # Extract first column as Series
        ws = pd.DataFrame(ws)
        ws = ws.iloc[:, 0]

        wd = pd.DataFrame(wd)
        wd = wd.iloc[:, 0]

        # Convert to u, v (meteorological convention)
        u = -ws * np.sin(np.deg2rad(wd))
        v = -ws * np.cos(np.deg2rad(wd))

        # Daily mean of components with 80 % availability check
        u_daily = cls.daily_calculation(u, method='mean', max_values_per_day=max_values_per_day)
        v_daily = cls.daily_calculation(v, method='mean', max_values_per_day=max_values_per_day)

        # --- Compute direction ---
        wd_mean = (np.degrees(np.arctan2(-u_daily, -v_daily)) + 360) % 360

        # Optional: mask calm winds
        speed_daily = np.sqrt(u_daily**2 + v_daily**2)
        wd_mean[speed_daily <= 0] = np.nan
        wd_mean = pd.DataFrame(wd_mean, index=u_daily.index)
        wd_mean.rename(columns={wd_mean.columns[0]: 'wd'}, inplace=True)

        return wd_mean

    @staticmethod
    def calculate_wind_speed_direction_from_u_v(u, v):
        """Calculate wind speed and direction from zonal (u) and meridional (v) wind components.

        Parameters:
            u : zonal wind (m/s, + eastward)
            v : meridional wind (m/s, + northward)

        Returns:
            speed (same shape): wind speed (m/s)
            direction (same shape): wind direction (degrees, from which wind blows)
        """
        u = pd.DataFrame(u)
        u = u.iloc[:, 0]

        v = pd.DataFrame(v)
        v = v.iloc[:, 0]

        # Wind direction (meteorological: from where it comes)
        direction = (np.degrees(np.arctan2(-u, -v)) + 360) % 360

        # Wind speed
        speed = np.sqrt(u**2 + v**2)

        # Optional: mask calm winds
        direction[speed <= 0] = np.nan

        direction = pd.DataFrame(direction, columns=['wd'], index=u.index)
        speed = pd.DataFrame(speed, columns=['ws'], index=u.index)

        return speed, direction

    @staticmethod
    def classify_wind_direction_in_degrees_into_cardinal_direction(wind_direction_in_degree, column_name):
        """Classify wind direction from degrees into cardinal directions.

        """
        if column_name not in wind_direction_in_degree.columns:
            raise KeyError(f"Station '{column_name}' not found in wind_direction_in_degree")

        wind_direction_in_degree = pd.DataFrame(data=wind_direction_in_degree[column_name])
        # Normalize to [0, 360)
        wd = wind_direction_in_degree[column_name] % 360
        # # Rename for clarity
        # wind_direction_in_degree = wind_direction_in_degree.rename(
        #     columns={column_name: 'wd'})
        # 16-point compass directions (+ wraparound N)
        directions = np.array(['N', 'NNE', 'NE', 'ENE',
                                'E', 'ESE', 'SE', 'SSE',
                                'S', 'SSW', 'SW', 'WSW',
                                'W', 'WNW', 'NW', 'NNW', 'N'])
        # Bin edges: 22.5° sectors centered on each direction
        bins = np.arange(11.25, 360, 22.5)
        # Classify
        wind_direction_in_degree[f"{column_name}_class"] = directions[np.digitize(wd, bins)]
        # Assign NaN where original data was NaN
        wind_direction_in_degree.loc[wind_direction_in_degree[column_name].isna(), f"{column_name}_class"] = np.nan

        return wind_direction_in_degree
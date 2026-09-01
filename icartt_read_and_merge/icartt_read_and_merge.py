"""Set of Functions that allow you to read in Icartt Files and Merge them."""
#  DESCRIPTION
#
#     This script provides all the functionality you need to read in
#     ICARTT data as a pandas dataframe & save it as a pickle for
#     quick usage in python scripts. Options are provided to merge
#     data from multiple icartt files into a single file, and to
#     remap the time averaging.
#
#  NOTES
#
#  * parser is based on icartt v2.0 spec. most convenient source, imo:
#    https://www-air.larc.nasa.gov/missions/etc/IcarttDataFormat.htm
#
#  * and spec reference on earthdata;s page:
#    https://cdn.earthdata.nasa.gov/conduit/upload/6158/ESDS-RFC-029v2.pdf
#
#  ACKNOWLEDGEMENTS
#    This modeule was modified from a really nice & much fancier module
#    (ornldaac_icartt_to_netcdf) written by mcnelisjj@ornl.gov
#    that was intended to convert ICARTT v2 files into netCDF.
#    Thanks to them for doing the bulk of the work
#
#  WRITTEN BY:
#    Dr. Jessica D.Haskins (jhaskins@alum.mit.edu)

import os
import re
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import mpu
from multiprocessing import Pool


def _warn(message: str = "( miscellaneous warning )"):
    """Print the input error message and exits the script with a failure."""
    # Args: message (str): warning message gets printed during eval and ignored
    print("   WARN: {}. Skipping.".format(message))


def _exit_with_error(message=str):
    """Print the input error message and exits the script with a failure."""
    sys.exit(print("ERROR: {} -- Abort.".format(message)))


def _crawl_directory(path: str, extension: str = None):
    """Crawl an input directory for a list of ICARTT files.\
     Parameters:------------------\
        path (str): full path to an input directory.\
        ext (str): An optional extension to limit search.\
    Returns:  A list of paths to data files (strings)."""
    selected_files = []  # Create an empty list
    
    # Default to .ict extension for ICARTT files if no extension specified
    if extension is None:
        extension = '.ict'
    
    # Files to exclude (system files, etc.)
    excluded_files = {'.DS_Store', 'Thumbs.db', '.pkl', '.pickle'}

    for root, dirs, files in os.walk(path):  # Walk directory.
        for f in files:  # Loop over files,
            fext = os.path.splitext(f)[1].lower()  # Get the extension (lowercase).
            
            # Skip excluded files
            if f in excluded_files or fext in excluded_files:
                continue

            # If file matches input extension
            if extension.lower() == fext:
                # Join to root for the full path.
                fpath = os.path.join(root, f)
                # Add to list.
                selected_files.append(fpath)

    # Return the complete list.
    return selected_files


def _filter_files_by_date_range(icartt_files: list, start_date_str: str,
                                 end_date_str: str):
    """Filter ICARTT files to only include those within the date range.

    Parameters:
        icartt_files: List of ICARTT file paths
        start_date_str: Start date string in format 'YYYY-MM-DD HH:MM:SS'
        end_date_str: End date string in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        Filtered list of ICARTT file paths
    """
    # Convert start and end date strings to datetime objects (just the date part)
    start_date = pd.to_datetime(start_date_str).date()
    end_date = pd.to_datetime(end_date_str).date()

    filtered_files = []
    skipped_files = []

    for ict in icartt_files:
        # Extract date from filename (format: YYYYMMDD)
        date_match = re.search(r'\d{8}', ict)

        if date_match is None:
            # If no date found in filename, include it (be conservative)
            _warn(f"No date found in filename '{os.path.basename(ict)}', including anyway")
            filtered_files.append(ict)
            continue

        # Parse the date from the filename
        date_str = date_match.group(0)
        try:
            file_date = datetime.datetime.strptime(date_str, '%Y%m%d').date()

            # Check if file date is within range
            if start_date <= file_date <= end_date:
                filtered_files.append(ict)
            else:
                skipped_files.append(ict)
        except ValueError:
            # If date parsing fails, include the file (be conservative)
            _warn(f"Could not parse date '{date_str}' in filename '{os.path.basename(ict)}', including anyway")
            filtered_files.append(ict)

    # Print summary of filtering
    if skipped_files:
        print(f" - Skipped {len(skipped_files)} file(s) outside date range:")
        for f in skipped_files:
            print(f"   - {os.path.basename(f)}")

    return filtered_files


def _organize_standard_and_multileg_flights(DATA: dict):
    """Organize the Multileg flights & parse them."""
    # A regular expression catches the multi leg flight suffix.
    multileg_regex = re.compile('_L[0-9].ict')

    # A dictionary stores the output filename and legs as child list.
    flights = {}

    for ict in DATA['ICARTT_FILES']:  # Loop over all files in the directory.
        # If regular expression is not matched anywhere in string,
        if re.search(multileg_regex, ict) is None:
            # Add to list of "standard" flights (e.g. not a leg)
            flights[ict] = ict

        else:  # Else if regular expression is matched in string.
            # The output file won't have the suffix.
            output_filename = ict[:-7] + ".ict"

            # Add this file to the dict of multi-leg flights.
            if output_filename not in flights:
                flights[output_filename] = [ict]
            else:
                flights[output_filename].append(ict)

    return flights  # Return the organized flights as a dictionary.


# Wind direction is a compass bearing in degrees (0-360). Matches 'WDIR',
# 'Wind_Direction', 'WindDir', 'wind_dir', with or without an instrument-name
# prefix. Deliberately does NOT match Latitude/Longitude/Pitch/Roll/Heading.
_WIND_DIR_RE = re.compile(r'(?:^|_)(?:WDIR|wind[ _]?dir(?:ection)?)', re.IGNORECASE)


def _wind_direction_cols(columns):
    """Columns holding a wind-direction bearing (must be vector-averaged)."""
    return [c for c in columns if _WIND_DIR_RE.search(str(c))]


def align2master_timeline(df: pd.DataFrame, startdt: str, enddt: str,
                          step_S: int, quiet: bool = True,
                          lim: int = None, datetime_index: bool = False,
                          tzf: str = 'UTC'):
    """Resample dataframes to appropriate timelines.

    Wind-direction columns are resampled as unit vectors (cos/sin components
    averaged or interpolated, then recombined with atan2) so the 0/360 wrap
    never corrupts the result — i.e. wind direction is vector-averaged, not
    scalar-averaged. All other columns keep the original scalar behaviour.
    """
    # Function to take a dataframe and appropriately remap it to a new time
    # index, considering the native sampling frequency as it is relative
    # to the desired new time. Writte 2/6/21, jessica d. haskins

    # --------------------------Inputs:-------------------------------------
    # df - dataframe, mustcontain column 'datetime'.
    # startdt, enddt = '2006-03-01 00:00:00', '2006-04-01 00:00:00'
    # step_S= 120 averaging step in seconds (120 for a 2 minute average).
    # quiet - Set to False to show sanity check on averaging.
    # lim -manually set the limit of # of points to include in an avg.
    # datetime index - Set to True if datetime is already an index of the df.
    if (datetime_index is False) and 'datetime' not in df:
        _exit_with_error(("Dataframe passed to align2master_timeline()",
                         "does not contain a column 'datetime'. "))

    # Make datetime an index and remove duplicates.
    if datetime_index is False:
        df = df[df['datetime'].notna()]  # if datetime is nan drop whole row.
        df = df.set_index('datetime')  # Make the datetime an index.
    df = df[~df.index.duplicated()]  # remove any duplicates rows

    # Wind direction is a compass bearing: averaging or interpolating it as a
    # scalar corrupts values across the 0/360 discontinuity (e.g. mean of 350
    # and 10 deg is 0, not 180). Convert each such column to cos/sin components
    # so the generic resampling below acts on continuous quantities; they are
    # recombined with atan2 afterward, which is exactly a unit-vector (circular)
    # average of the direction.
    wind_dir_cols = _wind_direction_cols(df.columns)
    orig_col_order = list(df.columns)
    for c in wind_dir_cols:
        rad = np.deg2rad(pd.to_numeric(df[c], errors='coerce'))
        df = df.assign(**{f'{c}__windcos': np.cos(rad),
                          f'{c}__windsin': np.sin(rad)})
    if wind_dir_cols:
        df = df.drop(columns=wind_dir_cols)

    # Some archived files carry out-of-order or duplicated timestamps
    # (seen in SEAC4RS); reindex/fill below require a monotonic unique index
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep='first')]

    # Get the average native sampling frequency in total seconds:
    tseries = df.index.to_series()
    mean_sep_s = tseries.diff().mean().total_seconds()
    if not np.isfinite(mean_sep_s) or mean_sep_s <= 0:
        # degenerate file (0-1 usable rows, or duplicated timestamps): any
        # grid spacing works for the few values present
        mean_sep_s = 1.0
    min_sep = int(np.round(mean_sep_s))

    if quiet is False:
        print('Native Mean Time Sep. (s): ', str(min_sep) + 's')

    # Clip the working time window to the actual data range + buffer.
    # This avoids creating million-row grids when the master timeline spans
    # weeks but each file only has a few hours of data.
    # Buffer must cover the interpolation reach (lim * min_sep seconds,
    # default ~4350s = 72.5 min) so edge values are computed identically
    # to the unclipped case.
    data_min = df.index.min()
    data_max = df.index.max()
    lim_seconds = lim * mean_sep_s if lim else 4350
    buffer = pd.Timedelta(seconds=max(lim_seconds, 300))  # at least 5 min
    full_start = pd.Timestamp(startdt, tz=tzf)
    full_end = pd.Timestamp(enddt, tz=tzf)
    # Floor/ceil to integer seconds so the clipped grid stays aligned with
    # the full grid (which starts on an integer second).
    clip_start = max(full_start, (data_min - buffer).floor('s'))
    clip_end = min(full_end, (data_max + buffer).ceil('s'))

    # If the native time seperation is less than X seconds, you'll
    # reindex it to our full date range, as close to native  freq as you can,
    # then take a roliing avg to get the X second avg on the time base we want
    if min_sep < step_S:
        if min_sep >= 1:
            native_freq = str(min_sep) + 's'
        else:
            # Sub-second native data (e.g. 20-25 Hz cloud probes): build the
            # near-native grid in milliseconds; rounding to whole seconds
            # would give an invalid '0s' frequency.
            native_freq = str(max(1, int(np.round(mean_sep_s * 1000)))) + 'ms'
        dts = pd.date_range(clip_start, clip_end, freq=native_freq, tz=tzf)

        if not lim:
            # don't fill if collect > than 1H out
            lim = np.round(4350 / max(mean_sep_s, 1e-3))

        dfn = df.reindex(dts, method='nearest', fill_value=np.nan,
                         limit=int(lim))

        # Take a centered boxcar average around the X s avg.
        df_new = dfn.rolling(str(int(step_S)) + 's').mean().resample(str(step_S) + 's').mean()
    else:
        # The native sampling frequency is longer than X seconds (min_sep >= step_S)
        # Use linear interpolation to place values on the desired time grid based on midpoint times
        print(('WARNING: You have input an averaging frequency that is LESS'),
              ('than this instruments average native sampling frequency.'),
              ('Using linear interpolation to fill values at the desired time step.'))
        dts = pd.date_range(clip_start, clip_end, freq=str(step_S) + 's', tz='UTC')
        if not lim:
            # Don't fill if collected > than 1H 15 mins out from here.
            lim = np.round(4350 / min_sep)
            if lim <= 0:
                lim = 1  # If lim not set and too small, just use 1 point.

        # Combine original index with target timeline to preserve original data points
        # This allows interpolation to work between actual data points
        combined_index = dts.union(df.index).sort_values()
        df_combined = df.reindex(combined_index)

        # Use linear interpolation, but only between valid (non-NaN) points
        # The limit parameter ensures we don't interpolate across large gaps
        # limit_direction='both' allows interpolation forward and backward
        df_interp = df_combined.interpolate(method='linear', limit=int(lim), limit_direction='both')

        # Select only the target timeline points
        df_new = df_interp.reindex(dts)

        # Important: pandas interpolate() will interpolate through NaN values if the gap
        # is within the limit. We need to ensure that if a target point would require
        # interpolation between two original data points where one or both are NaN (bad data),
        # the result remains NaN.
        #
        # The limit parameter in interpolate() handles distance, but we still need to check
        # if the bounding original data points are NaN (bad data flags).

        # Create a mask for valid interpolated values (vectorized via searchsorted).
        # For each target time we need to know if both bounding original data
        # points are non-NaN.  The previous per-target-time loop was O(N*M*K);
        # this version is O(N*log(M) + N*K) where N=target pts, M=original pts,
        # K=columns — orders-of-magnitude faster for 1-second output grids.
        orig_times_ns = np.asarray(df.index.astype(np.int64))
        target_times_ns = np.asarray(dts.astype(np.int64))

        # insert_pos[i] = index of first original time STRICTLY > target[i]
        insert_pos = np.searchsorted(orig_times_ns, target_times_ns, side='right')

        # Out-of-range: no original point before (0) or no point after (n)
        n_orig = len(orig_times_ns)
        out_of_range = (insert_pos == 0) | (insert_pos >= n_orig)

        # Clipped indices for safe array access (out-of-range rows handled by mask)
        before_idx_arr = np.clip(insert_pos - 1, 0, n_orig - 1)
        after_idx_arr  = np.clip(insert_pos,     0, n_orig - 1)

        # Numeric values at bracketing original points for every target time
        df_vals = df.values.astype(float)          # (n_orig, n_cols)
        vals_before = df_vals[before_idx_arr, :]   # (n_target, n_cols)
        vals_after  = df_vals[after_idx_arr,  :]

        # Invalid if out-of-range OR either bounding value is NaN
        invalid = (out_of_range[:, np.newaxis] |
                   np.isnan(vals_before) |
                   np.isnan(vals_after))

        valid_mask = pd.DataFrame(~invalid, index=dts, columns=df.columns)

        # Set invalid interpolated values to NaN
        df_new = df_new.where(valid_mask)

    # Reindex to the full master timeline. Regions outside the clipped
    # window are filled with NaN, which is correct — no data existed there.
    full_dts = pd.date_range(full_start, full_end, freq=str(step_S) + 's', tz=tzf)
    if not df_new.index.equals(full_dts):
        df_new = df_new.reindex(full_dts)

    # Recombine wind-direction unit vectors back into a bearing in [0, 360).
    # atan2 of the resampled (cos, sin) is the circular mean in the averaging
    # branch and a wrap-safe interpolation in the interpolation branch.
    for c in wind_dir_cols:
        cc, ss = f'{c}__windcos', f'{c}__windsin'
        if cc in df_new.columns and ss in df_new.columns:
            df_new[c] = np.degrees(np.arctan2(df_new[ss], df_new[cc])) % 360.0
            df_new = df_new.drop(columns=[cc, ss])
    if wind_dir_cols:
        df_new = df_new.reindex(
            columns=[col for col in orig_col_order if col in df_new.columns])

    # Plot the Original Data & the Re- Mapped stuff so you can see if its good:
    if quiet is False:
        one = df.columns[0]
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.plot(df[one], label="original",
                color='orange', linewidth=1)
        ax.plot(df_new[one],
                label="re-mapped", color='blue', linewidth=1)
        # ax.set_xlim(df.index[200], df.index[400])
        # ax.set_ylim(34, 39)
        ax.legend()
        plt.show()

    return df_new


def _find_datelike_cols(df: pd.DataFrame, icartt_file: str,
                        quiet: bool = True):
    """Identify the date like columns in a dataframe from an icartt file."""
    # List of partial Strings to search columnNames for that will ID them as
    # time cols. ***NOTE:  If you are getting a persistent error about this
    # function, try adding to the list of tm_names. E.g. add other timezone
    # denotations. Setting quiet to False will print your col names and
    # what it is finding so you can decide what to add to this list.
    tm_names = ['utc', 'cst', 'cdt', 'local', 'lst', 'est', 'pst',
                'gmt', 'lt', 'time', 'time_mid', 'central_standard',
                'eastern_standard', 'pacific_standard']

    print('All Column Names:', df.columns) if quiet is False else None

    # Identify all the names of time related columns in the dataframe.
    # Use word-boundary matching: require tm_name patterns to appear as
    # complete tokens (bounded by '_' or string start/end). This prevents
    # false positives like "altitude" matching "lt" or "static" matching "est".
    times = list()  # Empty list to contain columns with time-like names.
    for col in df.columns:
        col_lower = col.lower()
        if any(re.search(r'(^|_)' + re.escape(nm) + r'(_|$)', col_lower) for nm in tm_names):
            times.append(col_lower)  # fill the list with those names.
            # Rename all time cols in lowercase to make string matches easier
            if col != col_lower:  # Only rename if different
                df.rename(columns={col: col_lower}, inplace=True)

    # Make sure you haven't grabbed a day column for time by accident. Drop it.
    times = [t for t in times if 'day' not in t]
    
    # Ensure all times in the list actually exist in the dataframe
    times = [t for t in times if t in df.columns]

    print('Original Time Columns Found:', times) if quiet is False else None
    # Return the dataframe with lowercase time names, a list of the time
    # columns found, and a list of time strings you searched for in the col
    # names (this is just for consistency when looping...)
    return df, times, tm_names


def _make_time_midpoint_cols(df: pd.DataFrame, tm_names: list, times: list,
                             quiet: bool = True):
    """Take start/stop times from datelike cols and turn them into midpts."""
    # Get start/stop pairs for time to make a midpoint if you can find both.

    # Create a dict for time names & their ID'd "type" for scanning later.
    nn_times = {}
    for j in range(0, len(tm_names)):  # Check each time zone name
        start_j = None  # reset on each loop through a dif time_nm
        stop_j = None
        has_tm = None
        for i in range(0, len(times)):  # Check each time col name.
            # Assign start/stop pair variables.
            if (tm_names[j] in times[i]) and ('start' in times[i]):
                start_j = times[i]

            if (tm_names[j] in times[i]) and \
               ('stop' in times[i] or 'end' in times[i]):
                stop_j = times[i]

            # Or just let us know if this "time" col exists.
            if (tm_names[j] in times[i]):
                has_tm = times[i]

        # You found a start/stop pair for this time name.
        if (start_j is not None) and (stop_j is not None):
            # Ensure both start and stop columns exist in dataframe
            if start_j not in df.columns or stop_j not in df.columns:
                if quiet is False:
                    print(f"Warning: Start/stop pair ({start_j}, {stop_j}) not found in dataframe. Skipping midpoint creation.")
                continue
            
            # Make a new column in the data frame that is the midpoint.
            midpoint_name = tm_names[j] + '_mid'
            df[midpoint_name] = (df[start_j] + df[stop_j]) / 2

            # Remove the start & stop pair of time from the larger df
            df = df.drop(columns=[start_j, stop_j])

            # Update list of column names with times to reflect above.
            # Only add to times if the column was successfully created
            if midpoint_name in df.columns:
                # Remove old start/stop columns from times list first
                if start_j in times:
                    times.remove(start_j)
                if stop_j in times:
                    times.remove(stop_j)
                # Only add midpoint if it's not already in the list (avoid duplicates)
                if midpoint_name not in times:
                    times.append(midpoint_name)  # add midpoint name

        elif has_tm is not None:
            # Populate dict associating its actual timename and its "type" so
            # we can choose between times to use as master later if we want.
            nn_times[has_tm] = 'time_' + tm_names[j]

            # Didn't find a start/stop pair but do have a col with this tm nm.
            # df.rename(columns={has_tm: 'time_' + tm_names[j]},
            #          inplace=True)  # rename it so we know what it is called

            # Update list of column names with times to reflect above.
            # times.append('time_' + tm_names[j])  # add new name
            # times.remove(has_tm)  # remove old name

    # Filter times list to only include columns that actually exist in dataframe
    times = [t for t in times if t in df.columns]
    
    print('Time cols After Mid_Point Assign:',
          times) if quiet is False else None

    return df, times, tm_names, nn_times


def _pick_a_single_time_col(times: list, nn_times: dict, quiet: bool = True):
    """Decide which time column you prefer to use a indx if you got lots."""
    # Validate input
    if not times or len(times) == 0:
        _exit_with_error('_pick_a_single_time_col called with empty times list. '
                        'This should not happen - time columns should be validated before calling this function.')
    
    # Create dictionary for "picking" which time variable we prefer to use.
    # Change the rank for preferences here. Lower = more desirable.
    pref_time = {'time_utc': 1, 'utc_mid': 2,
                 'time_est': 3, 'est_mid': 4,
                 'time_cst': 5, 'cst_mid': 6,
                 'time_cdt': 7, 'cdt_mid': 8,
                 'time_pst': 9, 'pst_mid': 10,
                 'time_local': 11, 'local_mid': 12,
                 'time_lt': 13, 'lt_mid': 14,
                 'time_lst': 15, 'lst_mid': 16,
                 'time_time_mid': 17, 'time_mid_mid': 18,
                 'time_mid': 19, 'time_start': 20, 'time_stop': 21, 'time_end': 21}

    # Create dictionary to associate a "time" column name with its timezone.
    tz_info = {'time_utc': 'UTC', 'utc_mid': 'UTC',
               'time_est': 'EST5EDT', 'est_mid': 'EST5EDT',
               'time_cst': 'US/Central', 'cst_mid': 'US/Central',
               'time_cdt': 'CST6CDT', 'cdt_mid': 'CST6CDT',
               'time_pst': 'PST8PDT', 'pst_mid': 'PST8PDT',
               'time_mid': 'UTC', 'time_start': 'UTC', 'time_stop': 'UTC', 'time_end': 'UTC'}

    pref_arr = list()  # empty list to contain pref rank of "time cols"
    tz_arr = list()  # Empty list to contain the timezone of the "time cols"

    for n in range(0, len(times)):  # Get an array of # preferences for "times"
        # Get the nickname of the time col if its not a mid point.
        check_if_NOT_mid = nn_times.get(times[n], None)
        
        # Extract the time pattern from the column name (handle prefixed columns)
        time_pattern = None
        patterns_to_check = ['time_mid', 'utc_mid', 'est_mid', 'cst_mid', 'cdt_mid', 'pst_mid', 
                            'local_mid', 'lt_mid', 'lst_mid', 'time_utc', 'time_est', 'time_cst',
                            'time_cdt', 'time_pst', 'time_local', 'time_lt', 'time_lst',
                            'time_start', 'time_stop', 'time_end']
        for pattern in patterns_to_check:
            if pattern in times[n]:
                time_pattern = pattern
                break
        
        if check_if_NOT_mid is None:  # has no nickname, it is a midpoint or standalone time col
            # Try to match the full name first, then try extracted pattern
            pref_val = pref_time.get(times[n], pref_time.get(time_pattern, 100) if time_pattern else 100)
            tz_val = tz_info.get(times[n], tz_info.get(time_pattern, 100) if time_pattern else 100)
            pref_arr.append(pref_val)
            tz_arr.append(tz_val)
        else:  # has a nickname, pass that to get preference.
            # Try nickname first, then fall back to pattern matching if nickname doesn't match
            pref_val = pref_time.get(check_if_NOT_mid, pref_time.get(time_pattern, 100) if time_pattern else 100)
            tz_val = tz_info.get(check_if_NOT_mid, tz_info.get(time_pattern, 100) if time_pattern else 100)
            pref_arr.append(pref_val)
            tz_arr.append(tz_val)

    # Check if no time columns were found
    if len(pref_arr) == 0:
        _exit_with_error(('No time columns were found in the ICARTT file. '
                         'Try running the call to function with quiet=False '
                         'to see debug output.'))
    
    if min(pref_arr) == 100:
        _exit_with_error(('Time columnName in the "pick_a_single_time_col()" '
                         'function could not be properly identified. Try '
                         'running the call to function with quiet=False '
                         'to see debug output.'))

    # Pick one of the time columns based on your ranked preferences.
    min_pref = min(pref_arr)
    min_idx = pref_arr.index(min_pref)  # Find index of minimum preference
    time_pref = times[min_idx]  # Name of the pref time col
    tz_pref = tz_arr[min_idx]  # Timeezone of pre time col

    if tz_pref == 100:
        # Default to UTC if timezone can't be identified (common for generic "time" columns)
        tz_pref = 'UTC'
        if quiet is False:
            print(f'WARNING: Could not identify timezone for time column "{time_pref}", defaulting to UTC')

    print('Pref time col:', time_pref, ' in', tz_pref,
          'Timezone') if quiet is False else None

    bad_times = times  # duplicate list of all times
    bad_times.remove(time_pref)  # and drop all non pref times.

    # Return name of preffered column its timezone and the names of all the
    # columns that you don't want to use.
    return time_pref, tz_pref, bad_times


def icartt_time_to_datetime(df: pd.DataFrame, yr, mon, day, time_col: str,
                            tz_pref: str, remove_old_time: bool = True):
    """Convert seconds since midnight (icartt time col) 2 datetime obj col."""
    # Takes yr mon day in string or int form. Get to ints if strings passed.
    yr = int(yr) if type(yr) == str else yr
    mon = int(mon) if type(mon) == str else mon
    day = int(day) if type(day) == str else day

    # Check if the time column exists in the dataframe
    if time_col not in df.columns:
        # Try to find a matching column (case-insensitive or with prefix)
        matching_cols = [col for col in df.columns if time_col.lower() in col.lower() or col.lower() in time_col.lower()]
        if matching_cols:
            time_col = matching_cols[0]
        else:
            _exit_with_error(f"Time column '{time_col}' not found in dataframe. Available columns: {list(df.columns)}")

    # Add "timedelta" of seconds since midnight to the date the icarrt
    # file started on (typically in the file name).
    datetime_col_i = datetime.datetime(yr, mon, day) + \
        pd.to_timedelta(df[time_col], 's')
        
    # Teach this tz unaware pandas series its native timezone.
    datetime_col = datetime_col_i.dt.tz_localize(tz_pref)  # now it has a tz

    # Convert from native tz to UTC time.
    datetimecol_in_UTC = datetime_col.dt.tz_convert('UTC')

    df['datetime'] = datetimecol_in_UTC  # Create new column with date in UTC.

    if remove_old_time is True:  # Drop the old col from the df if asked
        df = df.drop(columns=time_col)

    return df


def master_icartt_time_parser(df: pd.DataFrame, icartt_file: str,
                               quiet: bool = True, remove_old_time:
                                   bool = True):
    """Identify the date like columns, convert to TZ aware datetimes)."""
    # Identify the time-like columns in this dataframe:
    df, times, tm_names = _find_datelike_cols(df, icartt_file, quiet)

    # Take Start/Stop pairs of time cols and convert them to midpoint times.
    df, times, tm_names, nn_times = _make_time_midpoint_cols(df, tm_names,
                                                             times, quiet)

    # Filter times list to only include columns that actually exist in dataframe
    times_before_filter = times.copy()
    times = [t for t in times if t in df.columns]
    
    # Debug: if filtering removed items, that's a problem
    if len(times_before_filter) > len(times):
        removed = [t for t in times_before_filter if t not in times]
        if not quiet:
            print(f"Warning: Removed {len(removed)} time column names that don't exist in dataframe: {removed}")
    
    # Check if we have any time columns at all
    if len(times) == 0:
        available_cols_str = ', '.join([f"'{c}'" for c in list(df.columns)[:30]])
        if len(df.columns) > 30:
            available_cols_str += f", ... (and {len(df.columns) - 30} more)"
        _exit_with_error(f"No time columns found in dataframe after filtering. "
                        f"Original times list: {times_before_filter}. "
                        f"Available columns: [{available_cols_str}]")
    
    # Pick which time var to use and which time zone its in.
    time_pref, tz_pref, bad_times = _pick_a_single_time_col(times,
                                                            nn_times, quiet)
    
    # CRITICAL: Ensure the selected time column actually exists in the dataframe
    # This is a final safety check before we try to use the column
    if time_pref not in df.columns:
        # Try to find a matching column (case-insensitive match)
        matching_cols = [col for col in df.columns if time_pref.lower() == col.lower()]
        if matching_cols:
            time_pref = matching_cols[0]
        else:
            # Try partial match (in case of prefixing issues)
            matching_cols = [col for col in df.columns if time_pref.lower() in col.lower() or col.lower().endswith('_' + time_pref.lower())]
            if matching_cols:
                time_pref = matching_cols[0]
            else:
                # Final check - if still not found, provide detailed error
                available_cols_str = ', '.join([f"'{c}'" for c in df.columns[:20]])  # Show first 20 columns
                if len(df.columns) > 20:
                    available_cols_str += f", ... (and {len(df.columns) - 20} more)"
                _exit_with_error(f"Selected time column '{time_pref}' not found in dataframe. "
                                f"Times list contained: {times}. "
                                f"Available columns: [{available_cols_str}]")
    
    if remove_old_time is True:  # Remove the non-preferred times
        # Only drop columns that actually exist
        bad_times_existing = [col for col in bad_times if col in df.columns]
        df = df.drop(columns=bad_times_existing)
        # you've dropped all other names from the df.
        times = list([time_pref])
    else:
        # you haven't dropped anything.
        times = list([time_pref, bad_times])

    # Get the date this data was collected on from the icarttfile name passed.
    date_full = re.search(r'\d{4}\d{2}\d{2}', icartt_file).group(0)
    yr = date_full[0:4]
    mm = date_full[4:6]
    dd = date_full[6:8]

    # Final validation: ensure time_pref exists in dataframe before proceeding
    if time_pref not in df.columns:
        available_cols_str = ', '.join([f"'{c}'" for c in list(df.columns)[:20]])
        if len(df.columns) > 20:
            available_cols_str += f", ... (and {len(df.columns) - 20} more)"
        _exit_with_error(f"Cannot proceed: Selected time column '{time_pref}' does not exist in dataframe. "
                        f"Times list was: {times}. "
                        f"Available columns: [{available_cols_str}]")
    
    # Tell the people which variable was chosen as "datetime".
    print('The time variable chosen to be converted to "datetime" is:',
          time_pref)
    
    # Convert the preffered time column to a column named 'datetime' and drop
    # all the other time columns from the larger dataframe.
    df = icartt_time_to_datetime(
        df, yr, mm, dd, time_pref, tz_pref, remove_old_time)

    # And update the times list to remove the old time and add "datetime"
    times.append('datetime')
    if remove_old_time is True:
        times = times.remove(time_pref)

    return df, times


def char_cleaner(mystring, ignore: list = []):
    """Clean up gross strings from weird characters."""
    after = mystring.strip()  # strip all leading/trailing whitespace

    # Then, replace common representations with a word.
    after = after.replace('%', 'percent')
    after = after.replace('_+_', '+')
    after = after.replace('-->', 'to')

    # A list of bad chars we don't want in our string.
    bad_chars = [' ', ',', '.', '"', '*', '!', '@', '#', '$', '^', '&',
                 '(', ')', '=', '?', '/', '\\', ':', ';', '~', '`', '<',
                 '>', ']', '[', '{', '}']

    for i in range(0, len(bad_chars)):
        if bad_chars[i] not in ignore:  # don't replace chars they want
            after = after.replace(bad_chars[i], '_')

    return after


def _build_meta_dict(icartt_file: str, meta: dict = {}, flt_num: int = None):
    """Take and combines metadata from different\
    icartt files into a dictionary. So you can access metadata from an\
    individual icartt file by typing in its instrument name & flt #."""
    # If meta empty, initialize the dict.
    if bool(meta) is False:
        meta = {'Instruments': {}, 'Data_Info': {}, 'Instrument_Info': {},
                'PI_Info': {}, 'Uncertainty': {}, 'Revision': {},
                'Stipulations': {}, 'Institution_Info': {}}
    # Open the file.
    with open(icartt_file, "r") as f:  # Get number of header rows
        header_row = int(f.readlines()[0].split(",")[0]) - 1

    with open(icartt_file, "r") as f:
        reader = csv.reader(f)
        ln_num = 0  # intitalize line counting var.
        for row in reader:
            line = " ".join(row)  # read line by line.

            # Icartt splits headers with a ":", use that to split them.
            before, sep, after = line.rpartition(":")
            after = char_cleaner(after, ignore=':')  # Pass to string cleaner.

            # First 3 lines have set parameters in ICARTT Files.
            if ln_num == 1:
                PI = after
            if ln_num == 2:
                Institution = after
            if ln_num == 3:
                Instrument = after
            
            # Once you know the instrument, you can start to build the dict
            # (becase we are using the instrument part of the dict index we
            # can't do it until we get to this line number. )
            if ln_num == 3:
                meta['PI_Info'][Instrument] = PI
                meta['Institution_Info'][Instrument] = Institution
                if flt_num is not None:
                    meta['Instruments'][flt_num] = Instrument
                else:
                    meta['Instruments'] = Instrument

            # The rest of the meta data is on arbitrary line #s based on how
            # much info the author of the ICARTT included, so just parse the
            # string to ID which row that is contained on. Then,  append info
            # from  this file into the meta dictionary indexed on the
            # instrument name
            if 'DATA_INFO' in before:
                meta['Data_Info'][Instrument] = after
            if 'UNCERTAINTY' in before:
                meta['Uncertainty'][Instrument] = after
            if 'REVISION' in before:
                meta['Revision'][Instrument] = after
            if 'INSTRUMENT_INFO' in before:
                meta['Instrument_Info'][Instrument] = after
            if 'STIPULATIONS_ON_USE' in before:
                meta['Stipulations'][Instrument] = after

            if ln_num > header_row - 1:
                break  # top once you reach data.
            ln_num = ln_num + 1

    return meta  # return dictionary with this info.


def read_icartt(icartt_file: str, flt_num: int = None, meta: dict = {},
                instr_name_prefix: bool = False, add_file_no: bool = False):
    """Parse a single ICARTT file to a pandas dataframe."""
    # Get the header row number from the ICARTT.
    with open(icartt_file, "r") as f:
        first_line = f.readlines()[0]
        header_row_num = int(first_line.split(",")[0]) - 1
    
    # Check if the header row is a separator row (all asterisks), if so use next row
    with open(icartt_file, "r") as f:
        lines = f.readlines()
        if header_row_num < len(lines):
            header_line = lines[header_row_num].strip()
            # If it's a separator row (all asterisks or similar), use the next row
            if header_line:
                # Check if it's mostly separator characters (asterisks, dashes, equals)
                cleaned_line = header_line.replace(' ', '').replace('\t', '')
                # More lenient check: if line is mostly separators, skip it
                if cleaned_line:
                    separator_count = sum(1 for c in cleaned_line if c in ['*', '-', '=', '_'])
                    if separator_count > len(cleaned_line) * 0.9 and len(cleaned_line) > 10:
                        header_row_num += 1

    # Parse the table starting where data begins (e.g. after the header).
    # Use skiprows to skip up to the header, then use header=0 to read column names from first row
    df = pd.read_csv(icartt_file, skiprows=header_row_num, nrows=0, delimiter=",", skipinitialspace=True)
    column_names = list(df.columns)
    # Now read the actual data starting from the row after the header
    df = pd.read_csv(icartt_file, skiprows=header_row_num + 1, names=column_names, delimiter=",", skipinitialspace=True)

    # Set possible error values to NaNs - single vectorized operation instead of 5 scans
    df.mask(df.isin([-9, -99, -999, -9999, -99999]), np.nan, inplace=True)
    
    # Strip leading/tailing white space around variable names
    df.columns = [c.strip() for c in list(df.columns)]

    # Build/ append metadata from ICARTT to a dictionary file.
    meta = _build_meta_dict(icartt_file, meta, flt_num)

    # If instr_name_prefix is set to True, add the instrument name as a
    # prefix to the column names in the super merge dataframe so you know
    # which instrument collected that data (useful if multiple instruments
    # measure "NO3" and named them all "NO3". If set to false, then
    # you'd have duplicate column names in the resulting super-merge dataframe)
    if instr_name_prefix is True:
        if flt_num is not None:  # Indexed by flt # if more than 1 icartt file.
            df = df.add_prefix(meta['Instruments'][flt_num] + '_')
        else:  # Not indexed by flt # if only  1 icartt file.
            df = df.add_prefix(meta['Instruments'] + '_')

    if add_file_no is True:
        # Create a column same length as data that contains the file #
        sz = len(df[df.columns[0]])  # get appropriate length
        fnum_arr = np.full(shape=sz, fill_value=flt_num, dtype=np.int)
        df['Flight_N'] = fnum_arr

    return df, meta  # dataframe with data, and df with metadata


def _read_icartt_multileg(icartt_file: str, flt_num: int = None, meta:
                          dict = [], instr_name_prefix: bool = True):
    """Parse multi-leg icartt files, combine into single df."""
    # Sort the list of input ICARTTs.
    icartts = sorted(icartt_file)

    df_list = []  # Accumulate dataframes in list for single concat

    for ict in icartts:  # Loop over the dif ICARTT Legs
        # Parse individual file
        df_i, meta_i = read_icartt(ict, flt_num=int(flt_num), meta=meta,
                                   instr_name_prefix=instr_name_prefix)
        meta = meta_i  # update metadata file... gets appended upstream.

        # Accumulate dataframe in list
        df_list.append(df_i)

    # Single concat at end - much faster than repeated append
    df = pd.concat(df_list, ignore_index=True)

    return df, meta  # Return the merged df and updated metadata


def _process_single_flight(args):
    """Process a single flight file. Designed for parallel execution via multiprocessing.Pool."""
    flight, icartt, flt_num, mode, prefix_opt, mstr_tmln = args

    print(f" - Processing: {os.path.basename(flight)}")

    # Fresh meta dict per file (no shared state)
    meta = {'Instruments': {}, 'Data_Info': {}, 'Instrument_Info': {},
            'PI_Info': {}, 'Uncertainty': {}, 'Revision': {},
            'Stipulations': {}, 'Institution_Info': {}}

    if type(icartt) is list:
        df_data, meta = _read_icartt_multileg(icartt, flt_num=flt_num,
                                               meta=meta, instr_name_prefix=prefix_opt)
        icartt_file = icartt[0]  # Use first leg filename for date extraction
    elif type(icartt) is str:
        add_file_no = True if mode == 'Stack_On_Top' else None
        df_data, meta = read_icartt(icartt, flt_num=flt_num, meta=meta,
                                     instr_name_prefix=prefix_opt, add_file_no=add_file_no)
        icartt_file = icartt

    df_data, times = master_icartt_time_parser(df_data, icartt_file, quiet=True, remove_old_time=True)

    if mode == 'Merge_Beside' and mstr_tmln:
        df_data = align2master_timeline(df_data, mstr_tmln[0], mstr_tmln[1],
                                         mstr_tmln[2], quiet=True, datetime_index=False)

    return flt_num, df_data, meta


def _merge_meta_dicts(meta_list):
    """Merge a list of per-file meta dicts into one combined dict."""
    combined = {'Instruments': {}, 'Data_Info': {}, 'Instrument_Info': {},
                'PI_Info': {}, 'Uncertainty': {}, 'Revision': {},
                'Stipulations': {}, 'Institution_Info': {}}
    for m in meta_list:
        for key in combined:
            if isinstance(m[key], dict):
                combined[key].update(m[key])
            else:
                combined[key] = m[key]
    return combined


def _main_loop_parse_flights(DATA: dict):
    """Looper for parsing indv flights in a directory."""
    # Make groupings of standard and multileg flights.
    DATA['FLIGHTS'] = _organize_standard_and_multileg_flights(DATA)

    n_workers = DATA.get('N_WORKERS', 1)
    mstr_tmln = DATA.get('MSTR_TMLN', None)

    # Build args list: pre-assign flight numbers by enumerating flights
    args_list = []
    for ct, (flight, icartt) in enumerate(DATA['FLIGHTS'].items(), start=1):
        args_list.append((flight, icartt, ct, DATA['MODE'],
                          DATA['PREFIX_OPT'], mstr_tmln))

    # Print summary before dispatching
    print(f"\nProcessing {len(args_list)} flight(s) with {n_workers} worker(s)...")

    # Process files either sequentially or in parallel
    if n_workers == 1:
        results = [_process_single_flight(a) for a in args_list]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_process_single_flight, args_list)

    # Sort results by flight number to ensure deterministic order
    results.sort(key=lambda r: r[0])

    # Merge all meta dicts
    meta = _merge_meta_dicts([r[2] for r in results])

    # Collect dataframes in flight-number order
    df_list = [r[1] for r in results]

    # Concatenate based on mode
    if DATA['MODE'] == 'Stack_On_Top':
        df_all = pd.concat(df_list, ignore_index=True)
    else:  # Merge_Beside mode
        # Handle duplicate column names - can occur even with prefix_instr_name=True
        # when multiple files come from the same instrument (e.g. multi-leg flights
        # that share column names). Accumulate non-overlapping frames for a single
        # fast concat at the end; only use combine_first for true overlaps.
        non_overlapping_list = [df_list[0]]
        all_cols_so_far = set(df_list[0].columns)

        for df_data in df_list[1:]:
            new_cols = set(df_data.columns)
            overlapping_cols = all_cols_so_far & new_cols
            non_overlapping_cols = new_cols - overlapping_cols

            if overlapping_cols:
                # Merge overlapping columns into the first frame using combine_first
                if non_overlapping_list:
                    # Materialize accumulated frames so we can update in place
                    df_all = pd.concat(non_overlapping_list, axis=1)
                    non_overlapping_list = [df_all]

                for col in overlapping_cols:
                    non_overlapping_list[0][col] = non_overlapping_list[0][col].combine_first(df_data[col])
                if non_overlapping_cols:
                    non_overlapping_list.append(df_data[list(non_overlapping_cols)])
            else:
                non_overlapping_list.append(df_data)

            all_cols_so_far |= new_cols

        # Single concat at the end
        df_all = pd.concat(non_overlapping_list, axis=1)

    # Check if the User wants us to align the Stacked data to a master timeline
    # Aligns AFTER all icartts have been loaded in.
    if (DATA['MODE'] == 'Stack_On_Top') and (bool(DATA.get('MSTR_TMLN'))
                                             is True):
        tmln = DATA['MSTR_TMLN']
        df_all = align2master_timeline(df_all, tmln[0], tmln[1], tmln[2],
                                       quiet=True, datetime_index=False)
    elif (DATA['MODE'] == 'Stack_On_Top'):
        df_all = df_all.set_index(['datetime', 'Flight_N'])

    return df_all, meta


def _handle_input_configuration(DATA: dict):
    """Make sure user passed appropriate input."""
    # icartt_directory validation only needed for non-Load_Pickle modes
    if DATA['DIR_ICARTT'] is None:
        _exit_with_error("icartt_directory must be specified for this mode")

    print('1. Input ICARTT directory:' + DATA['DIR_ICARTT'])

    # 1. Ensure ICARTT directory is valid.
    if not os.path.isdir(DATA['DIR_ICARTT']):
        _exit_with_error("Input ICARTT directory is invalid.")

    # 2. Ensure that directory  has ICARTT files in it. return  list.
    DATA['ICARTT_FILES'] = _crawl_directory(DATA['DIR_ICARTT'])

    if len(DATA['ICARTT_FILES']) == 0:
        # If no icartts were found, exit and notify user.
        _exit_with_error("No ICARTT files found in the input directory.")
    else:  # Else, inform on the number of ICARTT files
        print(" - Found [ {} ] ICARTTs.".format(len(DATA['ICARTT_FILES'])))

    # 2b. Filter files by date range if master_timeline is provided
    if bool(DATA.get('MSTR_TMLN')) is True:
        print(" - Filtering files by date range...")
        DATA['ICARTT_FILES'] = _filter_files_by_date_range(
            DATA['ICARTT_FILES'],
            DATA['MSTR_TMLN'][0],
            DATA['MSTR_TMLN'][1]
        )
        if len(DATA['ICARTT_FILES']) == 0:
            _exit_with_error("No ICARTT files found within the specified date range.")
        print(" - [ {} ] ICARTTs remain after date filtering.\n".format(len(DATA['ICARTT_FILES'])))
    else:
        print()

    # 3. Check that a valid mode has been passed
    valid_modes = ['Stack_On_Top', 'Merge_Beside']
    if DATA['MODE'] not in valid_modes:
        _exit_with_error(("Input mode entered is invalid."),
                         ("Valid Options are:" + valid_modes))

    # 4. Check that master_timeline info has been provided if necessary.
    if DATA['MODE'] == 'Merge_Beside':
        if bool(DATA.get('MSTR_TMLN')) is False:
            _exit_with_error(("For input mode 'Merge_Beside', input for "),
                             ("MSTR_TMLN is also needed."))

    return DATA  # give input back with the list of icartts now included.


def icartt_merger(icartt_directory: str = None,
                  mode_input: str = None,
                  master_timeline: list = [],
                  pickle_directory: str = None,
                  pickle_filename: str = 'icartt_merge_output',
                  prefix_instr_name: bool = True,
                  n_workers: int = 1):
    """Merge a directory of icarrts into a pandas dataframe & save as a pkl.

    # ========================================================================
    # ========================    INPUTS   ===================================
    # ========================================================================
    #
    #   (1) icartt_directory - A string containing the absolute path to a folder
    #                      which contains all the individual icartt files that
    #                      you wish to merge together. Not required if
    #                      mode_input='Load_Pickle'.
    #
    #   (2) mode_input - A string describing HOW you would like to merge these
    #                    icartt files in the icartt_directory together. Valid
    #                    options are "Stack_On_Top", "Merge_Beside", or
    #                    "Load_Pickle".
    #
    #     "Stack_On_Top": Each icartt file is for a different date,
    #      but contains data from multiple instruments or mutltiple
    #      measurements and  you want that data in a single
    #      file (e.g. indexed by time, and File/Flight #). Contents of
    #      individual icarrt files will be "stacked on top" of one another.
    #
    #     "Merge_Beside": Each icartt file is for the entire
    #      sampling period, but contains different measurements.
    #      You want to have all of these differnt measurments
    #      on the same time base, throughout the whole period. The contents
    #      of each icartt file will be "merged beside" one another.
    #
    #     "Load_Pickle": Load previously saved pickle files from
    #      pickle_directory instead of processing icartt files.
    #      Requires pickle_directory and pickle_filename to be specified.
    #
    #   (3) master_timeline - OPTIONAL if "Stack_On_Top" or "Load_Pickle",
    #                        required if "Merge_Beside". It is list with 3 items:
    #
    #       -  Startdate_str:  A string containing the start date of the
    #                        "mastertimeline" that all data  will be
    #                        merged to.Format is 'YYYY-MM-DD HH:MM:SS'
    #       -  Enddate_str:  A string containing the end date of the
    #                        "mastertimeline" that all data  will be
    #                        merged to. Format is 'YYYY-MM-DD HH:MM:SS'
    #       - Averaging_Step: An integer that is the number of seconds
    #                        for each timestep in between startdate and
    #                        end date. So 120 for a 2 minute average.
    #
    #    (4) pickle_directory - OPTIONAL string containing the abs path where
    #                         the output file will be written. If None, no files
    #                         will be saved (only returns df and meta). If empty
    #                         string (''), output will be stored in the input icartt_dir.
    #                         REQUIRED if mode_input='Load_Pickle'.
    #
    #    (5) pickle_filename - OPTIONAL string containing what you'd like the
    #                         output file to be called (not including its
    #                         extension). Default is 'icartt_merge_output'
    #
    #    (6) prefix_instr_name - OPTIONAL boolean value indicating whether you
    #                         would like to append the instrument name
    #                         contained in the icartt file to all the var
    #                         names. Default is True since when merging
    #                         icartt files it is common to have some PI's
    #                         measuring the same items & naming them the same.
    #
    #    (7) n_workers - OPTIONAL integer specifying the number of parallel
    #                    workers for processing ICARTT files. Default is 1
    #                    (sequential). Values > 1 use multiprocessing.Pool.
    #
    # ========================================================================
    """
    # If mode is Load_Pickle, load the pickle files and return
    if mode_input == 'Load_Pickle':
        if pickle_directory is None:
            _exit_with_error("pickle_directory must be specified when mode_input='Load_Pickle'")

        # Load the dataframe pickle
        df_path = os.path.join(pickle_directory, pickle_filename + '.pkl')
        if not os.path.exists(df_path):
            _exit_with_error(f"Pickle file not found: {df_path}")

        print(f"Loading dataframe from: {df_path}")
        df = pd.read_pickle(df_path)

        # Load the metadata pickle
        meta_path = os.path.join(pickle_directory, pickle_filename + '_meta.pickle')
        if not os.path.exists(meta_path):
            _exit_with_error(f"Metadata pickle file not found: {meta_path}")

        print(f"Loading metadata from: {meta_path}")
        meta = mpu.io.read(meta_path)

        print("Successfully loaded pickle files.")
        return df, meta

    # Format the input for easier referencing.
    inputs = {'DIR_ICARTT': icartt_directory,
              'DIR_OUTPUT': pickle_directory,
              'O_FILENAME': pickle_filename,
              'MODE': mode_input,
              'PREFIX_OPT': prefix_instr_name,
              'MSTR_TMLN': master_timeline,
              'N_WORKERS': n_workers}

    # Make sure you got appropriate inputs from the user, retrieve icartt files
    DATA = _handle_input_configuration(inputs)
    
    # Loop through parsing the flights & collecting them in a single dataframe.
    df, meta = _main_loop_parse_flights(DATA)

    # Remove rows where all data columns (excluding time index) are NaN
    # This significantly reduces file size for wide time windows with sparse data
    initial_rows = len(df)
    df = df.dropna(how='all')
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f'Removed {rows_removed} rows with all NaN values ({rows_removed/initial_rows*100:.1f}% of data)')

    # Save the Output only if output_directory is provided (not None).
    if DATA['DIR_OUTPUT'] is not None:
        filename = os.path.join(DATA['DIR_OUTPUT'], DATA['O_FILENAME'] + '.pkl')
        df.to_pickle(filename)
        
        # Save the metadata to a picke as well.
        filename_meta = os.path.join(DATA['DIR_OUTPUT'], DATA['O_FILENAME'] + '_meta.pickle')
        mpu.io.write(filename_meta, meta)
        
        # Tell the people where you saved it.
        print('Output dataframe and metadata saved at:' + filename)
    
    return df, meta

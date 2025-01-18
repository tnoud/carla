#  Helper functions
import gzip
import json
import os
import random
import re
import shutil
import glob
from typing import List, Dict, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Point

from utils.stats_tracker import stats_tracker
from utils.pipeline_setup import OUTPUT_DIR
from utils.pipeline_setup import PROJECT_ROOT
from utils import settings as s
#from utils.types import PlanLeg, PlanSegment, SegmentedPlan, SegmentedPlans, UnSegmentedPlan, UnSegmentedPlans
from utils.logger import logging

logger = logging.getLogger(__name__)


def open_text_file(file_path, mode):
    """
    Open a text file, also works for gzipped files.
    """
    is_gzip = False
    with open(file_path, 'rb') as f:
        # Read the first two bytes for the magic number
        magic_number = f.read(2)
        is_gzip = magic_number == b'\x1f\x8b'

    if is_gzip:
        return gzip.open(file_path, mode)
    else:
        return open(file_path, mode, encoding='utf-8')


def modify_text_file(input_file, output_file, replace, replace_with):
    """
    Replace text in a text file.
    Also works for gzipped files.
    """
    logger.info(f"Replacing '{replace}' with '{replace_with}' in {input_file}...")
    with open_text_file(input_file, 'rt') as f:
        file_content = f.read()

    modified_content = file_content.replace(replace, replace_with)

    with open_text_file(output_file, 'wt') as f:
        f.write(modified_content)
    logger.info(f"Wrote modified file to {output_file}.")


def create_leg_ids(legs_file):
    """
    Create IDs in MiD similar to the person or hh IDs.
    This does obviously not create unique leg ids in the expanded population, only in the input leg data for further processing.
    """
    logger.info(f"Creating unique leg ids in {legs_file}...")
    if s.LEG_ID_COL in legs_file.columns:
        logger.info(f"Legs file already has unique leg ids, skipping.")
        return

    # Create unique leg ids
    legs_file[s.LEG_ID_COL] = legs_file[s.PERSON_ID_COL].astype(str) + "_" + legs_file[s.LEG_NON_UNIQUE_ID_COL].astype(
        str)
    logger.info(f"Created unique leg ids.")
    return legs_file


def read_csv(csv_path: str, test_col=None, use_cols=None) -> pd.DataFrame:
    """
    Read a csv file with unknown separator and return a dataframe.
    Also works for gzipped files.

    :param csv_path: Path to the CSV file.
    :param test_col: Column name that should be present in the file for validation.
    :param use_cols: List of columns to use from the file. Defaults to all columns.
    :return: DataFrame with the contents of the CSV file.
    :raises: KeyError, ValueError if `test_col` is not found after attempting to read the file.
    """
    csv_path = make_path_absolute(csv_path)
    try:
        if csv_path.endswith('.gz'):
            with gzip.open(csv_path, 'rt') as f:
                df = pd.read_csv(f, sep=',', usecols=use_cols)
        else:
            df = pd.read_csv(csv_path, sep=',', usecols=use_cols)
        if test_col is not None:
            test = df[test_col]
    except (KeyError,
            ValueError):  # Sometimes also throws without test_col, when the file is not comma-separated. This is good.
        logger.info(f"ID column '{test_col}' not found in {csv_path}, trying to read as ';' separated file...")
        if csv_path.endswith('.gz'):
            with gzip.open(csv_path, 'rt') as f:
                df = pd.read_csv(f, sep=';', usecols=use_cols)
        else:
            df = pd.read_csv(csv_path, sep=';', usecols=use_cols)
        try:
            if test_col is not None:
                test = df[test_col]
        except (KeyError, ValueError):
            logger.error(f"ID column '{test_col}' still not found in {csv_path}, verify column name and try again.")
            raise
        logger.info("Success.")
    return df


def convert_to_point(point_input, target='Point'):
    """
    Forces all weird location representations to either Point or array.
    :param point_input: String input of the format 'x,y' or '[x,y]', a list [x, y], or a Shapely Point
    :param target: Desired output format, either 'Point' or 'array'
    :return: Shapely Point or numpy array
    """
    if point_input is None:
        return None
    if isinstance(point_input, Point):
        if target == 'array':
            return np.array([point_input.x, point_input.y])
        return point_input
    if isinstance(point_input, list) or isinstance(point_input, np.ndarray):
        if len(point_input) == 0:
            return None
        elif len(point_input) == 2 and all(isinstance(coord, (int, float)) for coord in point_input):
            if target == 'array':
                return np.array([point_input[0], point_input[1]])
            return Point(point_input)
        else:
            # if point_input.size == 2:
            #     # Try to unpack seemingly nested arrays
            #     return convert_to_point(point_input[0], target)
            raise ValueError("List or array input must be of the form [x, y] with numeric coordinates")
    if isinstance(point_input, float):
        if np.isnan(point_input):
            return None
    if isinstance(point_input, str):
        # Remove brackets if present
        while True:
            if point_input.startswith("[") and point_input.endswith("]"):
                point_input = point_input[1:-1]
            else:
                break
        if len(point_input) == 0:
            return None

        # Use a regular expression to extract numbers from a string
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", point_input)

        # Convert the extracted strings to float and create a Shapely Point or numpy array
        if len(matches) == 2:
            x, y = map(float, matches)
            if target == 'array':
                return np.array([x, y])
            return Point(x, y)
        else:
            raise ValueError(f"Invalid point input format: {point_input}")

    raise ValueError(f"Incompatible input: {type(point_input)}, {point_input}")


def seconds_from_datetime(datetime):
    """
    Convert a datetime object to seconds since midnight of the referenced day.
    :param datetime: A datetime object.
    """
    return (datetime - pd.Timestamp(s.BASE_DATE)).total_seconds()


def compress_to_gz(input_file, delete_original=True):
    logger.info(f"Compressing {input_file} to .gz...")
    with open(input_file, 'rb') as f_in:
        with gzip.open(f"{input_file}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if delete_original:
        os.remove(input_file)
    logger.info(f"Compressed to {input_file}.gz.")


def find_outer_boundary(gdf, method='convex_hull'):
    combined = gdf.geometry.unary_union

    # Calculate the convex hull or envelope
    if method == 'convex_hull':
        outer_boundary = combined.convex_hull
    elif method == 'envelope':
        outer_boundary = combined.envelope
    else:
        raise ValueError("Method must be 'convex_hull' or 'envelope'")

    return outer_boundary


def distribute_by_weights(data_to_distribute: pd.DataFrame, weighted_points_in_cells: pd.DataFrame, external_id_column,
                          cut_missing_ids=False):
    """
    Distribute data points from `weights_df` across the population dataframe based on weights (e.g. assign buildings to households).

    The function modifies the internal population dataframe by appending the point IDs from the weights dataframe
    based on their weights and the count of each ID in the population dataframe.

    Args:
        data_to_distribute (pd.DataFrame): DataFrame containing the data on cell level to distribute.
        weighted_points_in_cells (pd.DataFrame): DataFrame containing the ID of the geography, point IDs, and their weights.
        Must contain all geography IDs in the population dataframe. Is allowed to contain more (they will be skipped).
        external_id_column (str): The column name of the ID in the weights dataframe (e.g. 'BLOCK_NR').
        cut_missing_ids (bool): If True, IDs in the population dataframe that are not in the weights dataframe are cut from the population dataframe.
    """
    logger.info("Starting distribution by weights...")

    if not data_to_distribute[external_id_column].isin(weighted_points_in_cells[external_id_column]).all():
        if cut_missing_ids:
            logger.info(f"Not all geography IDs in the population dataframe are in the weights dataframe. "
                        f"Cutting missing IDs: {set(data_to_distribute[external_id_column]) - set(weighted_points_in_cells[external_id_column])}")
            data_to_distribute = data_to_distribute[
                data_to_distribute[external_id_column].isin(weighted_points_in_cells[external_id_column])].copy()
        else:
            raise ValueError(f"Not all geography IDs in the population dataframe are in the weights dataframe. "
                             f"Missing IDs: {set(data_to_distribute[external_id_column]) - set(weighted_points_in_cells[external_id_column])}")

    # Count of each ID in population_df
    id_counts = data_to_distribute[external_id_column].value_counts().reset_index()
    id_counts.columns = [external_id_column, '_processing_count']
    logger.info(f"Computed ID counts for {len(id_counts)} unique IDs.")

    # Merge with weights_df
    weighted_points_in_cells = pd.merge(weighted_points_in_cells, id_counts, on=external_id_column, how='left')

    def distribute_rows(group):
        total_count = group['_processing_count'].iloc[0]
        if total_count == 0 or pd.isna(total_count):
            logger.debug(f"Geography ID {group[external_id_column].iloc[0]} is not in the given dataframe, "
                         f"likely because no person/activity etc. exists there. Skipping distribution for this ID.")
            return []
        # Compute distribution
        group['_processing_repeat_count'] = (group['ewzahl'] / group['ewzahl'].sum()) * total_count
        group['_processing_int_part'] = group['_processing_repeat_count'].astype(int)
        group['_processing_frac_part'] = group['_processing_repeat_count'] - group['_processing_int_part']

        # Distribute remainder
        remainder = total_count - group['_processing_int_part'].sum()
        assert remainder >= 0 and remainder % 1 == 0, f"Remainder is {remainder}, should be a positive integer."
        remainder = int(remainder)
        top_indices = group['_processing_frac_part'].nlargest(remainder).index
        group.loc[top_indices, '_processing_int_part'] += 1

        # Expand rows based on int_part
        expanded = []
        for _, row in group.iterrows():
            expanded.extend([row.to_dict()] * int(row['_processing_int_part']))
        return expanded

    expanded_rows = []
    for _, group in weighted_points_in_cells.groupby(external_id_column):
        expanded_rows.extend(distribute_rows(group))

    expanded_weights_df = pd.DataFrame(expanded_rows).drop(
        columns=['_processing_count', '_processing_repeat_count', '_processing_int_part', '_processing_frac_part'])
    logger.info(f"Generated expanded weights DataFrame with {len(expanded_weights_df)} rows.")
    if len(expanded_weights_df) != data_to_distribute.shape[0]:
        raise ValueError(f"Expanded weights DataFrame has {len(expanded_weights_df)} rows, "
                         f"but the population DataFrame has {data_to_distribute.shape[0]} rows.")

    # Add a sequence column to both dataframes to prevent cartesian product on merge
    data_to_distribute['_processing_seq'] = data_to_distribute.groupby(external_id_column).cumcount()
    expanded_weights_df['_processing_seq'] = expanded_weights_df.groupby(external_id_column).cumcount()

    # Merge using the ID column and the sequence
    data_to_distribute = pd.merge(data_to_distribute, expanded_weights_df, on=[external_id_column, '_processing_seq'],
                                  how='left').drop(columns='_processing_seq')

    logger.info("Completed distribution by weights.")
    return data_to_distribute


def random_point_in_polygon(polygon):
    if not polygon.is_valid or polygon.is_empty:
        raise ValueError("Invalid polygon")

    min_x, min_y, max_x, max_y = polygon.bounds

    while True:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            return random_point


def calculate_condition_likelihoods(df, filter_col, target_col) -> dict:
    """
    Calculate the likelihood of a target condition being true under different conditions.
    :param df: DataFrame containing the relevant data.
    :param filter_col: Given the unique values in this column,
    :param target_col: calculate the likelihood of each target condition.
    :return: Dictionary with likelihoods for each unique value in the filter column.
    """
    likelihoods = {}

    for condition in df[filter_col].unique():
        likelihood = df[df[filter_col] == condition][target_col].mean()
        likelihoods[condition] = likelihood

    logger.info(f"Calculated likelihoods for {len(likelihoods)} unique values in {filter_col}.")
    logger.info(f"{likelihoods}")
    return likelihoods


def calculate_value_frequencies_df(df, filter_col, target_col) -> pd.DataFrame:
    """
    Calculate the normalized frequency of each target value for each unique value in the filter column.
    :param df:
    :param filter_col: Given all unique values in this column,
    :param target_col: calculate the frequency of each target value.
    :return:
    """
    # Create a grouped DataFrame
    grouped = df.groupby([filter_col, target_col]).size().unstack(fill_value=0)

    # Normalize the counts to get frequencies
    frequencies_df = grouped.div(grouped.sum(axis=1), axis=0)

    logger.info(f"Calculated frequencies for {len(frequencies_df)} unique values in {filter_col}.")
    logger.info(f"{frequencies_df}")

    return frequencies_df


def summarize_slack_factors(slack_df):
    """
    Take the slack_factors df and summarize them by activities, for
    use in the activity_placer.
    :param slack_df:
    :return:
    """
    logger.info(f"Summarizing slack factors for {len(slack_df)} rows...")
    slack_df = slack_df[(slack_df['slack_factor'] > 1) & (slack_df['slack_factor'] < 50)]
    logger.info(f"Dropped outliers and false positives, {len(slack_df)} rows remaining.")

    grouped = slack_df.groupby(['start_activity', 'via_activity', 'end_activity'])

    summary_df = grouped['slack_factor'].agg(['median', 'mean', 'std', 'count']).reset_index()

    # Rename columns for clarity
    summary_df.columns = ['start_activity', 'via_activity', 'end_activity',
                          'median_slack_factor', 'mean_slack_factor',
                          'std_slack_factor', 'count_observations']
    logger.info(f"Summarized slack factors for {len(summary_df)} unique activity combinations.")
    return summary_df


def calculate_travel_time_matrix(cells_gdf, speed, detour_factor=1.3):
    """
    Constructs a travel time matrix for teleported modes (e.g. walk, bike).

    :param cells_gdf: GeoDataFrame with cells.
    :param speed: Movement speed in meters per second.
    :param detour_factor: Factor by which the distance is multiplied.
    :return: DataFrame with columns FROM, TO, VALUE (travel time in seconds).
    """
    # Calculate the center of each cell
    ensure_crs(cells_gdf)
    centers = cells_gdf.geometry.centroid
    effective_speed = speed / detour_factor

    from_list = []
    to_list = []
    value_list = []

    # Calculate distances and travel times
    for i, from_point in enumerate(centers):
        logger.debug(f"Calculating travel times for cell {i} of {len(centers)}...")
        for j, to_point in enumerate(centers):
            distance = from_point.distance(to_point)  # Distance between centers
            travel_time = distance / effective_speed
            from_list.append(cells_gdf.iloc[i]["NAME"])
            to_list.append(cells_gdf.iloc[j]["NAME"])
            value_list.append(travel_time)

    # Create the DataFrame
    travel_time_df = pd.DataFrame({'FROM': from_list, 'TO': to_list, 'VALUE': value_list})

    return travel_time_df


def ensure_crs(gdf: gpd.GeoDataFrame, crs: str = 'EPSG:25832') -> gpd.GeoDataFrame:
    """
    Ensure the coordinate system of a GeoDataFrame is of a certain type.

    :param gdf: Input GeoDataFrame.
    :param crs: Coordinate reference system to transform to. Default is 'EPSG:32632' (UTM32N).
    :return: GeoDataFrame with transformed coordinate system.
    """
    if gdf.crs != crs:
        logger.info(f"Converting GeoDataFrame from {gdf.crs} to {crs}...")
        return gdf.to_crs(crs)
    logger.info(f"GeoDataFrame already in {crs}, skipping conversion.")
    return gdf


class SlackFactors:
    """
    Manages slack factors for different activities.
    """

    def __init__(self, slack_factors_csv_path: str):
        self.slack_factors_df = read_csv(slack_factors_csv_path)
        logger.info(f"Loaded slack factors from {slack_factors_csv_path}.")

    def get_slack_factor(self, activity_from: str, activity_via: str, activity_to: str) -> float:
        """
        Retrieve the slack factor for a given activity combination.
        """
        slack_factor_row = self.slack_factors_df.loc[
            (self.slack_factors_df['start_activity'] == activity_from) &
            (self.slack_factors_df['via_activity'] == activity_via) &
            (self.slack_factors_df['end_activity'] == activity_to)
            ]

        if not slack_factor_row.empty:
            slack_factor: float = slack_factor_row['median_slack_factor'].iloc[0]
        else:
            # Fallback to a default slack factor if not found
            logger.debug(f"No slack factor found for activities: {activity_from}, {activity_via}, {activity_to}. "
                         f"Using default slack factor of {s.DEFAULT_SLACK_FACTOR}")
            slack_factor = s.DEFAULT_SLACK_FACTOR

        return slack_factor

    def calculate_expected_time_with_slack(self, time_from_start_to_via: float, time_from_via_to_end: float,
                                           activity_from: str,
                                           activity_via: str, activity_to: str) -> float:
        """
        Calculates the expected time with slack for a given activity combination. Makes sure the returned time is plausible.

        """
        expected_time: float = ((time_from_start_to_via + time_from_via_to_end) /
                                self.get_slack_factor(activity_from, activity_via, activity_to))
        time_diff = np.abs(time_from_start_to_via - time_from_via_to_end)
        time_sum = time_from_start_to_via + time_from_via_to_end
        if expected_time < time_diff or expected_time > time_sum:  # this makes the trip impossible. We find a reasonable alternative:
            logger.debug(f"Expected time is infeasible. Returning time difference plus half of the smaller time.")
            expected_time = time_diff + (min(time_from_start_to_via, time_from_via_to_end) / 2)
        return expected_time

    def calculate_expected_distance_with_slack(self, distance_from_start_to_via: float, distance_from_via_to_end: float,
                                               activity_from: str, activity_via: str, activity_to: str) -> float:
        """
        Identical to calculate_expected_time_with_slack, but different parameter names for clarity.
        """
        expected_distance: float = ((distance_from_start_to_via + distance_from_via_to_end) /
                                    self.get_slack_factor(activity_from, activity_via, activity_to))
        distance_diff = np.abs(distance_from_start_to_via - distance_from_via_to_end)
        if expected_distance < distance_diff:
            logger.debug(
                f"Expected distance is too low. Returning distance difference plus half of the smaller distance.")
            expected_distance = distance_diff + (min(distance_from_start_to_via, distance_from_via_to_end) / 2)
        return expected_distance

    def get_all_estimated_times_with_slack(self, leg_chain, level=0):
        """
        Recursive function that adds columns for each level of slack factor calculation until all needed levels
        have been processed. The columns are named level_0, level_1, etc. and contain the slack-estimated direct
        time of all legs up to and including the entry (i.e., how long it would take without the detour of the
        in-between activities). The last column contains the estimated direct time from chain start to chain end.
        :param leg_chain: df
        :param level: int, highest level reached
        :return:
        """
        if level == 0:
            times_col = s.LEG_DURATION_MINUTES_COL
        else:
            times_col = f"level_{level}"

        # Base cases
        len_times_col = leg_chain[times_col].notna().sum()
        if len_times_col == 0:
            logger.warning(f"Received empty DataFrame, returning unchanged.")
            return leg_chain, level
        elif len_times_col == 1:
            logger.warning(f"Received single leg, returning unchanged.")
            return leg_chain, level
        elif len_times_col == 2:
            logger.debug(f"Two legs remain to estimate, solving last level.")
            return self.solve_level(leg_chain, level), level + 1
        # Recursive case
        else:
            logger.debug(f"More than two legs remain to estimate, solving level {level}.")
            updated_leg_chain = self.solve_level(leg_chain, level)
            return self.get_all_estimated_times_with_slack(updated_leg_chain, level + 1)

    def get_all_adjusted_times_with_slack(self, leg_chain, real_total_time):
        """
        When the real total time is known, this function can be used to adjust the estimated times with slack.
        This guarantees that a valid leg chain can be built.
        This method ties it all together.
        """
        df, highest_level = self.get_all_estimated_times_with_slack(leg_chain)

        if f'level_{highest_level}' in df.columns:
            highest_lvl_leg_index = df[df[f'level_{highest_level}'].notna()].index
        # Fallback for when the highest level is 0 and similar cases
        elif f'level_{highest_level - 1}' in df.columns:
            highest_level = highest_level - 1
            highest_lvl_leg_index = df[df[f'level_{highest_level}'].notna()].index
        elif s.LEG_DURATION_MINUTES_COL in df.columns:
            highest_level = 0
            return df, highest_level
        else:
            raise ValueError(f"Could not find any column to adjust.")

        # There should be one time in the highest level
        if len(highest_lvl_leg_index) != 1:
            logger.error(f"Expected 1 leg in highest level, found {len(highest_lvl_leg_index)}.")

        # Adjust the highest level
        df.at[highest_lvl_leg_index[0], f'level_{highest_level}'] = real_total_time

        # If highest level is 1, we're done
        if highest_level == 1:
            return df, highest_level

        # Get highest level bounds
        lower_bound = df.at[highest_lvl_leg_index[0], f'level_{highest_level}_lower_bound']
        upper_bound = df.at[highest_lvl_leg_index[0], f'level_{highest_level}_upper_bound']

        # Adjust all else if necessary
        if lower_bound <= real_total_time <= upper_bound:
            logger.debug(f"Real total time is within bounds of highest level.")
            return df, highest_level
        else:
            logger.debug(f"Real total time is outside bounds of highest level, adjusting...")

            # One below highest level down to including level 1
            for level in range(highest_level - 1, 0, -1):

                # Pair legs the same way as in the solver
                times_col = f"level_{level}"  # Here we expect some NaN values

                legs_to_process = df[df[times_col].notna()].copy()
                legs_to_process['original_index'] = legs_to_process.index

                # Reset index for reliable pairing
                legs_to_process.reset_index(drop=True, inplace=True)
                legs_to_process['pair_id'] = legs_to_process.index // 2

                for pair_id, group in legs_to_process.groupby('pair_id'):
                    if len(group) == 1:
                        continue
                    else:
                        # Get the higher leg index and its values
                        higher_leg_index = group['original_index'].iloc[-1]
                        higher_leg_value = df.at[higher_leg_index, f'level_{level + 1}']
                        higher_leg_lower_bound = df.at[higher_leg_index, f'level_{level + 1}_lower_bound']
                        higher_leg_upper_bound = df.at[higher_leg_index, f'level_{level + 1}_upper_bound']

                        if higher_leg_value < higher_leg_lower_bound:  # We have strongly overshot because the REAL higher value is lower than the lower bound
                            # Make longer leg shorter, shorter leg longer
                            leg1_value = df.at[group['original_index'].iloc[0], f'level_{level}']
                            leg2_value = df.at[group['original_index'].iloc[1], f'level_{level}']

                            if leg1_value > leg2_value:
                                L_bounds1 = abs(
                                    df.at[group['original_index'].iloc[0], f'level_{level}_lower_bound'] - leg1_value)
                                L_bounds2 = abs(
                                    df.at[group['original_index'].iloc[1], f'level_{level}_upper_bound'] - leg2_value)

                                delta_L_high = abs(higher_leg_value - higher_leg_lower_bound)

                                delta_L1 = (L_bounds1 * delta_L_high * (leg1_value + leg2_value) ** 2) / (
                                        higher_leg_value * (leg1_value * L_bounds1 + leg2_value * L_bounds2))
                                delta_L2 = (L_bounds2 * delta_L_high * (leg1_value + leg2_value) ** 2) / (
                                        higher_leg_value * (leg1_value * L_bounds1 + leg2_value * L_bounds2))

                                df.loc[group['original_index'].iloc[0], f'level_{level}'] -= delta_L1
                                df.loc[group['original_index'].iloc[1], f'level_{level}'] += delta_L2
                            else:
                                L_bounds1 = abs(
                                    df.at[group['original_index'].iloc[0], f'level_{level}_upper_bound'] - leg1_value)
                                L_bounds2 = abs(
                                    df.at[group['original_index'].iloc[1], f'level_{level}_lower_bound'] - leg2_value)

                                delta_L_high = abs(higher_leg_value - higher_leg_lower_bound)

                                delta_L1 = (L_bounds1 * delta_L_high * (leg1_value + leg2_value) ** 2) / (
                                        higher_leg_value * (leg1_value * L_bounds1 + leg2_value * L_bounds2))
                                delta_L2 = (L_bounds2 * delta_L_high * (leg1_value + leg2_value) ** 2) / (
                                        higher_leg_value * (leg1_value * L_bounds1 + leg2_value * L_bounds2))

                                df.loc[group['original_index'].iloc[0], f'level_{level}'] += delta_L1
                                df.loc[group['original_index'].iloc[1], f'level_{level}'] -= delta_L2

                        elif higher_leg_value > higher_leg_upper_bound:  # We have strongly undershot because the REAL higher value is higher than the upper bound
                            # Make both legs longer (both move to upper bound)
                            leg1_value = df.at[group['original_index'].iloc[0], f'level_{level}']
                            leg2_value = df.at[group['original_index'].iloc[1], f'level_{level}']

                            L_bounds1 = abs(
                                df.at[group['original_index'].iloc[0], f'level_{level}_upper_bound'] - leg1_value)
                            L_bounds2 = abs(
                                df.at[group['original_index'].iloc[1], f'level_{level}_upper_bound'] - leg2_value)

                            delta_L_high = abs(higher_leg_value - higher_leg_upper_bound)

                            delta_L1 = (L_bounds1 * delta_L_high * (leg1_value + leg2_value) ** 2) / (
                                    higher_leg_value * (leg1_value * L_bounds1 + leg2_value * L_bounds2))
                            delta_L2 = (L_bounds2 * delta_L_high * (leg1_value + leg2_value) ** 2) / (
                                    higher_leg_value * (leg1_value * L_bounds1 + leg2_value * L_bounds2))

                            df.loc[group['original_index'].iloc[0], f'level_{level}'] += delta_L1
                            df.loc[group['original_index'].iloc[1], f'level_{level}'] += delta_L2

                        else:
                            return df, highest_level

            return df, highest_level

    def solve_level(self, leg_chain, level):
        """
        Adds a column for given level of slack factor calculation, containing the time with slack of all legs up to and
        including the entry. For better performance, keep the number of columns to a minimum.
        :param leg_chain: df
        :param level: Next level to calculate
        :return: copy of leg_chain with added column
        """
        leg_chain = leg_chain.copy()
        leg_chain[f"level_{level + 1}"] = np.nan

        if level == 0:
            times_col = s.LEG_DURATION_MINUTES_COL
            if leg_chain[times_col].notna().sum() != len(leg_chain):
                logger.warning(f"Found NaN values in {times_col}, may produce incorrect results.")
        else:
            times_col = f"level_{level}"  # Here we expect some NaN values

        legs_to_process = leg_chain[leg_chain[times_col].notna()].copy()
        legs_to_process['original_index'] = legs_to_process.index

        # Reset index for reliable pairing
        legs_to_process.reset_index(drop=True, inplace=True)
        legs_to_process['pair_id'] = legs_to_process.index // 2

        for pair_id, group in legs_to_process.groupby('pair_id'):
            if len(group) == 1:
                time = group.iloc[0][times_col]
                upper_bound = time
                lower_bound = time
            else:
                time = self.calculate_expected_time_with_slack(group.iloc[0][times_col],
                                                               group.iloc[1][times_col],
                                                               group.iloc[0][s.ACT_FROM_INTERNAL_COL],
                                                               group.iloc[0][s.ACT_TO_INTERNAL_COL],
                                                               group.iloc[1][s.ACT_TO_INTERNAL_COL])

                upper_bound = (group.iloc[0][times_col] + group.iloc[1][times_col])
                lower_bound = (abs(group.iloc[0][times_col] - group.iloc[1][times_col]))

            leg_chain.loc[group['original_index'].iloc[-1], f"level_{level + 1}"] = time
            leg_chain.loc[group['original_index'].iloc[-1], f"level_{level + 1}_upper_bound"] = upper_bound
            leg_chain.loc[group['original_index'].iloc[-1], f"level_{level + 1}_lower_bound"] = lower_bound

        return leg_chain


class TTMatrices:
    """
    Manages travel time matrices for various modes of transportation.
    """

    def __init__(self, car_tt_matrices_csv_paths: List[str], pt_tt_matrices_csv_paths: List[str],
                 bike_tt_matrix_csv_path: str, walk_tt_matrix_csv_path: str):
        self.tt_matrices: Dict[str, Union[Dict[str, pd.DataFrame], pd.DataFrame, None]] = {'car': {}, 'pt': {},
                                                                                           'bike': None,
                                                                                           'walk': None}

        # Read car and pt matrices for each hour
        for mode, csv_paths in zip(['car', 'pt'], [car_tt_matrices_csv_paths, pt_tt_matrices_csv_paths]):
            for hour, path in enumerate(csv_paths):
                try:
                    self.tt_matrices[mode][str(hour)] = read_csv(path)
                except Exception as e:
                    logger.error(f"Error reading {mode} matrix for hour {hour}: {e}")
                    raise e

        # Read bike and walk matrices
        try:
            self.tt_matrices['bike'] = read_csv(bike_tt_matrix_csv_path)
            self.tt_matrices['walk'] = read_csv(walk_tt_matrix_csv_path)
        except Exception as e:
            logger.error(f"Error reading bike/walk matrices: {e}")
            raise e

        # Validation
        tt_rows_num: int = len(self.tt_matrices['car']['0'])

        for mode, matrices in self.tt_matrices.items():
            if mode in ['bike', 'walk']:
                if len(matrices) != 1:
                    logger.warning(f"Expected 1 {mode} matrix, found {len(matrices)}")
            else:
                if len(matrices) != 24:
                    logger.warning(f"Expected 24 {mode} matrices, found {len(matrices)}")

        logger.info(f"Loaded travel time matrices for {len(self.tt_matrices['car'])} hours.")

    def get_tt_matrix(self, mode: str, hour: int = None):
        """
        Retrieve the travel time matrix for a given mode and hour.
        :param mode: car, pt, bike, walk
        :param hour: 0 - 23
        :return:
        """
        if mode == "ride":
            mode = "car"
        if mode not in ['car', 'pt', 'bike', 'walk']:
            logger.error(f"Invalid mode: {mode}. Choices are 'car', 'pt', 'bike', 'walk'.")
            return self.tt_matrices['walk']

        if mode in ['car', 'pt']:
            if hour is None:
                logger.error(f"Hour must be specified for mode {mode}. Returning walk matrix.")
                return self.tt_matrices["walk"]
            return self.tt_matrices[mode].get(str(hour))

        return self.tt_matrices[mode]

    def get_weighted_tt_matrix_two_modes(self, mode1, weight1, mode2, weight2, hour=None):

        total_weight = weight1 + weight2
        weight1 = weight1 / total_weight
        weight2 = weight2 / total_weight

        tt_matrix1 = self.get_tt_matrix(mode1, hour)
        tt_matrix2 = self.get_tt_matrix(mode2, hour)

        if tt_matrix1 is None or tt_matrix2 is None:
            raise ValueError("One or both of the travel time matrices could not be retrieved.")

        weighted_tt_matrix = tt_matrix1.copy()
        weighted_tt_matrix['VALUE'] = tt_matrix1['VALUE'].multiply(weight1).add(tt_matrix2['VALUE'].multiply(weight2))

        return weighted_tt_matrix

    def get_weighted_tt_matrix_n_modes(self, mode_weights: Dict[str, float], hour: int = None) -> pd.DataFrame:
        """
        Get a weighted travel time matrix for multiple modes.
        :param mode_weights: Dictionary with mode names as keys and weights as values.
        :param hour: Hour of the day for modes with time-dependent matrices (car, pt). 0-23.
        """
        weighted_tt_matrix = None
        total_weight = sum(mode_weights.values())

        for mode, weight in mode_weights.items():
            tt_matrix = self.get_tt_matrix(mode, hour)
            weighted_matrix = tt_matrix['VALUE'] * (weight / total_weight)

            if weighted_tt_matrix is None:
                weighted_tt_matrix = tt_matrix
                weighted_tt_matrix['VALUE'] = weighted_matrix
            else:
                weighted_tt_matrix['VALUE'] += weighted_matrix

        return weighted_tt_matrix

    def get_travel_time(self, cell_from, cell_to, mode, hour=None):
        """
        Get the travel time between two cells for a specified mode and time of day.

        :param cell_from: Starting cell id.
        :param cell_to: Destination cell id.
        :param mode: Mode of transportation (car, pt, bike, walk).
        :param hour: Hour of the day for modes with time-dependent matrices (car, pt). 0-23.
        :return: Travel time between the two cells.
        """
        if mode == "ride":
            mode = "car"
        if mode not in ['car', 'pt', 'bike', 'walk']:
            logger.error(f"Invalid mode: {mode}. Choices are 'car', 'pt', 'bike', 'walk'.")
            return 20

        tt_matrix = self.get_tt_matrix(mode, hour)

        filtered_matrix = tt_matrix[(tt_matrix['FROM'] == cell_from) & (tt_matrix['TO'] == cell_to)]

        # Extract travel time from the filtered row
        if not filtered_matrix.empty:
            travel_time = filtered_matrix['VALUE'].iloc[0]
        else:
            logger.error(
                f"No travel time found for cells {cell_from} and {cell_to}. Using default value of 20 minutes.")
            travel_time = 20

        return travel_time

    def get_mode_weighted_travel_time(self, cell_from, cell_to, mode_weights: Dict[str, float], hour: int = None):
        travel_time = None
        total_weight = sum(mode_weights.values())

        for mode, weight in mode_weights.items():
            time = self.get_travel_time(cell_from, cell_to, mode, hour)
            weighted_time = time * (weight / total_weight)

            if travel_time is None:
                travel_time = weighted_time
            else:
                travel_time += weighted_time

        return travel_time


def sigmoid(x, beta, delta_T):
    """
    Sigmoid function for likelihood calculation.

    :param x: The input value (time differential) - can be a number, list, or numpy array.
    :param beta: Controls the steepness of the sigmoid's transition.
    :param delta_T: The midpoint of the sigmoid's transition.
    :return: Sigmoid function value.
    """
    x = np.array(x)  # Ensure x is a numpy array
    z = -beta * (x - delta_T)
    # Use np.clip to limit the values in z to avoid overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(z))


def check_distance(leg_to_find, leg_to_compare):
    distance_to_find = leg_to_find[s.LEG_DISTANCE_METERS_COL]
    distance_to_compare = leg_to_compare[s.LEG_DISTANCE_METERS_COL]

    if pd.isnull(distance_to_find) or pd.isnull(distance_to_compare):
        return False

    difference = abs(distance_to_find - distance_to_compare)
    range_tolerance = distance_to_find * 0.05

    return difference <= range_tolerance


def check_time(leg_to_find, leg_to_compare):
    # Using constant variables instead of strings
    leg_begin_to_find = leg_to_find[s.LEG_START_TIME_COL]
    leg_end_to_find = leg_to_find[s.LEG_END_TIME_COL]
    leg_begin_to_compare = leg_to_compare[s.LEG_START_TIME_COL]
    leg_end_to_compare = leg_to_compare[s.LEG_END_TIME_COL]

    # Reduce the time range for short legs to avoid false positives (NaN evaluates to False)
    time_range = pd.Timedelta(minutes=5) if leg_to_find[s.LEG_DURATION_MINUTES_COL] > 5 and leg_to_compare[
        s.LEG_DURATION_MINUTES_COL] > 5 else pd.Timedelta(minutes=2)

    if pd.isnull([leg_begin_to_find, leg_end_to_find, leg_begin_to_compare, leg_end_to_compare]).any():
        return False

    begin_difference = abs(leg_begin_to_find - leg_begin_to_compare)
    end_difference = abs(leg_end_to_find - leg_end_to_compare)

    return (begin_difference <= time_range) and (end_difference <= time_range)


def check_mode(leg_to_find, leg_to_compare):
    """
    Check if the modes of two legs are compatible.
    Note: Adjusting the mode "car" to "ride" based on age is now its own function.
    :param leg_to_find:
    :param leg_to_compare:
    :return:
    """
    mode_to_find = leg_to_find[s.MODE_INTERNAL_COL]
    mode_to_compare = leg_to_compare[s.MODE_INTERNAL_COL]

    if mode_to_find == mode_to_compare and mode_to_find != s.MODE_UNDEFINED:  # Make sure we don't pair undefined modes
        return True

    mode_pairs = {(s.MODE_CAR, s.MODE_RIDE), (s.MODE_RIDE, s.MODE_CAR),
                  (s.MODE_WALK, s.MODE_BIKE), (s.MODE_BIKE, s.MODE_WALK)}
    if (mode_to_find, mode_to_compare) in mode_pairs:
        return True

    if s.MODE_UNDEFINED in [mode_to_find, mode_to_compare]:
        # Assuming if one mode is undefined and the other is car, they pair as ride
        # The mode is not updated here (in contrast to prev. work), because we don't know yet if the leg is connected.
        return s.MODE_CAR in [mode_to_find, mode_to_compare]

    return False


def check_activity(leg_to_find, leg_to_compare):
    compatible_activities = {
        s.ACT_SHOPPING: [s.ACT_ERRANDS],
        s.ACT_ERRANDS: [s.ACT_SHOPPING, s.ACT_LEISURE],
        s.ACT_LEISURE: [s.ACT_ERRANDS, s.ACT_SHOPPING, s.ACT_MEETUP],
        s.ACT_MEETUP: [s.ACT_LEISURE]}

    type_to_find = leg_to_find[s.ACT_TO_INTERNAL_COL]
    type_to_compare = leg_to_compare[s.ACT_TO_INTERNAL_COL]

    if (type_to_find == type_to_compare or
            s.ACT_ACCOMPANY_ADULT in [type_to_find, type_to_compare] or
            s.ACT_PICK_UP_DROP_OFF in [type_to_find, type_to_compare]):
        return True
    elif s.ACT_UNSPECIFIED in [type_to_find, type_to_compare] or pd.isnull([type_to_find, type_to_compare]).any():
        logger.debug("Activity Type Undefined or Null (which usually means person has no legs).")
        return False
    # Assuming trip home (works, but not really plausible, thus commented out for now)
    # elif (type_to_find == s.ACTIVITY_HOME and type_to_compare != s.ACTIVITY_WORK) or \
    #         (type_to_compare == s.ACTIVITY_HOME and type_to_find != s.ACTIVITY_WORK):
    #     return True

    return type_to_compare in compatible_activities.get(type_to_find, [])


class Capacities:
    """
    Turns given capacities into point capacities that the locator can handle.
    - Loads data as either shp or csv (either points or cells shp must be given)
    - Translates between internal activity types and given capacities.
    - If cell capacities and shp point capacities are given, weighted-distributes the cell capacities to the points.
    - If cell capacities and shp points with possible activities are given, distributes the cell capacities to the points accordingly.
    - If cell capacities and shp raw points are given, distributes the cell capacities to the points evenly.
    - If cell capacities and csv point capacities are given, weighted-distributes; and creates random points.
    - If shp point capacities are given, uses those directly.

    The result are always located point capacities with the best possible information, that can be used by the activity placer.
    """

    def __init__(self, capa_cells_shp_path: str = None, capa_points_shp_path: str = None,
                 capa_cells_csv_path: str = None,
                 capa_points_csv_path: str = None):  # Flexibility not implemented fully

        logger.info("Initializing capacities...")
        # if capa_points_shp_path is not None:
        #     if capa_cells_shp_path is not None:
        #         self.capa_points_gdf = gpd.read_file(capa_points_shp_path)
        #         self.capa_cells_gdf = gpd.read_file(capa_cells_shp_path)
        #         self.capa_points_gdf = distribute_by_weights(self.capa_points_gdf, self.capa_cells_gdf, 'cell_id')
        #     elif capa_cells_csv_path is not None:
        #         self.capa_points_gdf = gpd.read_file(capa_points_shp_path)
        #         self.capa_cells_df = read_csv(capa_cells_csv_path)
        #         self.capa_points_gdf = distribute_by_weights(self.capa_points_gdf, self.capa_cells_df, 'cell_id')
        #     else:
        #         self.capa_points_gdf = gpd.read_file(capa_points_shp_path)
        #
        # elif capa_cells_shp_path is not None:
        #     if capa_points_csv_path is not None:
        #         self.capa_cells_gdf = gpd.read_file(capa_cells_shp_path)
        #         self.capa_points_df = read_csv(capa_points_csv_path)
        #         self.capa_points_df['geometry'] = self.capa_points_df['geometry'].apply(shapely.wkt.loads)
        #         self.capa_points_gdf = gpd.GeoDataFrame(self.capa_points_df, geometry='geometry')
        #         self.capa_points_gdf.crs = self.capa_cells_gdf.crs
        #         self.capa_points_gdf = distribute_by_weights(self.capa_points_gdf, self.capa_cells_gdf, 'cell_id')
        #     else:
        #         self.capa_cells_gdf = gpd.read_file(capa_cells_shp_path)
        #         self.capa_points_gdf = self.capa_cells_gdf.copy()
        #         self.capa_points_gdf['geometry'] = self.capa_points_gdf['geometry'].apply(random_point_in_polygon)
        #
        # else:
        #     raise ValueError("Either capa_points_shp_path or capa_cells_shp_path must be given.")
        # logger.info(f"Created capacity_gdf for {len(self.capa_points_gdf)} points.")

        # Load cells
        # if capa_cells_shp_path is None:
        #     capa_cells_shp_path = s.CAPA_CELLS_SHP_PATH
        if capa_cells_csv_path is None:
            capa_cells_csv_path = s.CAPA_CELLS_CSV_PATH
        # self.capa_cells_gdf = gpd.read_file(capa_cells_shp_path, usecols=["NAME"])
        self.capa_csv_df = read_csv(capa_cells_csv_path)

        # ensure_crs(self.capa_cells_gdf)
        #
        # # Add the cols from the csv to the gdf
        # for col in self.capa_csv_df.columns:
        #     if col not in self.capa_cells_gdf.columns:
        #         self.capa_cells_gdf[col] = self.capa_csv_df[col]

        self.translate_and_split_potentials()

        logger.info("Initialized capacities.")

    def translate_and_split_potentials(self, translation_dict=None):
        logger.info("Translating capacities...")
        if translation_dict is None:
            translation_dict = {
                "ID2020_~20": {"1": 1},
                "ID2020_~21": {"1": 1},
                "ID2020_~22": {"1": 1},
                "ID2020_~23": {"15": 0.01},
                "ID2020_~24": {"4": 0.8, "5": 0.2},
                "ID2020_~25": {"11": 0.5, "3": 0.5},
                "ID2020_~26": {"3": 1},
                "ID2020_~27": {"2": 0.1, "4": 0.9},
                "ID2020_~28": {"4": 1},
                "ID2020_~29": {"4": 1},
                "ID2020_~30": {"4": 1},
                "ID2020_~31": {"4": 0.8, "5": 0.2},
                "ID2020_~32": {"7": 1},
                "ID2020_~33": {"7": 0.7, "15": 0.3},
                "ID2020_~34": {"11": 1},
                "ID2020_~35": {"3": 1},
                "ID2020_~36": {"12": 1},
                "ID2020_~37": {"7": 1},
                "ID2020_~38": {"5": 1},
                "ID2020_~39": {"5": 0.8, "2": 0.1, "16": 0.1},
                "ID2020_~40": {"3": 1},
                "ID2020_~41": {"3": 1},
                "ID2020_~42": {"1": 0.5, "5": 0.5},
                "ID2020_~43": {"14": 1},
                "ID2020_~44": {"3": 1},
            }

        for original_col, translations in translation_dict.items():
            for new_col, weight in translations.items():
                if new_col not in self.capa_csv_df.columns:
                    self.capa_csv_df[new_col] = 0
                print(self.capa_csv_df[original_col])
                self.capa_csv_df[new_col] += self.capa_csv_df[original_col].fillna(0) * weight
            self.capa_csv_df.drop(columns=[original_col], inplace=True)

        logger.info("Translated capacities.")

    def round_capacities(self):
        """
        Round the capacities while preserving the sum.
        """
        logger.info("Rounding capacities to integers while preserving the sum...")
        for col in self.capa_csv_df.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(self.capa_csv_df[col]):
                continue

            # Calculate the sum before rounding
            sum_before_rounding = self.capa_csv_df[col].sum()

            # Round down the values in the column and keep track of the decimal part
            self.capa_csv_df[col], decimal_part = divmod(self.capa_csv_df[col], 1)

            # Calculate the difference between the sum before rounding and the sum after rounding
            diff = sum_before_rounding - self.capa_csv_df[col].sum()

            # Adjust the rounded values to preserve the sum
            while diff > 0:
                # Find the index of the maximum decimal part
                idx_max_decimal = decimal_part.idxmax()
                # Add 1 to the corresponding value in the DataFrame
                self.capa_csv_df.loc[idx_max_decimal, col] += 1
                # Subtract 1 from the corresponding value in the decimal part Series
                decimal_part.loc[idx_max_decimal] -= 1
                # Subtract 1 from the difference
                diff -= 1
        logger.info("Rounded capacities.")

    def get_capacity(self, activity_type, cell_name):
        """
        Get a capacity value from the GeoDataFrame based on the activity type and cell name.

        :param activity_type: The type of activity.
        :param cell_name: The name of the cell.
        :return: The capacity value.
        """
        # Turn activity type into a string
        try:
            activity_type = str(int(activity_type))
        except Exception:  # go on
            logger.debug(f"Activity type {activity_type} was not converted.")

        filtered_df = self.capa_csv_df[self.capa_csv_df['NAME'] == cell_name]

        if filtered_df.empty:
            return 0
        try:
            return filtered_df[activity_type].values[0]
        except KeyError:
            logger.debug(f"Invalid activity type for capacity: {activity_type}")
            return 0

    def decrement_capacity(self, activity_type, cell_name):
        """
        Decrements the capacity for a given activity type and cell.

        :param activity_type: The type of activity.
        :param cell_name: The name of the cell.
        """
        try:
            activity_type = str(int(activity_type))
        except Exception:  # go on
            logger.debug(f"Activity type {activity_type} was not converted.")

        if activity_type not in self.capa_csv_df.columns:
            logger.debug(f"Invalid activity type for capacity: {activity_type}. Not decrementing.")
            return

        filtered_df = self.capa_csv_df[self.capa_csv_df['NAME'] == cell_name]

        if not filtered_df.empty:
            self.capa_csv_df.loc[self.capa_csv_df['NAME'] == cell_name, activity_type] -= 1
        else:
            logger.error(f"No capacity found for cell: {cell_name}. Not decrementing.")

    def normalize_capacities(self, total_values):
        """
        Normalize the capacities for each activity type so that the sum for each activity type equals the corresponding total value.

        :param total_values: A dictionary where the keys are the activity types and the values are the total capacities for each activity type.
        """
        logger.info("Normalizing capacities...")
        # Convert all keys to int, then strings (from float)
        total_values = {str(int(k)): v for k, v in total_values.items()}

        # Normalize the individual capacities
        for activity_type, total_value in total_values.items():
            if activity_type in self.capa_csv_df.columns:
                total_capacity = self.capa_csv_df[activity_type].sum()

                # Normalize the capacities for the activity type
                self.capa_csv_df[activity_type] = (self.capa_csv_df[activity_type] / total_capacity) * total_value
        logger.info("Normalized capacities.")
        self.round_capacities()  # Smart-round the capacities to ints while preserving the sum


def convert_to_list(s):
    """
    This is weirdly and annoyingly necessary because of the way the lists are stored.
    This is needed to correctly convert the string representation of a list to an actual list.
    """
    if pd.isna(s):
        return s
    try:
        # Replace unwanted characters
        cleaned_string = s.replace("\'", "\"")
        # Use json.loads to correctly interpret the string as a list of strings
        return json.loads(cleaned_string)
    except ValueError:
        return s


def add_from_coord(df):
    """
    Add a 'from_activity' column to the DataFrame, which is the to_activity of the previous leg.
    For the first leg of each person, set 'from_activity' based on 'starts_at_home' (-> home or unspecified).
    :return:
    """
    logger.info("Adding/updating from_coord column...")
    # Sort the DataFrame by person ID and leg number (the df should usually already be sorted this way)
    df.sort_values(by=[s.UNIQUE_P_ID_COL, s.LEG_NON_UNIQUE_ID_COL], inplace=True)

    # Shift the 'to_activity' down to create 'from_activity' for each group
    df[s.COORD_FROM_COL] = df.groupby(s.PERSON_ID_COL)[s.COORD_TO_COL].shift(1)

    # For the first leg of each person, set 'from_coord' to home coord
    df.loc[(df[s.LEG_NON_UNIQUE_ID_COL] == 1), s.COORD_FROM_COL] = df.loc[
        (df[s.LEG_NON_UNIQUE_ID_COL] == 1), 'home_loc']

    logger.info("Done.")
    return df


def add_from_location(df, col_to, col_from, backup_existing_from_col=False):
    """
    Add a 'from_activity' column to the DataFrame, which is the to_activity of the previous leg.
    For the first leg of each person, set 'from_activity' based on 'starts_at_home' (-> home or unspecified).
    :return:
    """
    logger.info("Adding/updating from_coord column...")

    if backup_existing_from_col and col_from in df.columns:
        col_from_old = col_from + "_old"
        df[col_from_old] = df[col_from]

    # Sort the DataFrame by person ID and leg number (the df should usually already be sorted this way)
    df.sort_values(by=[s.UNIQUE_P_ID_COL, s.LEG_NON_UNIQUE_ID_COL], inplace=True)

    # Shift the 'to_activity' down to create 'from_activity' for each group
    df[col_from] = df.groupby(s.PERSON_ID_COL)[col_to].shift(1)

    # For the first leg of each person, set 'from_coord' to home coord
    df.loc[(df[s.LEG_NON_UNIQUE_ID_COL] == 1), col_from] = df.loc[
        (df[s.LEG_NON_UNIQUE_ID_COL] == 1), s.HOME_LOC_COL]

    logger.info("Done.")
    return df


def add_from_cell(df):
    """
    Add a 'from_activity' column to the DataFrame, which is the to_activity of the previous leg.
    For the first leg of each person, set 'from_activity' based on 'starts_at_home' (-> home or unspecified).
    :return:
    """
    logger.info("Adding/updating from_coord column...")
    # Sort the DataFrame by person ID and leg number (the df should usually already be sorted this way)
    df.sort_values(by=[s.UNIQUE_P_ID_COL, s.LEG_NON_UNIQUE_ID_COL], inplace=True)

    df[s.CELL_FROM_COL] = df.groupby(s.PERSON_ID_COL)[s.CELL_TO_COL].shift(1)

    # For the first leg of each person, set 'from_cell' to home cell
    df.loc[(df[s.LEG_NON_UNIQUE_ID_COL] == 1), s.CELL_FROM_COL] = df.loc[
        (df[s.LEG_NON_UNIQUE_ID_COL] == 1), s.HOME_CELL_COL]

    logger.info("Done.")
    return df


def add_from_cell_fast(person):
    """
    Add/update 'from_activity', which is the to_activity of the previous leg.
    For the first leg of each person, set 'from_activity' based on 'starts_at_home' (-> home or unspecified).
    Only for a single person.
    :return:
    """
    person = person.copy()
    logger.debug("Adding/updating from_coord column for single person...")
    # Sort the DataFrame by person ID and leg number (the df should usually already be sorted this way)
    person.sort_values(by=[s.LEG_NON_UNIQUE_ID_COL], inplace=True, ignore_index=True)

    person[s.CELL_FROM_COL] = person[s.CELL_TO_COL].shift(1)

    # For the first leg of the person, set 'from_cell' to home cell
    person.loc[(person[s.LEG_NON_UNIQUE_ID_COL] == 1), s.CELL_FROM_COL] = person.loc[
        (person[s.LEG_NON_UNIQUE_ID_COL] == 1), s.HOME_CELL_COL]

    return person


def find_nan_chains(df, column_name):
    """
    Finds chains of consecutive NaN values in the specified column of a DataFrame.
    Each chain includes the row directly after the NaN values.

    :param df: The DataFrame to search for NaN chains.
    :param column_name: The name of the column to check for NaN values.
    :return: A list of DataFrames, each containing a NaN chain and the row after.
    """
    is_nan = df[column_name].isna()
    starts_nan_chain = is_nan & (~is_nan.shift(fill_value=False))

    start_indices = df[starts_nan_chain].index
    chain_dfs = []

    for start_idx in start_indices:
        # Find the end of the NaN chain
        end_idx = start_idx
        while end_idx in df.index and pd.isna(df.loc[end_idx, column_name]):
            end_idx += 1

        if end_idx in df.index:
            chain_df = df.loc[start_idx:end_idx]
        else:  # If the NaN chain is at the end of the DataFrame
            chain_df = df.loc[start_idx:end_idx - 1]

        chain_dfs.append(chain_df)

    return chain_dfs


def assign_points(df, shapefile_path, df_cell_name_column, gdf_cell_name_column, point_column_name):
    """
    Assigns a random point within the cell of a shapefile to rows in the DataFrame,
    while preserving existing points in a specified column.

    Parameters:
    df (DataFrame): The main DataFrame with a column containing cell names.
    shapefile_path (str): Path to the shapefile.
    df_cell_name_column (str): Column in DataFrame that contains cell names.
    gdf_cell_name_column (str): Column in GeoDataFrame that contains cell names.
    point_column_name (str): Column in DataFrame where random points will be assigned.

    Returns:
    DataFrame: The DataFrame with an updated column for random points.
    """
    logger.info("Assigning points based on cells...")
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    logger.info(f"Loaded shapefile with {len(gdf)} cells.")
    # Ensure that the cell name column exists in both DataFrame and GeoDataFrame
    if df_cell_name_column not in df.columns or gdf_cell_name_column not in gdf.columns:
        raise ValueError(
            f"Column '{df_cell_name_column}' must exist in DataFrame and '{gdf_cell_name_column}' must exist in GeoDataFrame")

    # Function to generate a random point within a cell
    def random_point_in_cell(cell):
        minx, miny, maxx, maxy = cell.bounds
        while True:
            pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if cell.contains(pnt):
                return pnt

    # Create a dictionary to map cell names to their geometries
    cell_geometry_map = gdf.set_index(gdf_cell_name_column)['geometry'].to_dict()

    # Initialize the point column if not exists
    if point_column_name not in df.columns:
        df[point_column_name] = pd.NA

    # Iterating over each row in the DataFrame TESTING
    for index, row in df.iterrows():
        cell_name = row[df_cell_name_column]
        if pd.isna(row[point_column_name]):
            if cell_name in cell_geometry_map:
                # Debug print statements can be added here to check the process
                logger.debug(f"Assigning point for cell: {cell_name}")
                point = random_point_in_cell(cell_geometry_map[cell_name])
                df.at[index, point_column_name] = point

        else:
            logger.debug(f"Skipping assignment for cell: {cell_name}")

    return df


def translate_column(df: pd.DataFrame, source_col: str, new_col: str, value_type: str, from_key: str,
                     to_key: str) -> pd.DataFrame:
    """
    Generalized method for translating columns.
    :param df: pandas DataFrame
    :param source_col: The name of the source column in the DataFrame
    :param new_col: The name of the new column in the DataFrame
    :param value_type: The type of values to translate (e.g., 'modes', 'activities')
    :param from_key: The key in the value map to translate from (e.g., 'input_travel_survey', 'internal')
    :param to_key: The key in the value map to translate to (e.g., 'internal', 'MATSim')
    :return: DataFrame with a new column containing the translated values
    """
    translation_dict = {v[from_key]: v[to_key] for k, v in s.VALUE_MAPS[value_type].items() if v[from_key] is not None}

    logger.info(f"Translating column '{source_col}' to '{new_col}' using provided dictionary.")

    # Perform translation
    df[new_col] = df[source_col].map(translation_dict)

    # Identify values that were not found in the dictionary
    missing_values = df[df[new_col].isna()][source_col].unique()
    if len(missing_values) > 0:
        stats_tracker.log("Missing_translations", missing_values)
        logger.warning(
            f"Missing translations for {len(missing_values)} unique values in column '{source_col}': {missing_values}")

    return df


def generate_unique_household_id(df):
    col_name = s.UNIQUE_HH_ID_COL
    logger.info(f"Generating unique household IDs...")
    if col_name not in df.columns:
        df[col_name] = df[s.HOUSEHOLD_MID_ID_COL].astype(str) + "_" + df.index.astype(str)
        logger.info(f"Created new column {col_name}.")
    else:
        df[col_name] = df[s.HOUSEHOLD_MID_ID_COL].astype(str) + "_" + df.index.astype(str)
        logger.info(f"Overwrote existing column {col_name}.")
    return df


def generate_unique_person_id(df):
    col_name = s.UNIQUE_P_ID_COL
    logger.info(f"Generating unique person IDs...")
    if col_name not in df.columns:
        df[col_name] = df[s.UNIQUE_HH_ID_COL] + "_" + df[s.PERSON_ID_COL].astype(str)
        logger.info(f"Created new column {col_name}.")
    else:
        df[col_name] = df[s.UNIQUE_HH_ID_COL] + "_" + df[s.PERSON_ID_COL].astype(str)
        logger.info(f"Overwrote existing column {col_name}.")
    return df


def generate_unique_leg_id(df):
    col_name = s.UNIQUE_LEG_ID_COL
    logger.info(f"Generating unique leg IDs...")
    if col_name not in df.columns:
        df[col_name] = df[s.UNIQUE_P_ID_COL] + "_" + df[s.LEG_NON_UNIQUE_ID_COL].astype(str)
        logger.info(f"Created new column {col_name}.")
    else:
        df[col_name] = df[s.UNIQUE_P_ID_COL] + "_" + df[s.LEG_NON_UNIQUE_ID_COL].astype(str)
        logger.info(f"Overwrote existing column {col_name}.")
    return df


def create_output_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")
    return OUTPUT_DIR


def build_estimation_tree(distances: List[float]) -> List[List[List[float]]]:  # Tree level, Leg, Lengths
    """
    Build a tree of estimated distances from a list of distances.
    In helpers because it's used in multiple places.
    :param distances:
    :return:
    Example output:

    """
    logger.debug(f"Building estimation tree for {len(distances)} legs.")
    tree: List[List[List[float]]] = []

    while len(distances) > 1:
        new_distances: List[float] = []
        combined_pairs: List[List[float]] = []
        for i in range(0, len(distances) - 1, 2):
            combined_list: List[float] = estimate_length_with_slack(distances[i], distances[i + 1])
            new_distances.append(combined_list[2])
            combined_pairs.append(combined_list)

        if len(distances) % 2 != 0:
            # Carry over the estimation from the so-far built tree
            last_pair = tree[-1][-1] if tree else [distances[-1], distances[-1], distances[-1], distances[-1],
                                                   distances[-1]]
            combined_pairs.append(last_pair)
            new_distances.append(last_pair[2])  # Append only the center value for next level processing

        distances = new_distances
        tree.append(combined_pairs)

    return tree


def estimate_length_with_slack(length1, length2, slack_factor=2, min_slack_lower=0.2, min_slack_upper=0.2) -> List[
    float]:
    """min_slacks must be between 0 and 0.49"""

    length_sum = length1 + length2  # is also real maximum length
    length_diff = abs(length1 - length2)  # is also real minimum length
    shorter_leg = min(length1, length2)

    result = length_sum / slack_factor

    wanted_minimum = length_diff + shorter_leg * min_slack_lower
    wanted_maximum = length_sum - shorter_leg * min_slack_upper

    if result <= wanted_minimum:
        result = wanted_minimum
    elif result > wanted_maximum:
        result = wanted_maximum

    # assert result is a number
    assert not np.isnan(
        result), f"Result is NaN. Lengths: {length1}, {length2}, Slack factor: {slack_factor}, Min slack lower: {min_slack_lower}, Min slack upper: {min_slack_upper}"

    return [length_diff, wanted_minimum, result, wanted_maximum, length_sum]


def get_files(path: str, get_all: bool = False) -> Union[str, List[str]]:
    path = make_path_absolute(path)

    # Normalize the path to handle different OS path conventions
    normalized_path = os.path.normpath(path)

    # Correcting the path for glob usage
    glob_path = os.path.join(normalized_path, '*')

    # Check if the provided path is a file
    if os.path.isfile(normalized_path):
        return normalized_path

    # Get all files in the folder
    files = glob.glob(glob_path)

    if not files:
        raise FileNotFoundError(f'No files found in the folder: {normalized_path}')

    if get_all:
        return files
    elif len(files) == 1:
        return files[0]
    else:
        # Exclude files starting with 'X' or 'x'
        filtered_files = [f for f in files if not os.path.basename(f).startswith(('X', 'x'))]

        if not filtered_files:
            raise FileNotFoundError(f'No suitable files found in the folder: {normalized_path}')

        # Get the newest file among the remaining files
        newest_file = max(filtered_files, key=os.path.getctime)
        return newest_file

def make_path_absolute(path):
    """
    Make a file or folder path absolute if it isn't.
    """
    if not os.path.isabs(path):
        return os.path.join(PROJECT_ROOT, path)
    return path


def euclidean_distance(start: np.ndarray, end: np.ndarray) -> float:
    """Compute the Euclidean distance between two points."""
    return np.linalg.norm(end - start)


def get_min_max_distance(arr):
    """Get the minimum and maximum possible distance/radius (from a fixed point) given a list of distances.
    Works on integer distances and converts to integers!"""

    if len(arr) == 0:
        raise ValueError("No distances given.")
    if len(arr) == 1:
        return arr[0], arr[0]

    arr = np.array(arr, dtype=int)

    total_sum = sum(arr)

    # Is one leg longer than all others summed?
    remaining_distances = total_sum - arr
    single_leg_overshoot = max(arr - remaining_distances)
    min_diff = max(single_leg_overshoot, 0)

    return min_diff, total_sum

def spread_distances(distance1, distance2, iteration=0, first_step=20, base=1.5):
    """Increases the difference between two distances, keeping them positive."""
    step = first_step * (base ** iteration)
    if distance1 > distance2:
        distance1 += step
        distance2 -= step
    else:
        distance1 -= step
        distance2 += step
    return max(0, distance1), max(0, distance2)

def get_abs_distance_deviations(candidate_coordinates, location, wanted_distance):
    # Handle single-coordinate case by reshaping
    if candidate_coordinates.ndim == 1:  # Single coordinate (1D array)
        candidate_coordinates = candidate_coordinates[np.newaxis, :]  # Make it 2D

    # Calculate distances
    candidate_distances = np.linalg.norm(candidate_coordinates - location, axis=1)
    return np.abs(candidate_distances - wanted_distance)

def get_main_activity_leg(person_legs):
    main_activity_leg = None
    main_activity_index = None
    for i, leg in enumerate(person_legs):
        if leg['is_main_activity']:
            main_activity_leg = leg
            main_activity_index = i
            break

    if not main_activity_leg:
        to_activities = [leg['to_act_type'] for leg in person_legs]
        if isinstance(to_activities, list):
            assert all(act == s.ACT_HOME for act in to_activities), (
                "Person has no main activity but has non-home legs."
            )
        else:
            assert to_activities == s.ACT_HOME, (
                "Person has no main activity but has non-home legs."
            )
        return None, None
    return main_activity_index, main_activity_leg


def filter_feasible_data(data):
    """Only keep persons where all segments are feasible."""
    filtered_data = {}

    for person_id, trips in data.items():
        person_feasible = True  # Assume person is feasible until proven otherwise

        for segments in trips:
            # Extract distances and calculate direct distance for the entire segment
            distances = np.array([leg['distance'] for leg in segments])
            first_location = segments[0]['from_location']
            last_location = segments[-1]['to_location']
            direct_distance = euclidean_distance(first_location, last_location)

            # Check feasibility of this segment group
            if not check_feasibility(distances, direct_distance):
                person_feasible = False
                break  # No need to check further segments for this person

        # If all segments are feasible, include the person in the output
        if person_feasible:
            filtered_data[person_id] = trips

    return filtered_data

def check_feasibility(distances, direct_distance, consider_total_distance=True):
    """
    @Author: Hoerl, Sebastian
    :param distances:
    :param direct_distance:
    :param consider_total_distance:
    :return:
    """
    return calculate_feasibility(distances, direct_distance, consider_total_distance) == 0.0

def calculate_feasibility(distances, direct_distance, consider_total_distance=True):
    """
    @Author: Hoerl, Sebastian
    :param distances:
    :param direct_distance:
    :param consider_total_distance:
    :return:
    """
    # Really elegant way to calculate the feasibility of any chain

    total_distance = np.sum(distances)
    delta_distance = 0.0

    # Remaining is the diff between each individual dist and the sum of all dists (so remaining is the sum of all distances except itself)
    remaining_distance = total_distance - distances
    # So this checks if we can get "back" to the end if one dist is very large and gets us far away
    # If delta is larger than one, we can't get back to the end
    delta = max(distances - direct_distance - remaining_distance)

    # Delta gets positive if the real dist is larger than the sum of all distances
    if consider_total_distance:
        delta = max(delta, direct_distance - total_distance)

    return float(max(delta, 0))
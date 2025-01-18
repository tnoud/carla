import math
import random
import json

from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from utils import helpers as h, pipeline_setup, settings as s
from utils.logger import logging
from utils.stats_tracker import stats_tracker

from typing import NamedTuple, Tuple
from frozendict import frozendict

logger = logging.getLogger(__name__)

class Leg(NamedTuple):
    unique_leg_id: str
    from_location: np.ndarray
    to_location: np.ndarray
    distance: float
    to_act_type: str
    to_act_identifier: str


Segment = Tuple[Leg, ...]  # A segment of a plan (immutable tuple of legs)
SegmentedPlan = Tuple[Segment, ...]  # A full plan split into segments
SegmentedPlans = frozendict[str, SegmentedPlan]  # All agents' plans (person_id -> SegmentedPlan)


class TargetLocations:
    """
    Spatial index of activity locations split by type.
    This class is used to quickly find the nearest activity locations for a given location.
    """

    def __init__(self, json_folder_path: str):
        self.data: Dict[str, Dict[str, np.ndarray]] = self.load_reformatted_osmox_data(h.get_files(json_folder_path))
        self.indices: Dict[str, cKDTree] = {}

        for type, pdata in self.data.items():
            logger.info(f"Constructing spatial index for {type} ...")
            self.indices[type] = cKDTree(pdata["coordinates"])

    @staticmethod
    def load_reformatted_osmox_data(file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Convert lists back to numpy arrays
        for purpose in data:
            for key in data[purpose]:
                data[purpose][key] = np.array(data[purpose][key])
        return data

    def query_closest(self, type: str, location: np.ndarray, num_candidates: int = 1) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the nearest activity locations for one or more points.
        :param type: The type category to query.
        :param location: A 1D numpy array for a single point (e.g., [1.5, 2.5]) or a 2D numpy array for multiple points (e.g., [[1.5, 2.5], [3.0, 4.0]]).
        :param num_candidates: The number of nearest candidates to return.
        :return: A tuple containing numpy arrays: identifiers, coordinates, and potentials of the nearest candidates.
        """
        # Query the KDTree directly (handles both 1D and 2D inputs)
        _, indices = self.indices[type].query(location, k=num_candidates)

        # Retrieve data for the nearest candidates
        data_type = self.data[type]
        return (
            data_type["identifiers"][indices],
            data_type["coordinates"][indices],
            data_type["potentials"][indices],
        )

    def query_within_ring(self, act_type: str, location: np.ndarray, radius1: float, radius2: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the activity locations within a ring defined by two radii around a location and type.
        :param act_type: The activity category to query.
        :param location: A 1D numpy array representing the location to query (coordinates [1.5, 2.5]).
        :param radius1: Any of the two radii defining the ring.
        :param radius2: The other one.
        :return: A tuple containing identifiers, coordinates, and remaining potentials of candidates.
        """
        # Ensure location is a 2D array with a single location
        location = location.reshape(1, -1)

        outer_radius = max(radius1, radius2)
        inner_radius = min(radius1, radius2)

        # Query points within the outer radius
        tree: cKDTree = self.indices[act_type]
        outer_indices = tree.query_ball_point(location, outer_radius)[0]  # Indices of points within the outer radius

        if not outer_indices:
            return None

        # Query points within the inner radius
        inner_indices = tree.query_ball_point(location, inner_radius)[0]  # Indices of points within the inner radius

        # Filter indices to get only points in the annulus
        annulus_indices = list(set(outer_indices) - set(inner_indices))

        if not annulus_indices:
            return None

        # Retrieve corresponding activity data
        data_type = self.data[act_type]
        identifiers = data_type["identifiers"][annulus_indices]
        coordinates = data_type["coordinates"][annulus_indices]
        potentials = data_type["potentials"][annulus_indices]

        return identifiers, coordinates, potentials

    def query_within_two_overlapping_rings(self, act_type: str, location1: np.ndarray, location2: np.ndarray,
                                           radius1a: float, radius1b: float, radius2a: float, radius2b: float,
                                           max_number_of_candidates: int = None):

        location1 = location1[None, :] if location1.ndim == 1 else location1
        location2 = location2[None, :] if location2.ndim == 1 else location2

        outer_radius1, inner_radius1 = max(radius1a, radius1b), min(radius1a, radius1b)
        outer_radius2, inner_radius2 = max(radius2a, radius2b), min(radius2a, radius2b)

        outer_indices1 = self.indices[act_type].query_ball_point(location1, outer_radius1)[0]
        outer_indices2 = self.indices[act_type].query_ball_point(location2, outer_radius2)[0]

        if not outer_indices1 or not outer_indices2:
            return None

        outer_intersection = set(outer_indices1).intersection(outer_indices2)
        if not outer_intersection:
            return None

        inner_indices1 = set(self.indices[act_type].query_ball_point(location1, inner_radius1)[0])
        inner_indices2 = set(self.indices[act_type].query_ball_point(location2, inner_radius2)[0])

        overlapping_indices = list(outer_intersection - (inner_indices1.union(inner_indices2)))
        if not overlapping_indices:
            return None

        if max_number_of_candidates and len(overlapping_indices) > max_number_of_candidates:
            overlapping_indices = np.random.choice(overlapping_indices, max_number_of_candidates, replace=False)

        data = self.data[act_type]
        overlapping_indices = np.array(overlapping_indices)
        candidate_identifiers = data["identifiers"][overlapping_indices]
        candidate_coordinates = data["coordinates"][overlapping_indices]
        candidate_potentials = data["potentials"][overlapping_indices]

        return candidate_identifiers, candidate_coordinates, candidate_potentials

    def find_ring_candidates(self, act_type: str, center: np.ndarray, radius1: float, radius2: float, max_iterations=20,
                             min_candidates=10, restrict_angle=False, direction_point=None,
                             angle_range=math.pi / 1.5) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find candidates within a ring around a center point.
        Iteratively increase the radii until a sufficient number of candidates is found."""
        i = 0
        if logger.isEnabledFor(logging.DEBUG): logger.debug(
            f"Finding candidates for type {act_type} within a ring around {center} with radii {radius1} and {radius2}.")
        while True:
            candidates = self.query_within_ring(act_type, center, radius1, radius2)
            if candidates is not None:
                if len(candidates[0]) >= min_candidates:
                    if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Found {len(candidates[0])} candidates.")
                    stats_tracker.log(f"Find_ring_candidates: Iterations for {act_type}", i)
                    return candidates
            radius1, radius2 = h.spread_distances(radius1, radius2, iteration=i, first_step=20)
            i += 1
            if logger.isEnabledFor(logging.DEBUG): logger.debug(
                f"Iteration {i}. Increasing radii to {radius1} and {radius2}.")
            if i > max_iterations:
                raise ValueError(f"Not enough candidates found after {max_iterations} iterations.")

    def find_overlapping_rings_candidates(self, act_type: str, location1: np.ndarray, location2: np.ndarray,
                                          radius1a: float, radius1b: float, radius2a: float, radius2b: float,
                                          min_candidates=1, max_candidates=None, max_iterations=15):
        """Find candidates within two overlapping rings (donuts) around two center points.
        Iteratively increase the radii until a sufficient number of candidates is found.
        """

        i = 0
        while True:
            candidates = self.query_within_two_overlapping_rings(
                act_type, location1, location2, radius1a, radius1b, radius2a, radius2b, max_candidates)
            if candidates is not None and len(candidates[0]) >= min_candidates:
                if logger.isEnabledFor(logging.DEBUG): logger.debug(
                    f"Found {len(candidates[0])} candidates after {i} iterations.")
                stats_tracker.log(f"Find_ring_candidates: Iterations for {act_type}", i)
                return candidates, i
            radius1a, radius1b = h.spread_distances(radius1a, radius1b, iteration=i, first_step=50)
            radius2a, radius2b = h.spread_distances(radius2a, radius2b, iteration=i, first_step=50)
            i += 1
            if logger.isEnabledFor(logging.DEBUG): logger.debug(
                f"Iteration {i}. Increasing radii to {radius1a}, {radius1b} and {radius2a}, {radius2b}.")
            if i > max_iterations:
                raise RuntimeError(f"Not enough candidates found after {max_iterations} iterations.")


class EvaluationFunction:

    @staticmethod
    def evaluate_candidates(potentials: np.ndarray = None, dist_deviations: np.ndarray = None,
                            number_of_candidates: int = None) -> np.ndarray:
        """
        Scoring function collection for the candidates based on potentials and distances.

        :param potentials: Numpy array of potentials for the returned locations.
        :param dist_deviations: Distance deviations from the target (if available).
        :param number_of_candidates:
        :return: Non-normalized, absolute scores.
        """
        # if distances is not None and potentials is not None:
        #     return potentials / distances
        # if potentials is not None:
        #     return potentials
        if dist_deviations is not None:
            return np.maximum(0, 1000000 - dist_deviations)

            # return 1 / distances

            # best_index = np.argmin(distances)
            # scores = np.zeros_like(distances)
            # scores[best_index] = 1
            # return scores
        else:
            if number_of_candidates is None:
                return np.full((len(potentials),), 1000000)
            return np.full((number_of_candidates,), 1000000)

    @classmethod
    def select_candidate_indices(
            cls,
            scores: np.ndarray,
            num_candidates: int,
            strategy: str = 'monte_carlo',
            top_portion: float = 0.5,
            coords: np.ndarray = None,
            num_cells_x: int = 20,
            num_cells_y: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the indices of candidates based on their normalized scores using Monte Carlo sampling,
        a top-n strategy, a mixed strategy, or spatial downsampling.

        :param scores: A 1D numpy array of scores corresponding to candidates.
        :param num_candidates: The number of candidates to select.
        :param strategy: Selection strategy ('monte_carlo', 'top_n', 'mixed', or 'spatial_downsample').
        :param top_portion: Portion of candidates to select from the top scores when using the 'mixed' strategy.
        :param coords: 2D numpy array of shape (n, 2) with candidate spatial coordinates (required for spatial_downsample).
        :param num_cells_x: Number of cells along the longitude (spatial_downsample).
        :param num_cells_y: Number of cells along the latitude (spatial_downsample).
        :return: A tuple containing:
                 - The selected indices of the best candidates.
                 - A 1D array of the scores corresponding to the selected indices.
        """
        assert len(scores) > 0, "The scores array cannot be empty."
        if num_candidates >= len(scores):
            stats_tracker.increment("Select_candidates_indices: All candidates selected")
            return np.arange(len(scores)), scores

        if strategy == 'monte_carlo':
            stats_tracker.increment("Scoring runs (Monte Carlo)")
            normalized_scores = scores / np.sum(scores, dtype=np.float64)
            chosen_indices = np.random.choice(len(scores), num_candidates, p=normalized_scores, replace=False)

        elif strategy == 'top_n':
            stats_tracker.increment("Scoring runs (Top N)")
            chosen_indices = np.argsort(scores)[-num_candidates:][::-1]  # Top scores in descending order

        elif strategy == 'mixed':
            stats_tracker.increment("Scoring runs (Mixed)")
            num_top = int(np.ceil(num_candidates * top_portion))
            num_monte_carlo = num_candidates - num_top

            sorted_indices = np.argsort(scores)[-num_top:][::-1]
            remaining_indices = np.setdiff1d(np.arange(len(scores)), sorted_indices)

            if len(remaining_indices) > 0 and num_monte_carlo > 0:
                remaining_scores = scores[remaining_indices]
                normalized_remaining_scores = remaining_scores / np.sum(remaining_scores, dtype=np.float64)
                monte_carlo_indices = np.random.choice(remaining_indices, num_monte_carlo,
                                                       p=normalized_remaining_scores, replace=False)
                chosen_indices = np.concatenate((sorted_indices, monte_carlo_indices))
            else:
                chosen_indices = sorted_indices

        elif strategy == 'spatial_downsample':
            assert coords is not None, "Coordinates (coords) are required for spatial_downsample strategy."
            stats_tracker.increment("Scoring runs (Spatial Downsample)")
            chosen_indices = cls.even_spatial_downsample(
                coords, num_cells_x=num_cells_x, num_cells_y=num_cells_y
            )[:num_candidates]

        elif strategy == 'top_n_spatial_downsample':
            assert coords is not None, "Coordinates (coords) are required for top_n_spatial_downsample strategy."
            stats_tracker.increment("Scoring runs (Top N Spatial Downsample)")

            # Sort scores in descending order
            sorted_indices = np.argsort(scores)[::-1]
            sorted_scores = scores[sorted_indices]

            # Identify the cutoff score
            cutoff_score = sorted_scores[num_candidates - 1] if len(sorted_scores) >= num_candidates else sorted_scores[
                -1]

            # Find all indices with scores >= cutoff_score (this may be more than num_candidates if scores are equal)
            top_indices = np.where(scores >= cutoff_score)[0]

            # Check if spatial downsampling is needed
            if len(top_indices) > num_candidates:
                num_cells = max(1, int(np.sqrt(num_candidates)) + 1)  # Slightly above the square root of candidates
                chosen_indices = cls.even_spatial_downsample(
                    coords, num_cells_x=num_cells, num_cells_y=num_cells
                )
            else:
                # Use the sorted indices if no downsampling is needed
                chosen_indices = sorted_indices[:num_candidates]

        else:
            raise ValueError(
                "Invalid selection strategy. Use 'monte_carlo', 'top_n', 'mixed', or 'spatial_downsample'.")

        chosen_scores = scores[chosen_indices]
        return chosen_indices, chosen_scores

    @classmethod
    def select_candidates(
            cls,
            candidates: Tuple[np.ndarray, ...],
            scores: np.ndarray,
            num_candidates: int,
            strategy: str = 'monte_carlo',
            top_portion: float = 0.5,
            coords: np.ndarray = None,
            num_cells_x: int = 20,
            num_cells_y: int = 20
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """
        Selects a specified number of candidates based on their scores using various strategies.

        :param candidates: A tuple of arrays with the candidates.
        :param scores: A 1D array of scores corresponding to the candidates.
        :param num_candidates: The number of candidates to select.
        :param strategy: Selection strategy ('monte_carlo', 'top_n', 'mixed', or 'spatial_downsample').
        :param top_portion: Portion of candidates to select from the top scores when using the 'mixed' strategy.
        :param coords: 2D numpy array of candidate spatial coordinates (required for 'spatial_downsample'). If no
                        coordinates are provided, candidates[1] is used as coordinates.
        :param num_cells_x: Number of cells along the longitude (spatial_downsample).
        :param num_cells_y: Number of cells along the latitude (spatial_downsample).
        :return: A tuple containing:
            - A tuple of arrays with the selected candidates.
            - A 1D array of the scores corresponding to the selected candidates.
        """
        assert len(candidates[0]) == len(scores), "The number of candidates and scores must match."
        if strategy == 'keep_all':
            return candidates, scores
        if (strategy == 'spatial_downsample' or strategy == "top_n_spatial_downsample") and coords is None:
            coords = candidates[1]

        chosen_indices, chosen_scores = cls.select_candidate_indices(
            scores, num_candidates, strategy, top_portion, coords, num_cells_x, num_cells_y
        )

        selected_candidates = tuple(
            np.atleast_1d(arr[chosen_indices].squeeze()) if arr is not None else None for arr in candidates
        )

        if num_candidates == 1:
            return (
                tuple(
                    (
                        np.atleast_1d(selected_candidates[0]),  # IDs (n,)
                        np.atleast_2d(selected_candidates[1]),  # Coordinates (n, 2)
                        np.atleast_1d(selected_candidates[2]),  # Potentials (n,)
                    )
                ),
                np.atleast_1d(chosen_scores)  # Scores (n,)
            )

        return selected_candidates, chosen_scores

    @staticmethod
    def even_spatial_downsample(coords, num_cells_x=20, num_cells_y=20):
        """
        Downsample points and return indices of the kept points.

        Parameters:
        - coords: 2D coordinates array (n, 2)
        - num_cells_x: Number of cells along the longitude.
        - num_cells_y: Number of cells along the latitude.

        Returns:
        - A list of indices of the points that are kept after downsampling.
        """
        lats = coords[:, 0]
        lons = coords[:, 1]

        min_lat, max_lat = lats.min(), lats.max()
        min_lon, max_lon = lons.min(), lons.max()

        lat_range = max_lat - min_lat or 1e-9
        lon_range = max_lon - min_lon or 1e-9

        lat_step = lat_range / max(num_cells_y, 1)
        lon_step = lon_range / max(num_cells_x, 1)

        total_cells = num_cells_x * num_cells_y
        filled_cells = set()
        kept_indices = []

        for i in range(len(coords)):
            lat, lon = lats[i], lons[i]
            cell_x = min(int((lon - min_lon) / lon_step), num_cells_x - 1)
            cell_y = min(int((lat - min_lat) / lat_step), num_cells_y - 1)
            cell_id = cell_y * num_cells_x + cell_x

            if cell_id not in filled_cells:
                kept_indices.append(i)
                filled_cells.add(cell_id)

            # Stop early if all cells are filled
            if len(filled_cells) == total_cells:
                break

        return kept_indices


class CircleIntersection:
    def __init__(self, target_locations: TargetLocations):
        self.target_locations = target_locations

    def find_circle_intersection_candidates(self, start_coord: np.ndarray, end_coord: np.ndarray, type: str,
                                            distance_start_to_act: float, distance_act_to_end: float,
                                            num_candidates: int, only_return_valid=False):
        """
        Find n location candidates for a given activity type between two known locations.
        Returns two sets of candidates if two intersection points are found, otherwise only one set.
        """
        # Find the intersection points
        intersect1, intersect2 = self.find_circle_intersections(
            start_coord, distance_start_to_act,
            end_coord, distance_act_to_end, only_return_valid
        )
        if intersect1 is None and intersect2 is None:
            if only_return_valid:
                return None, None, None
            raise RuntimeError("Reached impossible state.")

        # Handle both intersections in a single batch query if both points exist
        if intersect2 is not None:
            # Stack intersection points into a single array
            locations_to_query = np.array([intersect1, intersect2])

            # Perform a batched query for both intersection points
            candidate_identifiers, candidate_coordinates, candidate_potentials = self.target_locations.query_closest(
                type, locations_to_query, num_candidates
            )

            if num_candidates == 1:
                return candidate_identifiers, candidate_coordinates, candidate_potentials

            # Concatenate the results from the batch query
            combined_identifiers = np.concatenate(candidate_identifiers, axis=0)
            combined_coordinates = np.concatenate(candidate_coordinates, axis=0)
            combined_potentials = np.concatenate(candidate_potentials, axis=0)
        else:
            # Query only the first intersection point
            combined_identifiers, combined_coordinates, combined_potentials = self.target_locations.query_closest(
                type, intersect1, num_candidates
            )
            if num_candidates == 1:
                return np.atleast_1d(combined_identifiers), np.atleast_2d(combined_coordinates), np.atleast_1d(
                    combined_potentials)

        return combined_identifiers, combined_coordinates, combined_potentials

    def find_circle_intersections(self, center1: np.ndarray, radius1: float, center2: np.ndarray, radius2: float,
                                  only_return_valid=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the intersection points of two circles.

        :param center1: The center of the first circle (e.g., np.array([x1, y1])).
        :param radius1: The radius of the first circle.
        :param center2: The center of the second circle (e.g., np.array([x2, y2])).
        :param radius2: The radius of the second circle.
        :param only_return_valid: If True, only return valid intersection points, else None
        :return: A tuple containing one or two intersection points (each as a np.ndarray).
        """

        x1, y1 = center1
        x2, y2 = center2
        r1 = radius1
        r2 = radius2

        # Calculate the distance between the two centers
        d = h.euclidean_distance(center1, center2)

        if logger.isEnabledFor(logging.DEBUG): logger.debug(
            f"Center 1: {center1}, Radius 1: {radius1}, Center 2: {center2}, Radius 2: {radius2}")

        # Handle non-intersection conditions:
        if d < 1e-4:
            raise RuntimeError("The case of identical start and end should be handled by the donut-function.")
            # if abs(r1 - r2) < 1e-4:
            #     if logger.isEnabledFor(logging.DEBUG): logger.debug("Infinite intersections: The start and end points and radii are identical.")
            #     if logger.isEnabledFor(logging.DEBUG): logger.debug("Choosing a point on the perimeter of the circles.")
            #     intersect = np.array([x1 + r1, y1])
            #     return intersect, None
            # else:
            #     if logger.isEnabledFor(logging.DEBUG): logger.debug("No intersection: The circles are identical but have different radii.")
            #     if logger.isEnabledFor(logging.DEBUG): logger.debug("Choosing a point on the perimeter of the circles.")
            #     intersect = np.array([x1 + r1, y1])
            #     return intersect, None

        if d > (r1 + r2):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "No direct intersection: The circles are too far apart.")
            if only_return_valid:
                return None, None

            proportional_distance = r1 / (r1 + r2)
            point_on_line = center1 + proportional_distance * (center2 - center1)

            return point_on_line, None

        # One circle inside the other with no intersection
        if d < abs(r1 - r2):
            if logger.isEnabledFor(logging.DEBUG): logger.debug(
                "No intersection: One circle is fully inside the other.")
            if only_return_valid:
                return None, None

            # Identify which circle is larger
            if r1 > r2:
                larger_center, larger_radius = center1, r1
                smaller_center, smaller_radius = center2, r2
            else:
                larger_center, larger_radius = center2, r2
                smaller_center, smaller_radius = center1, r1

            # Find a point on the line connecting the centers
            proportional_distance = (d + smaller_radius + 0.5 * (larger_radius - smaller_radius - d)) / d
            midpoint = larger_center + proportional_distance * (smaller_center - larger_center)

            if logger.isEnabledFor(logging.DEBUG): logger.debug(f"Midpoint: {midpoint}")
            return midpoint, None

        if d == (r1 + r2) or d == abs(r1 - r2):
            logger.info("Whaaat? Tangential circles: The circles touch at exactly one point.")

            a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
            hi = 0  # Tangential circles will have h = 0 as h = sqrt(r1^2 - a^2)

            x3 = x1 + a * (x2 - x1) / d
            y3 = y1 + a * (y2 - y1) / d

            intersection = np.array([x3, y3])

            return intersection, None

        # Calculate points of intersection
        a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
        hi = np.sqrt(r1 ** 2 - a ** 2)

        x3 = x1 + a * (x2 - x1) / d
        y3 = y1 + a * (y2 - y1) / d

        intersect1 = np.array([x3 + hi * (y2 - y1) / d, y3 - hi * (x2 - x1) / d])
        intersect2 = np.array([x3 - hi * (y2 - y1) / d, y3 + hi * (x2 - x1) / d])

        return intersect1, intersect2

    def get_best_circle_intersection_location(self, start_coord: np.ndarray, end_coord: np.ndarray, act_type: str,
                                              distance_start_to_act: float, distance_act_to_end: float,
                                              num_circle_intersection_candidates=None, selection_strategy='top_n',
                                              max_iterations=15, only_return_valid=False):
        """
        Place a single activity at one of the closest locations.
        :param start_coord: Coordinates of the start location.
        :param end_coord: Coordinates of the end location.
        :param act_type: Type of activity (e.g., 'work', 'shopping').
        :param distance_start_to_act: Distance from start location to activity.
        :param distance_act_to_end: Distance from activity to end location.
        :param num_circle_intersection_candidates: Number of candidates to consider.
        :param selection_strategy: Strategy for selecting the best candidate.
        :param max_iterations: Maximum number of iterations for finding candidates.
        :param only_return_valid: If True, only return feasible locations, else None.
        :return: Tuple containing the selected identifier, coordinates, potential, and score.
        """

        # If start and end locations are very close, fallback to ring search
        if h.euclidean_distance(start_coord, end_coord) < 1e-4:
            if only_return_valid and abs(distance_act_to_end - distance_start_to_act) > 10:  # 10m deviation is fine
                return None, None, None, None
            radius1, radius2 = h.spread_distances(distance_start_to_act, distance_act_to_end)
            candidate_ids, candidate_coords, candidate_potentials = self.target_locations.find_ring_candidates(
                act_type, start_coord, radius1, radius2, max_iterations=max_iterations, min_candidates=1
            )
        else:
            # Find intersection candidates between start and end
            candidate_ids, candidate_coords, candidate_potentials = self.find_circle_intersection_candidates(
                start_coord, end_coord, act_type, distance_start_to_act, distance_act_to_end,
                num_candidates=num_circle_intersection_candidates
            )
            if candidate_ids is None:
                if only_return_valid:
                    return None, None, None, None
                raise RuntimeError("Reached impossible state.")

        # Calculate distance deviations
        distance_deviations = (
                h.get_abs_distance_deviations(candidate_coords, start_coord, distance_start_to_act) +
                h.get_abs_distance_deviations(candidate_coords, end_coord, distance_act_to_end)
        )

        # Evaluate and select the best candidate
        scores = EvaluationFunction.evaluate_candidates(candidate_potentials, distance_deviations)
        best_index = EvaluationFunction.select_candidate_indices(scores, 1, selection_strategy)[0]

        # Extract the selected candidate's data
        best_id = candidate_ids[best_index][0]
        best_coord = candidate_coords[best_index][0]
        best_potential = candidate_potentials[best_index][0]
        best_score = scores[best_index][0]

        return best_id, best_coord, best_potential, best_score


class CARLA:
    """
    """

    def __init__(self, target_locations: TargetLocations, segmented_plans: SegmentedPlans, config):
        self.target_locations = target_locations
        self.segmented_plans = segmented_plans
        self.c_i = CircleIntersection(target_locations)

        self.number_of_branches = config['number_of_branches']
        self.min_candidates_complex_case = config['min_candidates_complex_case']
        self.candidates_two_leg_case = config['candidates_two_leg_case']
        self.max_candidates = config['max_candidates']
        self.anchor_strategy = config['anchor_strategy']
        self.selection_strategy_complex_case = config['selection_strategy_complex_case']
        self.selection_strategy_two_leg_case = config['selection_strategy_two_leg_case']
        self.max_radius_reduction_factor = config['max_radius_reduction_factor']
        self.max_iterations_complex_case = config['max_iterations_complex_case']
        self.only_return_valid_persons = config['only_return_valid_persons']

    def run(self):
        placed_dict = {}
        for person_id, segments in tqdm(self.segmented_plans.items(), desc="Processing persons"):
            placed_dict[person_id] = []
            for segment in segments:
                placed_segment, _ = self.solve_segment(segment)
                if placed_segment is not None:
                    placed_dict[person_id].append(placed_segment)
                elif not self.only_return_valid_persons:
                    raise RuntimeError("None should only be returned when only valid persons are requested.")
        return placed_dict

    def _get_anchor_index(self, num_legs: int) -> int:
        """Determine the anchor index based on strategy."""
        if self.anchor_strategy == "lower_middle":
            return num_legs // 2 - 1
        elif self.anchor_strategy == "upper_middle":
            return num_legs // 2
        elif self.anchor_strategy == "start":
            return 0
        elif self.anchor_strategy == "end":
            return num_legs - 1
        else:
            raise ValueError("Invalid anchor strategy.")

    def solve_segment(self, segment: Segment) -> Tuple[Segment, float]:
        """Recursively solve a segment for multiple candidates."""
        if len(segment) == 0:
            raise ValueError("No legs in segment.")
        elif len(segment) == 1:  # Base case for single leg
            assert segment[0].from_location.size > 0 and segment[0].to_location.size > 0, \
                "Start and end locations must be known."
            return segment, 0
        elif len(segment) == 2:  # Base case for two legs
            best_loc = self.c_i.get_best_circle_intersection_location(
                segment[0].from_location, segment[1].to_location, segment[0].to_act_type,
                segment[0].distance, segment[1].distance, self.candidates_two_leg_case,
                self.selection_strategy_two_leg_case, self.max_iterations_complex_case,
                self.only_return_valid_persons
            )
            if best_loc[0] is None:
                if self.only_return_valid_persons:
                    return None, 0
                raise RuntimeError("Reached impossible state.")
            updated_leg1 = segment[0]._replace(to_location=best_loc[1], to_act_identifier=best_loc[0])
            updated_leg2 = segment[1]._replace(from_location=best_loc[1])
            return (updated_leg1, updated_leg2), best_loc[3]  # act_score

        # Recursive case
        anchor_idx = self._get_anchor_index(len(segment))
        location1 = segment[0].from_location
        location2 = segment[-1].to_location
        act_type = segment[anchor_idx].to_act_type

        # Generate candidate locations
        distances = np.array([leg.distance for leg in segment])
        distances_start_to_act = distances[:anchor_idx + 1]  # Up to and including anchor
        distances_act_to_end = distances[anchor_idx + 1:]  # From anchor + 1 to end

        # Radii describing the search area (two overlapping donuts)
        min_possible_distance1, max_possible_distance1 = h.get_min_max_distance(distances_start_to_act)
        min_possible_distance2, max_possible_distance2 = h.get_min_max_distance(distances_act_to_end)

        # Limit the search space, as the maximum radii will almost never be needed in valid trips
        if self.max_radius_reduction_factor:
            min_possible_distance1 *= self.max_radius_reduction_factor
            max_possible_distance1 *= self.max_radius_reduction_factor

        candidates, iterations = self.target_locations.find_overlapping_rings_candidates(
            act_type, location1, location2,
            min_possible_distance1, max_possible_distance1,
            min_possible_distance2, max_possible_distance2,
            self.min_candidates_complex_case, self.max_candidates,
            self.max_iterations_complex_case)
        candidate_ids, candidate_coords, candidate_potentials = candidates

        # Evaluate candidates
        if iterations > 0:  # We need to find distance deviations of each candidate to score them
            candidate_deviations = np.zeros(len(candidate_ids))
            # We only count deviations of lowest-level legs to avoid double counting
            if len(distances_start_to_act) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location1,
                                                                      distances_start_to_act)
            elif len(distances_act_to_end) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location2,
                                                                      distances_act_to_end)
            local_scores = EvaluationFunction.evaluate_candidates(candidate_potentials, candidate_deviations)
        else:  # No distance deviations expected, just score by potentials
            candidate_deviations = np.zeros(len(candidate_ids))
            if len(distances_start_to_act) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location1,
                                                                      distances_start_to_act)
            if len(distances_act_to_end) == 1:
                candidate_deviations += h.get_abs_distance_deviations(candidate_coords, location2,
                                                                      distances_act_to_end)
            if np.any(candidate_deviations != 0):
                raise ValueError("Total deviations should be zero.")
            local_scores = EvaluationFunction.evaluate_candidates(candidate_potentials, None,
                                                                  len(candidate_ids))

        selected_candidates, selected_scores = EvaluationFunction.select_candidates(
            candidates, local_scores, self.number_of_branches, self.selection_strategy_complex_case
        )

        # Process each candidate and split segments
        full_segs = []
        branch_scores = []
        for i in range(len(selected_candidates[0])):
            new_coord = selected_candidates[1][i]
            new_id = selected_candidates[0][i]

            # Create updated legs (safe copies, not modifying originals)
            updated_leg1 = segment[anchor_idx]._replace(to_location=new_coord, to_act_identifier=new_id)
            updated_leg2 = segment[anchor_idx + 1]._replace(from_location=new_coord)

            # Split into subsegments with safely updated legs
            subsegment1 = (*segment[:anchor_idx], updated_leg1)
            subsegment2 = (updated_leg2, *segment[anchor_idx + 2:])

            # Recursively solve each subsegment
            located_seg1, score1 = self.solve_segment(subsegment1)
            located_seg2, score2 = self.solve_segment(subsegment2)

            if located_seg1 is None or located_seg2 is None:
                if self.only_return_valid_persons:
                    return None, 0
                raise RuntimeError("Reached impossible state.")
            # Combine results and track scores
            total_score = score1 + score2 + selected_scores[i]
            branch_scores.append(total_score)
            full_segs.append((*located_seg1, *located_seg2))

        # Return the best solution
        best_idx = np.argmax(branch_scores)
        return full_segs[best_idx], branch_scores[best_idx]


def new_segment_plans(plans: SegmentedPlans) -> SegmentedPlans:
    """
    Segment the plan of each person into segments where only the start and end locations are known.
    :param plans: SegmentedPlans (frozendict of person_id -> SegmentedPlan).
    :return: SegmentedPlans (frozendict of person_id -> SegmentedPlan).
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Segmenting legs for {len(plans)} persons.")

    segmented_result = {}

    for person_id, legs in plans.items():
        segments = []  # List to hold completed segments
        current_segment = []  # Temporary list for building a segment

        for leg in legs:
            current_segment.append(leg)
            if leg.to_location.size > 0:  # End of a segment when to_location is set
                segments.append(tuple(current_segment))  # Immutable tuple for the segment
                current_segment = []  # Start a new segment

        # Add any remaining legs as a final segment
        if current_segment:
            segments.append(tuple(current_segment))

        # Store the segmented plan as an immutable tuple
        segmented_result[person_id] = tuple(segments)

    # Return the result wrapped in a frozendict
    return frozendict(segmented_result)


def convert_to_segmented_plans(df: pd.DataFrame) -> SegmentedPlans:
    """
    Convert a DataFrame into SegmentedPlans using Leg structure.

    :param df: DataFrame containing the trip data.
    :return: SegmentedPlans (frozendict of person_id -> SegmentedPlan).
    """

    def safe_array(loc):
        """Convert location to np.array, replace None with an empty array."""
        return np.array(loc) if loc is not None else np.array([])

    def row_to_leg(leg_tuple) -> Leg:
        return Leg(
            unique_leg_id=leg_tuple[0],
            from_location=safe_array(leg_tuple[3]),
            to_location=safe_array(leg_tuple[4]),
            distance=float(leg_tuple[2]) if leg_tuple[2] is not None else 0.0,
            to_act_type=leg_tuple[1] if leg_tuple[1] is not None else "unknown",
            to_act_identifier=None
        )

    # Extract legs information into tuples
    legs_info_df = pd.DataFrame({
        s.UNIQUE_P_ID_COL: df[s.UNIQUE_P_ID_COL],
        'leg_info': list(zip(
            df[s.UNIQUE_LEG_ID_COL],
            df[s.ACT_TO_INTERNAL_COL],
            df[s.LEG_DISTANCE_METERS_COL],
            [safe_array(loc) for loc in df['from_location']],
            [safe_array(loc) for loc in df['to_location']],
        ))
    })

    # Group by unique person identifier and convert to SegmentedPlans
    grouped = legs_info_df.groupby(s.UNIQUE_P_ID_COL)['leg_info'].apply(list)
    segmented_plans = frozendict({
        person_id: tuple(
            tuple(row_to_leg(leg_tuple) for leg_tuple in segment)  # Segment as tuple of Legs
        )
        for person_id, segment in grouped.items()
    })

    return segmented_plans


def write_placement_results_dict_to_population_df(placement_results_dict, population_df, merge_how = 'left') -> pd.DataFrame:
    """Writes the placement results from the dictionary to the big DataFrame."""
    records = []
    for person_id, segments in placement_results_dict.items():
        for segment in segments:
            for leg in segment:
                records.append(leg)

    data_df = pd.DataFrame(records)

    # Check columns
    mandatory_columns = [s.UNIQUE_LEG_ID_COL, 'from_location', 'to_location']
    optional_columns = ['to_act_name', 'to_act_potential', 'to_act_identifier']

    for col in mandatory_columns:
        if col not in data_df.columns:
            raise ValueError(f"Mandatory column '{col}' is missing in data_df.")

    existing_optional_columns = [col for col in optional_columns if col in data_df.columns]
    existing_columns = mandatory_columns + existing_optional_columns

    # Perform the merge with the existing columns
    merged_df = population_df.merge(data_df[existing_columns], on=s.UNIQUE_LEG_ID_COL, how=merge_how)

    # Combine columns to prioritize non-NaN values from data_df (_x is the original column, _y is the new one)
    # From and to location are always expected to be present (even before placement, there will be home locations)
    merged_df['from_location'] = merged_df['from_location_y'].combine_first(merged_df['from_location_x'])
    merged_df['to_location'] = merged_df['to_location_y'].combine_first(merged_df['to_location_x'])
    merged_df = merged_df.drop(columns=['from_location_x', 'from_location_y', 'to_location_x', 'to_location_y'])

    try:
        merged_df['to_act_identifier'] = merged_df['to_act_identifier_y'].combine_first(
            merged_df['to_act_identifier_x'])
        merged_df = merged_df.drop(columns=['to_act_identifier_x', 'to_act_identifier_y'])
    except KeyError:
        pass
    try:
        merged_df['to_act_name'] = merged_df['to_act_name_y'].combine_first(merged_df['to_act_name_x'])
        merged_df = merged_df.drop(columns=['to_act_name_x', 'to_act_name_y'])
    except KeyError:
        pass
    try:
        merged_df['to_act_potential'] = merged_df['to_act_potential_y'].combine_first(merged_df['to_act_potential_x'])
        merged_df = merged_df.drop(columns=['to_act_potential_x', 'to_act_potential_y'])
    except KeyError:
        pass

    # Make sure no merge postfixes are left
    assert not any([col.endswith('_x') or col.endswith('_y') for col in merged_df.columns]), "Postfixes left."
    return merged_df
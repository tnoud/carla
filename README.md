# CARLA: Location-Assignment Algorithm

## Overview

CARLA (Constrained Anchor-based Recursive Location Assignment) is an algorithm for assigning activity locations in activity-based transport models. It balances **location potential** and **distance accuracy**, while handling real-world challenges like sparse data or infeasible distances. The implementation works directly with the provided sample data, but the modular design allows for easy adjustments.

## Running CARLA

To run the algorithm, execute the `run_location_assignment.py` file. The script processes activity locations using the CARLA algorithm and saves the results in CSV format.

## Inputs

CARLA requires two main inputs:

1. **Population Data**:
   - A dataset describing individuals and their trips (currently read as a pandas dataframe, but then reformatted into a much more efficient frozen dict).
   - Must include columns for:
     - **Unique Person ID (`s.UNIQUE_P_ID_COL`)**: Identifies each person.
     - **Unique Trip ID (`s.UNIQUE_LEG_ID_COL`)**: Identifies each trip/leg.
     - **From Location (`from_location`)**: The start coordinates of each trip.
     - **To Location (`to_location`)**: The end coordinates of each trip (can be empty if unknown).
     - **Trip Distance (`s.LEG_DISTANCE_METERS_COL`)**: The expected trip length in meters.
     - **Activity Type (`s.ACT_TO_INTERNAL_COL`)**: The type of activity associated with the trip destination.

2. **Location Data**:
   - A file containing potential locations for each activity type (currently read as a JSON file).
   - Each entry must include:
     - **Activity Type**: The type of activities the location supports (e.g., "work", "shopping").
     - **Location Coordinates**: The latitude and longitude of the location.
     - **Location Potential**: A measure of the location's attractiveness or capacity. Just set all to 0 if this is unknown.

These inputs allow CARLA to match individuals' trips to suitable activity locations while adhering to distance constraints and optimizing for location potential.

## Configurations

- **`number_of_branches`**: Number of location candidates (branches) to evaluate for each trip chain.
- **`min_candidates_complex_case`**: Minimum number of candidates to consider for the complex case (recommended: 10–20).
- **`candidates_two_leg_case`**: Number of candidates for the two-leg case (recommended: ~20, possibly more)
- **`max_candidates`**: Maximum number of candidates to evaluate (set to `None` for no limit, recommended: `None`).
- **`anchor_strategy`**: Determines the anchor point for splitting trip chains. Options include:
  - `lower_middle` (Recommended)
  - `upper_middle`
  - `start`
  - `end`
- **`selection_strategy_complex_case`**: Strategy for selecting candidates in the complex case. Options include:
  - `top_n`
  - `monte_carlo`
  - `spatial_downsample`
  - `top_n_spatial_downsample` (Usually the best)
  - `mixed`
- **`selection_strategy_two_leg_case`**: Candidate selection strategy for two-leg case (same options as above, should be set to `top_n`.).
- **`max_iterations_complex_case`**: Maximum iterations for finding candidates in complex cases before erroring.

## Outputs

The script generates the following files:
- **`location_assignment_result_carla.csv`**: Contains the assigned locations for each activity.
- **`location_assignment_stats.txt`**: Includes performance metrics and runtime statistics.
---

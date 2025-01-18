"""
Load settings from settings.yaml file and define constants for use in the pipelines.
"""

import yaml
import utils.pipeline_setup as mps
from utils.logger import logging
logger = logging.getLogger(__name__)

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        logger.info(f"Loaded config from {file_path}")
    return config

settings = load_yaml_config(mps.PROJECT_ROOT + '/settings.yaml')

# Files
INPUT_FILES = settings['input_files']
SHAPE_BOUNDARY_FILE = INPUT_FILES['shape_boundary_file']
REGION_WITHOUT_CITY_GPKG_FILE = INPUT_FILES['region_without_city_gpkg_file']
EXPANDED_HOUSEHOLDS_FILES: list = INPUT_FILES['expanded_households_files']
MiD_HH_FOLDER = INPUT_FILES['mid_hh_folder']
MiD_PERSONS_FOLDER = INPUT_FILES['mid_persons_folder']
MiD_TRIPS_FOLDER = INPUT_FILES['mid_trips_folder']
BUILDINGS_IN_LOWEST_GEOGRAPHY_WITH_WEIGHTS_FILE = INPUT_FILES['buildings_in_lowest_geography_with_weights_file']
ENHANCED_MID_FOLDER = INPUT_FILES['enhanced_mid_folder']

CAPA_CELLS_CSV_PATH = INPUT_FILES['capa_cells_csv_path']
CAPA_CELLS_SHP_PATH = INPUT_FILES['capa_cells_shp_path']

SLACK_FACTORS_FILE = INPUT_FILES['slack_factors_file']

OUTPUT_FILES = settings['output_files']
POPULATION_ANALYSIS_OUTPUT_FILE = OUTPUT_FILES['population_analysis_output_file']
STATS_FILE = OUTPUT_FILES['stats_file']
ENHANCED_MID_FILE = OUTPUT_FILES['enhanced_mid_file']

# Columns
ID_COLUMNS: dict = settings['id_columns']
HH_COLUMNS: dict = settings['hh_columns']
P_COLUMNS: dict = settings['person_columns']
L_COLUMNS: dict = settings['leg_columns']
GEO_COLUMNS: dict = settings['geography_columns']

# Household-related columns
HOUSEHOLD_MID_ID_COL = ID_COLUMNS['household_mid_id_column']
HOUSEHOLD_POPSIM_ID_COL = ID_COLUMNS['household_popsim_id_column']
H_CAR_IN_HH_COL = HH_COLUMNS['car_in_hh_column']
H_REGION_TYPE_COL = HH_COLUMNS['region_type_column']
H_NUMBER_OF_CARS_COL = HH_COLUMNS['number_of_cars_column']

# Person-related columns
PERSON_ID_COL = ID_COLUMNS['person_id_column']
PERSON_AGE_COL = P_COLUMNS['person_age']
CAR_AVAIL_COL = P_COLUMNS['car_avail']
HAS_LICENSE_COL = P_COLUMNS['has_license']
NUMBER_OF_LEGS_COL = P_COLUMNS['number_of_legs']

# Leg-related columns
LEG_ID_COL = ID_COLUMNS['leg_id_column']
LEG_NON_UNIQUE_ID_COL = ID_COLUMNS['leg_non_unique_id_column']
LEG_START_TIME_COL = L_COLUMNS['leg_start_time']
LEG_END_TIME_COL = L_COLUMNS['leg_end_time']
LEG_DURATION_MINUTES_COL = L_COLUMNS['leg_duration_minutes']
LEG_DURATION_SECONDS_COL = 'leg_duration_seconds'
LEG_DISTANCE_KM_COL = L_COLUMNS['leg_distance_km']
LEG_DISTANCE_METERS_COL = 'leg_distance_meters'
FIRST_LEG_STARTS_AT_HOME_COL = L_COLUMNS['first_leg_starts_at_home']
LEG_IS_RBW_COL = L_COLUMNS['leg_is_rbw']

# Geography-related columns
TT_MATRIX_CELL_ID_COL = ID_COLUMNS['tt_matrix_cell_id_column']

# Value maps
VALUE_MAPS = settings['value_maps']

MODE_CAR = VALUE_MAPS['modes']['car']['internal']
MODE_PT = VALUE_MAPS['modes']['pt']['internal']
MODE_RIDE = VALUE_MAPS['modes']['ride']['internal']
MODE_BIKE = VALUE_MAPS['modes']['bike']['internal']
MODE_WALK = VALUE_MAPS['modes']['walk']['internal']
MODE_UNDEFINED = VALUE_MAPS['modes']['undefined']['internal']

ACT_WORK = VALUE_MAPS['activities']['work']['internal']
ACT_BUSINESS = VALUE_MAPS['activities']['business']['internal']
ACT_EDUCATION = VALUE_MAPS['activities']['education']['internal']
ACT_SHOPPING = VALUE_MAPS['activities']['shopping']['internal']
ACT_ERRANDS = VALUE_MAPS['activities']['errands']['internal']
ACT_PICK_UP_DROP_OFF = VALUE_MAPS['activities']['pick_up_drop_off']['internal']
ACT_LEISURE = VALUE_MAPS['activities']['leisure']['internal']
ACT_HOME = VALUE_MAPS['activities']['home']['internal']
ACT_RETURN_JOURNEY = VALUE_MAPS['activities']['return_journey']['internal']
ACT_OTHER = VALUE_MAPS['activities']['other']['internal']
ACT_EARLY_EDUCATION = VALUE_MAPS['activities']['early_education']['internal']
ACT_DAYCARE = VALUE_MAPS['activities']['daycare']['internal']
ACT_ACCOMPANY_ADULT = VALUE_MAPS['activities']['accompany_adult']['internal']
ACT_SPORTS = VALUE_MAPS['activities']['sports']['internal']
ACT_MEETUP = VALUE_MAPS['activities']['meetup']['internal']
ACT_LESSONS = VALUE_MAPS['activities']['lessons']['internal']
ACT_UNSPECIFIED = VALUE_MAPS['activities']['unspecified']['internal']

MODE_INTERNAL_COL = "mode_internal"
MODE_MID_COL = L_COLUMNS['leg_main_mode']
MODE_MATSIM_COL = "mode_matsim"

ACT_TO_INTERNAL_COL = "activity_to_internal"
ACT_FROM_INTERNAL_COL = "activity_from_internal"
ACT_MID_COL = L_COLUMNS['leg_target_activity']
ACT_MATSIM_COL = "activity_matsim"

CAR_NEVER = VALUE_MAPS['car_availability']['never']

CAR_IN_HH_NO = VALUE_MAPS['car_in_hh']['no']
CAR_IN_HH_YES = VALUE_MAPS['car_in_hh']['yes']

LICENSE_YES = VALUE_MAPS['license']['yes']
LICENSE_NO = VALUE_MAPS['license']['no']
LICENSE_UNKNOWN = VALUE_MAPS['license']['unknown']
ADULT_OVER_16_PROXY = VALUE_MAPS['license']['adult_over_16_proxy']
PERSON_UNDER_16 = VALUE_MAPS['license']['person_under_16']

FIRST_LEG_STARTS_AT_HOME = VALUE_MAPS['misc']['first_leg_starts_at_home']

# Misc
LOWEST_LEVEL_GEOGRAPHY = settings['lowest_level_geography']
PLAY_FAILURE_ALERT: bool = settings['misc']['dun_dun_duuun']  # Play sound on error (useful for long runs)
BASE_DATE = "2020-01-01"  # Arbitrary date for converting times to datetime objects

SAMPLE_SIZE = settings['sample_size']

N_CLOSEST_CELLS = settings['n_closest_cells']
DEFAULT_SLACK_FACTOR = settings['default_slack_factor']

SIGMOID_BETA = settings['sigmoid_beta']
SIGMOID_DELTA_T = settings['sigmoid_delta_t']

# Column names that are set at runtime
PROCESSING_COLUMNS = settings['processing_columns']

UNIQUE_LEG_ID_COL = PROCESSING_COLUMNS['unique_leg_id']
UNIQUE_HH_ID_COL = PROCESSING_COLUMNS['unique_household_id']
UNIQUE_P_ID_COL = PROCESSING_COLUMNS['unique_person_id']

# These names aren't exposed in the yaml because they'll probably never change and just clutter the file
FACILITY_ID_COL = "facility_id"
FACILITY_X_COL = "facility_x"
FACILITY_Y_COL = "facility_y"
FACILITY_ACTIVITIES_COL = "facility_activities"

MIRRORS_MAIN_ACTIVITY_COL = "mirrors_main_activity"

HH_HAS_CONNECTIONS_COL = "hh_has_connections"
P_HAS_CONNECTIONS_COL = "p_has_connections"
NUM_CONNECTED_LEGS_COL = "num_connected_legs"

CELL_FROM_COL = "cell_from"
CELL_TO_COL = "cell_to"

COORD_FROM_COL = "coord_from"
COORD_TO_COL = "coord_to"

HOME_CELL_COL = "home_cell"
HOME_LOC_COL = "home_location"

# MODE_TRANSLATED_COL = "mode_translated_string"

# Columns added by the MiD enhancer
RANDOM_LOCATION_COL = 'random_location'
ACT_DUR_SECONDS_COL = 'activity_duration_seconds'
NUMBER_OF_LEGS_INCL_IMPUTED_COL = 'number_of_legs_incl_imputed'
IS_IMPUTED_TIME_COL = 'imputed_time'
IS_IMPUTED_LEG_COL = 'imputed_leg'
LIST_OF_CARS_COL = 'list_of_cars'

TO_ACTIVITY_WITH_CONNECTED_COL = "to_activity_with_connected"  # Leg_to_activity with activity overwritten by connected_legs

IS_PROTAGONIST_COL = "is_protagonist"
IS_MAIN_ACTIVITY_COL = "is_main_activity"

CONNECTED_LEGS_COL = "connected_legs"

HOME_TO_MAIN_METERS_COL = "home_to_main_meters"  # Same distance type as leg distances
HOME_TO_MAIN_SECONDS_COL = "home_to_main_seconds"  # Same time type as leg times (here:minutes)
HOME_TO_MAIN_TIME_IS_ESTIMATED_COL = "home_to_main_time_is_estimated"
HOME_TO_MAIN_DIST_IS_ESTIMATED_COL = "home_to_main_distance_is_estimated"

MAIN_MODE_TO_MAIN_ACT_TIMEBASED_COL = "main_mode_to_main_act_timebased"
MAIN_MODE_TO_MAIN_ACT_DISTBASED_COL = "main_mode_to_main_act_distbased"

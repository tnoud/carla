import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)))  # Assuming pipeline_setup.py is one level down from the project root

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', current_time)




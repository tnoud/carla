import logging.config
import os
from utils.pipeline_setup import OUTPUT_DIR

# output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'pipeline.log')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class MultiLineFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        return message.replace('\n', '\n' + ' ' * 20)


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'multiline': {
            '()': MultiLineFormatter,
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'multiline',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': LOG_FILE_PATH,
            'formatter': 'multiline',
        },
    },
    'loggers': {
        '': {
            'handlers':['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}

# Set up the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

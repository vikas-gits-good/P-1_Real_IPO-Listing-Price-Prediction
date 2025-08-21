import logging
import os
from datetime import datetime

# Define Log File Names
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGFILE_ETL = f"{timestamp}_etl.log"

# Define Log Folder paths
log_fold_etl = os.path.join(os.getcwd(), "logs", "etl")

# Create Log Folders if they don't exist
os.makedirs(log_fold_etl, exist_ok=True)

# Full log file paths
log_path_etl = os.path.join(log_fold_etl, LOGFILE_ETL)


# Custom formatter to format the line number
class CustomFormatter(logging.Formatter):
    def format(self, record):
        digits = 4
        record.lineno = f"{record.lineno:0{digits}}"  # Format line number to 4 digits
        return super().format(record)


# Set up logging configuration
logging.basicConfig(
    filename=log_path_etl,
    format="[%(asctime)s] %(lineno)s %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Apply the custom formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(CustomFormatter(handler.formatter._fmt))

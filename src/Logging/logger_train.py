import logging
import os
from datetime import datetime

# Define Log File Names
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGFILE_TRAIN = f"{timestamp}_train.log"

# Define Log Folder paths
log_fold_train = os.path.join(os.getcwd(), "logs", "train")

# Create Log Folders if they don't exist
os.makedirs(log_fold_train, exist_ok=True)

# Full log file paths
log_path_train = os.path.join(log_fold_train, LOGFILE_TRAIN)


# Custom formatter to format the line number
class CustomFormatter(logging.Formatter):
    def format(self, record):
        digits = 4
        record.lineno = f"{record.lineno:0{digits}}"  # Format line number to 4 digits
        return super().format(record)


# Set up logging configuration
logging.basicConfig(
    filename=log_path_train,
    format="[%(asctime)s] %(lineno)s %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Apply the custom formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(CustomFormatter(handler.formatter._fmt))

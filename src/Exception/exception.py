import sys
import logging


# For raising error
class CustomException(Exception):
    def __init__(self, error: Exception = None):
        _, _, exc_tb = sys.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.log_msg = f"Error: File - {self.file_name} , line - [{self.lineno}], error - [{str(error)}]"

    def __str__(self):
        return self.log_msg


# For logging error
def LogException(error: Exception = None, prefix: str = None):
    _, _, exc_tb = sys.exc_info()
    lineno = exc_tb.tb_lineno
    file_name = exc_tb.tb_frame.f_code.co_filename
    log_msg = f"Error: File - {file_name} , line - [{lineno}], error - [{str(error)}]"
    log_msg = f"{prefix}: {log_msg}" if prefix else log_msg
    logging.info(log_msg)

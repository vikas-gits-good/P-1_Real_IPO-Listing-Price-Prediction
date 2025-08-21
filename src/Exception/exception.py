import sys
from Network_Security.Logging.logger_train import logger_train


class NetworkSecurityException(Exception):
    def __init__(self, error_msg, err_details: sys = sys):
        self.error_msg = error_msg
        _, _, exc_tb = err_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error: File - [{self.file_name}], line - [{self.lineno}], error - [{str(self.error_msg)}]"


if __name__ == "__main__":
    try:
        logger_train.info("Testing Exception")
        a = 1 / 0
        print("This should not print")
    except Exception as e:
        logger_train.info("Error executing Exception file")
        raise NetworkSecurityException(e)

import sys


class CustomException(Exception):
    def __init__(self, error_msg, err_details: sys = sys):
        self.error_msg = error_msg
        _, _, exc_tb = err_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error: File - [{self.file_name}], line - [{self.lineno}], error - [{str(self.error_msg)}]"

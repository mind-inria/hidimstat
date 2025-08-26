class InternalError(BaseException):
    """
    Create an error for internal error of the library

    Parameters
    ----------
    message: str
        Message of explanation of the error
    """

    def __init__(self, message):
        self.message = message

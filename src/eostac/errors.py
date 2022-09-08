class DownloadError(Exception):
    """Raised when an image can not be downloaded
    """
    def __init__(
        self,
        error,
        message='The image could not be downloaded.'
    ):
        self.error = error
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.error}: {self.message}"


class CorruptedError(Exception):
    """Raised when an image can not be opened
    """
    def __init__(
        self,
        error,
        message='The image could not be opened.'
    ):
        self.error = error
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.error}: {self.message}"

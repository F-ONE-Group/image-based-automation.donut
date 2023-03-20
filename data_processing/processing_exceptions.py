class ColumnsNotPresentException(Exception):
    def __init__(self, message):
        super().__init__(message)


class CoordsAndTextsNotCompliantException(Exception):
    def __init__(self, message):
        super().__init__(message)


class MergeException(Exception):
    def __init__(self):
        super().__init__(message)

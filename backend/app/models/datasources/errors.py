

class InvalidDataPackageZipFileError(ValueError):
    pass


class InvalidDataPackageFileNameError(ValueError):
    pass

class DataPackageIdNotFoundError(FileNotFoundError):
    pass

class DataPackageZipNotFoundError(FileNotFoundError):
    pass

class MultipleDataPackageZipFilesError(ValueError):
    pass

class FileEntryNotFoundError(FileNotFoundError):
    pass
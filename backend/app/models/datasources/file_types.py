from enum import Enum
import io

from charset_normalizer import from_bytes
import pandas as pd
import pymupdf

ARCHIVE_FILE_FORMATS = frozenset({
    ".zip",
})
TABLE_FILE_FORMATS = frozenset({
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".odf",
    ".ods",
    ".odt",
    ".csv",
})
PDF_FILE_FORMATS = frozenset({".pdf"})
IMAGE_FILE_FORMATS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif"})


class FileType(Enum):
    ARCHIVE = "archive"
    STRUCTURED_TEXT = "structured_text"
    TABLE = "table"
    PDF = "pdf"
    IMAGE = "image"
    DEFAULT = "plain_text"


def normalize_file_extension(file_extension: str) -> str:
    file_extension = file_extension.lower()
    if file_extension and not file_extension.startswith("."):
        return f".{file_extension}"
    return file_extension


def determine_file_type(file_extension: str) -> FileType:
    file_extension = normalize_file_extension(file_extension)
    if file_extension in ARCHIVE_FILE_FORMATS:
        return FileType.ARCHIVE
    elif file_extension in TABLE_FILE_FORMATS:
        return FileType.TABLE
    elif file_extension in PDF_FILE_FORMATS:
        return FileType.PDF
    elif file_extension in IMAGE_FILE_FORMATS:
        return FileType.IMAGE
    else:
        return FileType.DEFAULT


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text content from a PDF file."""
    text = ""
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()  # pyright: ignore[reportOperatorIssue]
    return text


def extract_text_from_sheets(file_name: str, content: bytes) -> str:
    """Extract text content from a spreadsheet file (Excel, CSV)."""

    suffix = normalize_file_extension(file_name.split(".")[-1])

    sheets: dict[str, pd.DataFrame]

    if suffix == ".csv":
        sheets = {file_name.split(".")[0]: pd.read_csv(io.StringIO(content.decode("utf-8", errors="ignore")))}
    elif suffix in TABLE_FILE_FORMATS and suffix != ".csv":
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
    else:
        raise ValueError(f"Unsupported file format for Excel extraction: {suffix}")

    text = ""
    idx = 1
    total_sheets = len(sheets)
    for sheet_name, df in sheets.items():
        # 1. Identify all rows that are fully numeric
        is_numeric = df.apply(lambda r: pd.to_numeric(r, errors="coerce").notnull().eq(r.notnull()).all(), axis=1)

        # 2. Check if the row above AND the row below are also numeric
        # .shift(1) looks at the previous row; .shift(-1) looks at the next row
        to_drop = is_numeric & is_numeric.shift(1) & is_numeric.shift(-1)

        # 3. Filter the DataFrame (Keep rows where 'to_drop' is False)
        df_filtered = df[~to_drop.fillna(False)]

        # 4. Convert to CSV string
        csv_string = df_filtered.to_csv(index=False)

        text += f"Sheet{idx}/{total_sheets}: {sheet_name}\n\n{csv_string}"

        if idx < total_sheets:
            text += "\n\n\n"  # Add some spacing between sheets
        idx += 1

    return text


def extract_text_from_file(
    content: bytes,
    file_name: str,
) -> str:
    """Extract text content from a file based on its type.

    Dispatches to the appropriate format-specific extractor.
    Returns raw bytes for images, text for text/table/PDF files,
    or the decoded text if the file type is not extractable.

    Args:
        content: Raw file bytes.
        file_name: str,
    """
    suffix = normalize_file_extension(file_name.split(".")[-1])
    file_type = determine_file_type(suffix)

    if file_type == FileType.TABLE:
        return extract_text_from_sheets(file_name=file_name, content=content)

    if file_type == FileType.PDF:
        return extract_text_from_pdf(content=content)

    if file_type == FileType.IMAGE:
        return "[Image content cannot be extracted as text]"

    # Default: try charset detection for text-like files
    return str(from_bytes(content).best())

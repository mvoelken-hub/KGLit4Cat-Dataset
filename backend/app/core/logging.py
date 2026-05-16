from pathlib import Path
import logging

logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

LOG_FILE = Path("app/temp/api.log")


def setup_logging(runtime_dir: Path) -> None:
    log_file = runtime_dir / "api.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup_logging() is called more than once
    if root_logger.handlers:
        return

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


logger = logging.getLogger()
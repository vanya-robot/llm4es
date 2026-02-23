"""
Скачивание MovieLens-100K.
https://files.grouplens.org/datasets/movielens/ml-100k.zip
"""
import os
import zipfile
import urllib.request

from src.utils.logging import get_logger

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_movielens_100k(data_dir: str = "data/raw") -> str:
    """Скачать и распаковать ML-100K. Возвращает путь к папке."""
    logger = get_logger()
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    extract_dir = os.path.join(data_dir, "ml-100k")

    if os.path.isdir(extract_dir) and os.path.isfile(os.path.join(extract_dir, "u.data")):
        logger.info(f"MovieLens-100K already exists at {extract_dir}, skipping download.")
        return extract_dir

    logger.info(f"Downloading MovieLens-100K from {ML100K_URL} ...")
    urllib.request.urlretrieve(ML100K_URL, zip_path)
    logger.info(f"Downloaded to {zip_path}")

    logger.info("Extracting archive ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    os.remove(zip_path)
    logger.info(f"Extracted to {extract_dir}")
    return extract_dir


if __name__ == "__main__":
    download_movielens_100k()

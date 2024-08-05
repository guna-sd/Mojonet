import os
import requests
from tqdm import tqdm

def datasets_downloader(url, download_folder='', chunk_size=8192 * 4):
    """
    Downloads a dataset from a given URL to a specified folder.

    Args:
        url (str): The URL from which to download the dataset.
        download_folder (str, optional): The folder where the dataset will be saved. Defaults to 'mojonet/datasets'.
        chunk_size (int, optional): The size of each chunk to download. Defaults to 8192 * 4.

    Returns:
        str: The path to the downloaded dataset file.
    """
    filename = url.split('/')[-1]
    folder = os.path.join(os.path.expanduser('~/.local/share/mojonet/datasets'), download_folder)
    file_path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)

    if os.path.exists(file_path):
        print(f"File '[{filename}]' already exists at {file_path}...😅")
        return file_path

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
              desc=f"Downloading: {filename}") as progress_bar:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

    print("Download completed...😉")
    return file_path


def models_downloader(url, download_folder='', chunk_size=8192 * 4):
    """
    Downloads a model from a given URL to a specified folder.

    Args:
        url (str): The URL from which to download the model.
        download_folder (str, optional): The folder where the model will be saved. Defaults to 'mojonet/models'.
        chunk_size (int, optional): The size of each chunk to download. Defaults to 8192 * 4.

    Returns:
        str: The path to the downloaded model file.
    """
    filename = url.split('/')[-1]
    folder = os.path.join(os.path.expanduser('~/.local/share/mojonet/models'), download_folder)
    file_path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)

    if os.path.exists(file_path):
        print(f"File '[{filename}]' already exists at {file_path}...😅")
        return file_path

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
              desc=f"Downloading: {filename}") as progress_bar:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

    print("Download completed...😉")
    return file_path


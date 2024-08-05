import os
import requests
from tqdm import tqdm

def datasets_downloader(url, download_folder='mojonet/datasets', chunk_size=8192 * 4):

    filename = url.split('/')[-1]
    folder = os.path.join(os.path.expanduser('~/.local/share'), download_folder)
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


def models_downloader(url, download_folder='mojonet/models', chunk_size=8192 * 4):

    filename = url.split('/')[-1]
    folder = os.path.join(os.path.expanduser('~/.local/share'), download_folder)
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


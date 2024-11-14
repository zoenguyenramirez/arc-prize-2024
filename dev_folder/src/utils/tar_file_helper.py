import subprocess
import threading
from threading import Thread
import logging
import os
from typing import Union, TypeVar, overload

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def extract_tar_file(tar_file: str, blocking: bool):
    if not is_tar_file(tar_file):
        return tar_file
    
    extracted_tar_file = tar_file[:-len('.tar.xz')]
    if file_exist(extracted_tar_file):
        return extracted_tar_file
    
    if blocking:
        logging.info(f'extracting {tar_file} in blocking mode')
        shell_extract_tar_file(tar_file, extracted_tar_file)
        return extracted_tar_file
    else:
        logging.info(f'extracting {tar_file} in NON-blocking mode')
        thread = threading.Thread(target=shell_extract_tar_file, args=(tar_file,extracted_tar_file))
        thread.start()
        return thread

def shell_extract_tar_file(tar_file: str, extracted_tar_file:str):
    subprocess.run(['tar', '-xf', tar_file], check=True)
    assert file_exist(extracted_tar_file)

def is_tar_file(file_path: str) -> bool:
    return file_path.endswith('.tar.xz')

def file_exist(file_path: str) -> bool:
    return os.path.exists(file_path)

def release_extracted_tar_file(extracted_file: str):
    tar_file = f'{extracted_file}.tar.xz'
    if not os.path.exists(tar_file):
        return
    
    if os.path.exists(extracted_file):
        try:
            os.remove(extracted_file)
            logging.info(f"Successfully deleted {extracted_file}")
        except OSError as e:
            logging.info(f"Error: {e.strerror}. Unable to delete {extracted_file}")
    else:
        logging.info(f"{extracted_file} does not exist")
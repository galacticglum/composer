'''
Downloads the Yamaha International e-Piano Competition dataset.
Source: http://www.piano-e-competition.com/.

'''

import re
import time
import click
import shutil
import argparse
import requests
from pathlib import Path

def _rmtree(path, ignore_errors=False, onerror=None, timeout=10):
    '''
    A wrapper method for 'shutil.rmtree' that waits up to the specified
    `timeout` period, in seconds.
    '''

    shutil.rmtree(path, ignore_errors, onerror)

    if path.is_dir():
        print('shutil.rmtree - Waiting for \'{}\' to be removed...'.format(path))
        # The destination path has yet to be deleted. Wait, at most, the timeout period.
        timeout_time = time.time() + timeout
        while time.time() <= timeout_time:
            if not path.is_dir():
                break

parser = argparse.ArgumentParser(description='Downloads the Yamaha International e-Piano Competition dataset.')
parser.add_argument('output', type=str, help='The output directory.')
args = parser.parse_args()

# The pages to scrape. Not all years are available on the website (e.g. 2012) so we have to manually define and maintain this. 
PAGES = ['2002', '2004', '2008', '2009', '2011', '2013', '2014', '2015', '2017', '2018']
ROOT_URL = 'http://www.piano-e-competition.com'
PAGE_URL_FORMAT = '{}/midi_{{}}.asp'.format(ROOT_URL)
DOWNLOAD_URLS = [PAGE_URL_FORMAT.format(page) for page in PAGES]

output_path = Path(args.output)
if output_path.exists():
    if not output_path.is_dir():
        print('The output path, \'{}\', is not a valid directory.'.format(args.output))
        exit(1)

    prompt = 'The output path, \'{}\', already exists! Would you like to remove it?'.format(args.output)
    if click.confirm(prompt):
        _rmtree(output_path)

output_path.mkdir(exist_ok=True, parents=True)
for url in DOWNLOAD_URLS:
    html = requests.get(url).text
    for midi in re.findall(r'[^"]+\.mid', html, re.IGNORECASE):
        midi_url_path = Path(midi)
        midi_filepath = output_path / (midi_url_path.stem + midi_url_path.suffix)
        
        with open(midi_filepath, 'wb+') as file_handle:
            midi_url = '{}{}{}'.format(ROOT_URL, '/' if not midi.startswith('/') else '', midi)
            print('Downloading {}'.format(midi_url))

            start_time = time.time()
            response = requests.get(midi_url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:
                file_handle.write(response.content)
            else:
                total_length = int(total_length)
                with click.progressbar(length=total_length) as bar:
                    for chunk in response.iter_content(chunk_size=4096):
                        file_handle.write(chunk)
                        bar.update(len(chunk))
            
import requests
import os
import gzip
import shutil
from pathlib import Path
from translate.storage.tmx import tmxfile


class OpenSubtitles_CaEn_Dataset():
    class reader():
        def __init__(self, file):
            self.__file = file

        def translations(self):
            tmx_file = tmxfile(self.__file, "ca", "en")
            for node in tmx_file.unit_iter():
                yield {"ca": node.source, "en": node.target}

    def __init__(self):
        self.url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/tmx/ca-en.tmx.gz"
        home = Path.home()
        self.folder = home.joinpath(".cache/lemon/opensubtitles/")
        self.download_filename = self.folder.joinpath("ca-en.tmx.gz")
        self.data_filename = self.folder.joinpath("data")

    def __download_if_required(self):
        # Get the data if it doesn't exist
        if not os.path.isfile(self.data_filename):
            if not os.path.isfile(self.download_filename):
                print("Downloading OpenSubtitles Dataset...")
                self.folder.mkdir(parents=True, exist_ok=True)
                with open(self.download_filename, "wb") as f:
                    r = requests.get(self.url)
                    f.write(r.content)
            # Uncompress
            with gzip.open(self.download_filename, "rb") as fin, open(self.data_filename, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            # Remove the download
            os.remove(self.download_filename)

    def __enter__(self):
        self.__download_if_required()
        self.file = open(self.data_filename, "rb")
        return self.reader(self.file)

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

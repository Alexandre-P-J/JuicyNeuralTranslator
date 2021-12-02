import requests
import os
from pathlib import Path


class TatoebaChallenge_EsEn_Dataset():
    class reader():
        def __init__(self, file):
            self.__file = file

        def translations(self):
            for line in self.__file:
                line = line.strip().split("\t")
                yield {"en": line[2], "es": line[3]} # original uses spa/eng.

    def __init__(self):
        self.url = "https://raw.githubusercontent.com/Helsinki-NLP/Tatoeba-Challenge/master/data/test/eng-spa/test.txt"
        home = Path.home()
        self.folder = home.joinpath(".cache/lemon/tatoebachallengeesen/")
        self.data_filename = self.folder.joinpath("data")

    def __download_if_required(self):
        # Get the data if it doesn't exist
        if not os.path.isfile(self.data_filename):
            print("Downloading Tatoeba Challenge Dataset...")
            self.folder.mkdir(parents=True, exist_ok=True)
            with open(self.data_filename, "wb") as f:
                r = requests.get(self.url)
                f.write(r.content)

    def __enter__(self):
        self.__download_if_required()
        self.file = open(self.data_filename, "r")
        return self.reader(self.file)

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

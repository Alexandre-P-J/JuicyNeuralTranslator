from pathlib import Path


class TatoebaTest_EnEs_Dataset():
    __folder = Path(__file__).parent.resolve()
    __filepath = __folder.joinpath("tatoeba-test-v2020-07-28.eng-spa.txt")

    class reader():
        def __init__(self, file):
            self.__file = file

        def translations(self):
            for line in self.__file.readlines():
                data = line.split("\t")
                if len(data) == 4:
                    eng = data[2]
                    spa = data[3] if data[3][-1] != "\n" else data[3][:-1]
                    yield {"en": eng, "es": spa}

    def __enter__(self):
        self.file = open(self.__filepath, "r")
        return self.reader(self.file)

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

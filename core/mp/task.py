
from multiprocessing import Queue, Process


class MPProcess(Process):
    def __int__(self, name):
        super().__init__()

        self.name = name


    def run(self):
        print(self.name)
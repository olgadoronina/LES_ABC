from params import *
from tqdm import tqdm
from time import sleep


class Parallel(object):
    def __init__(self, processes=mp.cpu_count()):
        self.pool = mp.Pool(processes=processes)
        self.results = None
        logging.info('\n' + str(processes) + " workers")

    def run(self, func, tasks):
        if PROGRESSBAR == 1:
            self.results = []
            with tqdm(total=N) as pbar:
                for i, res in tqdm(enumerate(self.pool.imap_unordered(func, tasks)), desc='ABC algorithm'):
                    self.results.append(res)
                    pbar.update()
            pbar.close()
        elif PROGRESSBAR == 2:
            self.results = self.pool.map_async(func, tasks)
            while not self.results.ready():
                print("Done ", N - self.results._number_left, '/', N)
                sleep(5)
            self.pool.close()
            self.pool.join()
        else:
            self.results = self.pool.map(func, tasks)
            self.pool.close()
        self.pool.terminate()

    def get_results(self):
        if PROGRESSBAR == 1:
            return [x for x in self.results]
        if PROGRESSBAR == 2:
            return self.results.get()
        else:
            return self.results

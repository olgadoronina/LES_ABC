import logging
import multiprocessing as mp
from time import sleep


class Parallel(object):

    def __init__(self, N_total, progressbar, processes=mp.cpu_count()):
        self.pool = mp.Pool(processes=processes)
        self.N = N_total
        self.results = None
        self.bar = progressbar
        logging.info('\n' + str(processes) + " workers")

    def run(self, func, tasks):
        if self.bar == 1:
            from tqdm import tqdm
            self.results = []
            with tqdm(total=len(tasks)) as pbar:
                for i, res in tqdm(enumerate(self.pool.imap_unordered(func, tasks)), desc='ABC algorithm'):
                    self.results.append(res)
                    pbar.update()
            pbar.close()
        elif self.bar == 2:
            self.results = self.pool.map_async(func, tasks)
            while not self.results.ready():
                done = len(tasks) - self.results._number_left*self.results._chunksize
                logging.info("Done {}% ({}/{})".format(int(done/len(tasks)*100), done, len(tasks)))
                sleep(20)
            self.pool.close()
            self.pool.join()
        else:
            self.results = self.pool.map(func, tasks)
            self.pool.close()
        self.pool.terminate()

    def get_results(self):
        if self.bar == 1:
            return [x for x in self.results]
        if self.bar == 2:
            return self.results.get()
        else:
            return self.results


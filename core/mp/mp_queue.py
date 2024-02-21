import queue
from multiprocessing import Queue, Lock


class MessageQueue(object):
    def __init__(self, q_size=0):
        self.queue = Queue(maxsize=q_size)
        self.max_size = q_size
        self.lock = Lock()

    def put(self, data, block=True, to=None, discard=True):
        with self.lock:
            if self.max_size == 0:
                self.queue.put(data, block, to)
            else:
                try:
                    self.queue.put(data, block, to)
                except queue.Full:
                    if discard is True:
                        # ignore data
                        return
                    else:   # replace
                        _tmp = list()
                        while not self.queue.empty():
                            d = self.queue.get()
                            _tmp.append(d)
                        _tmp[-1] = data
                        while len(_tmp) != 0:
                            d = _tmp.pop(0)
                            self.queue.put(d)

    def get(self, block=True, timeout=1, default=None):
        """
        if block is False, timeout ignore
        """
        with self.lock:
            if self.queue.empty() is False:
                data = self.queue.get(block=block, timeout=timeout)
            else:
                data = default

        return data

    def is_emtpy(self):
        return self.queue.empty()

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, size):
        self._max_size = size

    def queue_size(self):
        return self.queue.qsize()

    def close(self):
        self.queue.close()

    @staticmethod
    def Full():
        return queue.Full

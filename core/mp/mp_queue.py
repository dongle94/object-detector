import queue
from multiprocessing import Queue, Lock

import time

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
                        print("큐가 꽉차서 버립니다.", data)
                        return
                    else:   # replace
                        print("큐 대체 시작", self.queue.full())
                        t = time.time()
                        _tmp = list()
                        while not self.queue.empty():
                            d = self.queue.get()
                            print(f"뺐다 {d}")
                            _tmp.append(d)
                        print("대체 전", _tmp)
                        _tmp[-1] = data
                        print("대체 후", _tmp)
                        while len(_tmp) != 0:
                            d = _tmp.pop(0)
                            print(f"넣는다 {d}")
                            self.queue.put(d)
                        print(f"큐가 꽉차서 대체합니다. {time.time()-t:.5f}")

    def get(self, block=True, timeout=1):
        """
        if block is False, timeout ignore
        """
        with self.lock:
            data = self.queue.get(block=block, timeout=timeout)

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

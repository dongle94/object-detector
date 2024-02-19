from queue import Queue
from threading import Lock


class MessageQueue(object):

    def __init__(self, q_size=0):
        self.queue = Queue(maxsize=q_size)
        self.lock = Lock()

    def put(self, data):
        self.queue.put(data)

    def get(self, block=True, timeout=None):
        """
        if block is False, timeout ignore
        """
        with self.lock:
            data = self.queue.get(block=block, timeout=timeout)

        return data

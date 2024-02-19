from queue import Queue

class MessageQueue(object):

    def __init__(self):

        self.queue = Queue(maxsize=0)

    def put(self, data):
        self.queue.put(data)

    def __repr__(self):
        # return self.que
        pass
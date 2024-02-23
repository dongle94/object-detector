import signal
from multiprocessing import Queue, Process

from core.mp.mp_queue import MessageQueue


class Job(object):
    def __init__(self):
        self.name = self.__class__.__name__

    def init(self):
        pass

    def process(self, data):
        print(f"{self.name}: {data}!")
        return data

    def __repr__(self):
        return f"{self.name}"

    def close(self):
        pass


# class Task(object):
#     def __init__(self, job=None, pool_size=1, use_default_queue=True):
#         if use_default_queue:
#             self.input_queue = MessageQueue(q_size=1)
#             self.output_queue = MessageQueue(q_size=1)
#         else:
#             self.input_queue = None
#             self.output_queue = None
#         self.job = job
#         self.output_timeout = 5
#
#         self._is_started = False
#         self._is_initialized = False
#         self._need_finish = False
#
#     def init(self):
#         if self._is_initialized:
#             return
#         self._is_initialized = True
#
#         # self.pool = thread_util.ThreadPool(self.pool_size)
#         if self.job:
#             try:
#                 self.job.init()
#             except Exception as e:
#                 msg = e.message if hasattr(e, 'message') else e
#                 print(f"Failed initializing job {self}, {msg}")
#                 raise e
#
#         signal.signal(signal.SIGTERM, self._stop_signal)
#
#     def start(self):
#         if self._is_started:
#             return
#         self._is_started = True
#         self.init()
#         self.work_loop()
#
#     def stop(self):
#         self._need_finish = True
#         self.job.close()
#         if self.input_queue:
#             self.input_queue.close()
#         if self.output_queue:
#             self.output_queue.close()
#
#     def _stop_signal(self):
#         self._need_finish = True
#
#     def work_loop(self):
#         print(f"Started {self}")
#         self._is_started = True
#         self.init()
#
#         try:
#             pass
#         except Exception as e:
#             print(f"Exception in _work_loop: {e}")
#
#         self.stop()
#         print(f"Finished {self}")
#
#     def _process(self, item):
#         result = self.job.process(item)
#         self._process_result(result, self.output_queue)
#
#     def _process_result(self, res, output_queue):
#         if output_queue is None or res is None:
#             return
#         while output_queue and self._need_finish is False:
#             try:
#                 output_queue.put(res, block=True, timeout=self.output_timeout)
#             except:
#                 pass


class TaskLauncher(object):
    def __init__(self):
        self.task_list = []

    def add_task(self, task):
        self.task_list.append(task)

    def remove_task(self, task):
        if task in self.task_list:
            task.stop()

            self.task_list.remove(task)

    def start(self):
        for t in self.task_list:
            t.start()

    def stop(self):
        for t in self.task_list:
            t.stop()

        self.task_list = []


# class MPTaskLauncher(Process):
#     def __init__(self, ):
#         super().__init__()
#
#
#         self.name
#         self.input_queue
#         self.output_queue
#
#
#
#     def run(self):
#         print(self.name)
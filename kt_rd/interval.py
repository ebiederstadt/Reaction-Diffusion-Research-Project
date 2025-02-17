import threading
import time


class SetInterval:
    def __init__(self, interval, action):
        self.interval = interval
        self.action = action
        self.stop_event = threading.Event()
        thread = threading.Thread(target=self._set_interval, daemon=True)
        thread.start()

    def _set_interval(self):
        next_time = time.time() + self.interval
        while not self.stop_event.wait(next_time - time.time()):
            next_time += self.interval
            self.action()

    def cancel(self):
        self.stop_event.set()

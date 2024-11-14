import os
import fcntl
import time

class PipeStreamer:
    '''
    This requires a reader to be ready first
    '''
    def __init__(self, pipe_path='/tmp/synapse_fifo'):
        self.pipe_path = pipe_path
        self.fifo = None

    def __enter__(self):
        self._create_pipe()
        self._open_pipe()
        self._set_non_blocking()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _create_pipe(self):
        if not os.path.exists(self.pipe_path):
            os.mkfifo(self.pipe_path)

    def _open_pipe(self):
        self.fifo = open(self.pipe_path, 'w')

    def _set_non_blocking(self):
        fd = self.fifo.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    def write(self, data):
        try:
            self.fifo.write(data)
            self.fifo.flush()
        except IOError:
            # Handle the case where the pipe is full
            pass

    def close(self):
        if self.fifo:
            self.fifo.close()

    @staticmethod
    def stream_data(pipe_path='/tmp/myfifo', interval=0.1):
        with PipeStreamer(pipe_path) as streamer:
            while True:
                streamer.write("Some data\n")
                time.sleep(interval)

import sys
from torch.utils.tensorboard import SummaryWriter
from typing import Any, List, Tuple, Union

class TFLogger(object):
    """Tensorboard logger."""
    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)
    def graph_summary(self, model):
        self.writer.add_graph(model)        
    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, step)
    def histogram_summary(self, tag, value, step):      
        """Add histogram summary."""
        self.writer.add_histogram(tag, value, step)
    def image_summary(self, tag, image, step):      
        """Add image summary."""
        # image is [k, w, h, c] format   ##obsolete
        # image is CHW format
        if len(image.size())>3:
            self.writer.add_images(tag, image, step)
        else:
            self.writer.add_image(tag, image, step)
        
class DummyLogger(object):
    """Dummy logger."""
    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.log_dir = log_dir
    def graph_summary(self, model):
        return None
    def scalar_summary(self, tag, value, step):
        return None
    def histogram_summary(self, tag, value, step):
        return None
    def image_summary(self, tag, image, step):
        return None
    
# from stylegan
class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""
    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None
        if file_name is not None:
            self.file = open(file_name, file_mode)
        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
    
        
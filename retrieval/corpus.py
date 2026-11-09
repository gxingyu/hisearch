import json
import os
from pathlib import Path

import numpy as np


class Corpus:
    def __init__(self, path, offsets_path):
        self.path = Path(path)
        self.offsets = np.load(offsets_path, mmap_mode="r")
        self.descriptor = os.open(self.path, os.O_RDONLY)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, position):
        position = int(position)
        if not 0 <= position < len(self):
            raise IndexError(position)
        start = int(self.offsets[position])
        size = int(self.offsets[position + 1]) - start
        return json.loads(os.pread(self.descriptor, size, start))

    def close(self):
        os.close(self.descriptor)

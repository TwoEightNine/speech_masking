import random

import numpy as np
from abc import ABC, abstractmethod


class MaskingRule(ABC):

    @abstractmethod
    def mask(self, spec: np.array) -> np.array:
        pass

    @abstractmethod
    def unmask(self, spec: np.array) -> np.array:
        pass


class OffsetMaskingRule(MaskingRule):

    def __init__(self, offset):
        if offset <= 0:
            raise Exception(f"Offset must be positive! Received {offset}")
        self.offset = offset

    def mask(self, spec: np.array) -> np.array:
        spec[:-self.offset] = spec[self.offset:]
        spec[-self.offset:] = 0
        return spec

    def unmask(self, spec: np.array) -> np.array:
        spec[self.offset:] = spec[:-self.offset]
        spec[:self.offset] = 0
        return spec


class RandomOffsetMaskingRule(MaskingRule):

    def __init__(self, offset_min, offset_max, change_period):
        if offset_min <= 0 or offset_max <= 0:
            raise Exception(f"Offset must be positive! Received {offset_min}, {offset_max}")
        if change_period <= 0:
            raise Exception(f"Change period must be positive! Received {change_period}")

        self.offset_min = offset_min
        self.offset_max = offset_max
        self.change_period = change_period

        self._prev_offset = None
        self._next_offset = None

        self._current_step = 0
        self.__generate_offset()

    def mask(self, spec: np.array) -> np.array:
        if self._current_step == self.change_period:
            self.__generate_offset()

        offset = self.__get_current_offset()
        offset = random.randint(self.offset_min, self.offset_max)
        self._current_step += 1

        spec[:-offset] = spec[offset:]
        spec[-offset:] = 0
        return spec

    def unmask(self, spec: np.array) -> np.array:
        offset = (self.offset_min + self.offset_max) // 2
        spec[offset:] = spec[:-offset]
        spec[:offset] = 0
        return spec

    def __get_current_offset(self):
        range = self._next_offset - self._prev_offset
        delta = int(range * self._current_step / self.change_period)
        return self._prev_offset + delta

    def __generate_offset(self):
        self._prev_offset = self._next_offset
        self._next_offset = random.randint(self.offset_min, self.offset_max)

        if not self._prev_offset:
            self._prev_offset = random.randint(self.offset_min, self.offset_max)

        self._current_step = 0

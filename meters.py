import bisect
from collections import OrderedDict
from typing import Dict, Optional
import torch
import numpy as np


class Meter(object):
    """Base class for Meters."""

    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    @property
    def smoothed_value(self) -> float:
        """Smoothed value used for logging."""
        raise NotImplementedError


def safe_round(number, ndigits):
    if hasattr(number, "__round__"):
        return round(number, ndigits)
    elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
        return safe_round(number.item(), ndigits)
    elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
        return safe_round(number.item(), ndigits)
    else:
        return number


class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def __init__(self, round: Optional[int] = None):
        self.round = round
        self.reset()

    def reset(self):
        self.val = None  # most recent update
        self.sum = 0  # sum from all updates
        self.count = 0  # total n from all updates

    def update(self, val, n=1):
        if val is not None:
            self.val = val
            if n > 0:
                self.sum = self.sum + (val * n)
                self.count = self.count + n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else self.val

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val


class SumMeter(Meter):
    """Computes and stores the sum"""

    def __init__(self, round: Optional[int] = None):
        self.round = round
        self.reset()

    def reset(self):
        self.sum = 0  # sum from all updates

    def update(self, val):
        if val is not None:
            self.sum = self.sum + val

    @property
    def smoothed_value(self) -> float:
        val = self.sum
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val


class MetersDict(OrderedDict):
    """A sorted dictionary of :class:`Meters`.

    Meters are sorted according to a priority that is given when the
    meter is first added to the dictionary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priorities = []

    def __setitem__(self, key, value):
        assert key not in self, "MetersDict doesn't support reassignment"
        priority, value = value
        bisect.insort(self.priorities, (priority, len(self.priorities), key))
        super().__setitem__(key, value)
        for _, _, key in self.priorities:  # reorder dict to match priorities
            self.move_to_end(key)

    def add_meter(self, key, meter, priority):
        self.__setitem__(key, (priority, meter))

    def get_smoothed_value(self, key: str) -> float:
        """Get a single smoothed value."""
        meter = self[key]
        if isinstance(meter, MetersDict._DerivedMeter):
            return meter.fn(self)
        else:
            return meter.smoothed_value

    def get_smoothed_values(self) -> Dict[str, float]:
        """Get all smoothed values."""
        return OrderedDict(
            [(key, self.get_smoothed_value(key)) for key in self.keys() if not key.startswith("_")]
        )

    def reset(self):
        """Reset Meter instances."""
        for meter in self.values():
            if isinstance(meter, MetersDict._DerivedMeter):
                continue
            meter.reset()

    class _DerivedMeter(Meter):
        """A Meter whose values are derived from other Meters."""

        def __init__(self, fn):
            self.fn = fn

        def reset(self):
            pass


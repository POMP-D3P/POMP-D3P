import contextlib
import uuid
from collections import defaultdict
from typing import Callable, List, Optional
from meters import *


_aggregators = OrderedDict()
_active_aggregators = OrderedDict()
_active_aggregators_cnt = defaultdict(lambda: 0)


def reset() -> None:
    """Reset all metrics aggregators."""
    _aggregators.clear()
    _active_aggregators.clear()
    _active_aggregators_cnt.clear()

    # The "default" aggregator observes all logged values.
    _aggregators["default"] = MetersDict()
    _active_aggregators["default"] = _aggregators["default"]
    _active_aggregators_cnt["default"] = 1


reset()


@contextlib.contextmanager
def aggregate(name: Optional[str] = None, new_root: bool = False):
    """Context manager to aggregate metrics under a given name.

    Aggregations can be nested. If *new_root* is ``False``, then logged
    metrics will be recorded along the entire stack of nested
    aggregators, including a global "default" aggregator. If *new_root*
    is ``True``, then this aggregator will be the root of a new
    aggregation stack, thus bypassing any parent aggregators.

    Note that aggregation contexts are uniquely identified by their
    *name* (e.g., train, valid). Creating a context with an existing
    name will reuse the corresponding :class:`MetersDict` instance.
    If no name is given, then a temporary aggregator will be created.

    Usage::

        with metrics.aggregate("train"):
            for step, batch in enumerate(epoch):
                with metrics.aggregate("train_inner") as agg:
                    metrics.log_scalar("loss", get_loss(batch))
                    if step % log_interval == 0:
                        print(agg.get_smoothed_value("loss"))
                        agg.reset()
        print(metrics.get_smoothed_values("train")["loss"])

    Args:
        name (str): name of the aggregation. Defaults to a
            random/temporary name if not given explicitly.
        new_root (bool): make this aggregation the root of a new
            aggregation stack.
    """
    if name is None:
        # generate a temporary name
        name = str(uuid.uuid4())
        assert name not in _aggregators
        agg = MetersDict()
    else:
        assert name != "default"
        agg = _aggregators.setdefault(name, MetersDict())

    if new_root:
        backup_aggregators = _active_aggregators.copy()
        _active_aggregators.clear()
        backup_aggregators_cnt = _active_aggregators_cnt.copy()
        _active_aggregators_cnt.clear()

    _active_aggregators[name] = agg
    _active_aggregators_cnt[name] += 1

    yield agg

    _active_aggregators_cnt[name] -= 1
    if _active_aggregators_cnt[name] == 0 and name in _active_aggregators:
        del _active_aggregators[name]

    if new_root:
        _active_aggregators.clear()
        _active_aggregators.update(backup_aggregators)
        _active_aggregators_cnt.clear()
        _active_aggregators_cnt.update(backup_aggregators_cnt)


def get_active_aggregators() -> List[MetersDict]:
    return list(_active_aggregators.values())


def log_scalar(
    key: str, value: float, weight: float = 1, priority: int = 10, round: Optional[int] = None,
):
    """Log a scalar value.

    Args:
        key (str): name of the field to log
        value (float): value to log
        weight (float): weight that this value contributes to the average.
            A weight of 0 will always log the latest value.
        priority (int): smaller values are logged earlier in the output
        round (Optional[int]): number of digits to round to when displaying
    """
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, AverageMeter(round=round), priority)
        agg[key].update(value, weight)


def log_scalar_sum(
    key: str, value: float, priority: int = 10, round: Optional[int] = None,
):
    """Log a scalar value that is summed for reporting.

    Args:
        key (str): name of the field to log
        value (float): value to log
        priority (int): smaller values are logged earlier in the output
        round (Optional[int]): number of digits to round to when displaying
    """
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, SumMeter(round=round), priority)
        agg[key].update(value)


def reset_meter(name: str, key: str) -> None:
    """Reset Meter instance aggregated under a given *name* and *key*."""
    meter = get_meter(name, key)
    if meter is not None:
        meter.reset()


def reset_meters(name: str) -> None:
    """Reset Meter instances aggregated under a given *name*."""
    meters = get_meters(name)
    if meters is not None:
        meters.reset()


def get_meter(name: str, key: str) -> Meter:
    """Get a single Meter instance aggregated under *name* and *key*.

    Returns:
        Meter or None if no metrics have been logged under *name* and *key*.
    """
    if name not in _aggregators:
        return None
    return _aggregators[name].get(key, None)


def get_meters(name: str) -> MetersDict:
    """Get Meter instances aggregated under a given *name*.

    Returns:
        MetersDict or None if no metrics have been logged under *name*.
    """
    return _aggregators.get(name, None)


def get_smoothed_value(name: str, key: str) -> float:
    """Get a single smoothed value.

    Raises:
        KeyError: if no metrics have been logged under *name* and *key*.
    """
    return _aggregators[name].get_smoothed_value(key)


def get_smoothed_values(name: str) -> Dict[str, float]:
    """Get smoothed values aggregated under a given *name*.

    Raises:
        KeyError: if no metrics have been logged under *name*.
    """
    return _aggregators[name].get_smoothed_values()

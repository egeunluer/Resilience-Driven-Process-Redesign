"""
Working-time clock shared by the miner and the simulator.

A wait that ends in a work session always ends during working hours in the log (a
worker started that session). Measured in wall-clock time and replayed from a random
moment, such a wait would often end at night and then queue until the next shift,
counting the off-hours twice. Measuring these waits in working time (only the
calendar's working days and hours count) and converting back at simulation time keeps
their end inside the shift, as in the log.

Working time is counted in seconds from a Monday-00:00 epoch, on `days` (0 = Monday)
between hour_min and hour_max local time.
"""
from datetime import datetime, timedelta

import numpy as np

EPOCH = datetime(2000, 1, 3)    # a Monday


class WorkingClock:
    def __init__(self, days, hour_min, hour_max):
        self.days = sorted(days)
        self.h0 = hour_min * 3600
        self.H = (hour_max - hour_min) * 3600
        self.per_week = len(self.days) * self.H
        # working seconds before the start of each weekday within a week
        self.cum = np.zeros(8)
        for d in range(7):
            self.cum[d + 1] = self.cum[d] + (self.H if d in self.days else 0)
        self.is_day = np.array([d in self.days for d in range(7)])

    def to_working(self, local_seconds):
        """Working seconds since EPOCH for local wall-clock seconds since EPOCH (array or scalar)."""
        s = np.asarray(local_seconds, dtype=float)
        week, rem = np.divmod(s, 7 * 86400)
        wd, tod = np.divmod(rem, 86400)
        wd = wd.astype(int)
        within = np.where(self.is_day[wd], np.clip(tod - self.h0, 0, self.H), 0.0)
        return week * self.per_week + self.cum[wd] + within

    def from_working(self, b):
        """Earliest local seconds since EPOCH at which working time `b` is reached (scalar)."""
        week, rem = divmod(float(b), self.per_week)
        for d in self.days:
            if rem < self.cum[d] + self.H or d == self.days[-1]:
                return week * 7 * 86400 + d * 86400 + self.h0 + (rem - self.cum[d])
        raise AssertionError

    @staticmethod
    def seconds(dt):
        """Local seconds since EPOCH for a naive local datetime."""
        return (dt - EPOCH).total_seconds()

    @staticmethod
    def datetime(seconds):
        return EPOCH + timedelta(seconds=seconds)

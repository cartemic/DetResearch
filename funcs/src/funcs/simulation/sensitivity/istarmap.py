# https://stackoverflow.com/a/57364423
# iterable pool.starmap for use with tqdm
import sys
from multiprocessing import pool as mpp

# Allow private member access and magic values
# ruff: noqa: SLF001,PLR2004

major = sys.version_info.major
minor = sys.version_info.minor
if (major > 3) or (major == 3 and minor >= 8):
    # noinspection PyUnresolvedReferences,PyArgumentList
    def istarmap(self, func, iterable, chunksize=1):
        """
        starmap-version of imap
        """
        self._check_running()
        if chunksize < 1:
            raise ValueError(f"Chunksize must be 1+, not {chunksize:n}")

        task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
        result = mpp.IMapIterator(self)
        self._taskqueue.put(
            (self._guarded_task_generation(result._job, mpp.starmapstar, task_batches), result._set_length)
        )
        return (item for chunk in result for item in chunk)
else:
    # noinspection PyUnresolvedReferences,PyArgumentList
    def istarmap(self, func, iterable, chunksize=1):
        """
        starmap-version of imap
        """
        if self._state != mpp.RUN:
            raise ValueError("Pool not running")

        if chunksize < 1:
            raise ValueError(f"Chunksize must be 1+, not {chunksize:n}")

        task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
        result = mpp.IMapIterator(self._cache)
        self._taskqueue.put(
            (self._guarded_task_generation(result._job, mpp.starmapstar, task_batches), result._set_length)
        )
        return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap

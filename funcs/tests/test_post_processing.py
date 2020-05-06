import os
from funcs import post_processing as pp


DRIVE = {
    "posix": "/d",
    "nt": "D\\"
}
DIR_RAW = os.path.join(
    DRIVE[os.name],
    "Data",
    "Raw"
)


class TestCollectOldData:
    # noinspection PyTypeChecker
    def test_good_dir(self):
        # this date should have 25 completed tests per the schlieren output
        assert len(pp._ProcessOldData._collect_test_dirs(DIR_RAW, "2019-10-22")) == 25

    # noinspection PyTypeChecker
    def test_bad_dir(self):
        # make sure things don't break on a day with no tests
        assert len(pp._ProcessOldData._collect_test_dirs(DIR_RAW, "2049-37-55")) == 0


class TestCollectSchlierenDirs:
    # noinspection PyTypeChecker
    def test_good_old_dir(self):
        # this old date should have 25 completed tests per the schlieren output
        assert len(pp._collect_schlieren_dirs(DIR_RAW, "2019-10-22")) == 25

    # noinspection PyTypeChecker
    def test_good_new_dir(self):
        # this new date should have 14 completed tests per the schlieren output
        assert len(pp._collect_schlieren_dirs(DIR_RAW, "2019-11-07")) == 14

    # noinspection PyTypeChecker
    def test_bad_dir(self):
        # make sure things don't break on a day with no tests
        assert len(pp._collect_schlieren_dirs(DIR_RAW, "2049-37-55")) == 0

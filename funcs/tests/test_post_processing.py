import os
from funcs import post_processing as pp


DIR_RAW = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
    "post_processing"
)


class TestCollectOldData:
    # noinspection PyTypeChecker
    def test_good_dir(self):
        assert len(pp._ProcessStructure0._collect_test_dirs(
            DIR_RAW,
            "2019-10-22"
        )) == 4

    # noinspection PyTypeChecker
    def test_bad_dir(self):
        # make sure things don't break on a day with no tests
        assert len(pp._ProcessStructure0._collect_test_dirs(
            DIR_RAW,
            "2049-37-55"
        )) == 0


class TestCollectSchlierenDirs:
    # noinspection PyTypeChecker
    def test_good_old_dir(self):
        # for structure 0
        assert len(pp._collect_schlieren_dirs(DIR_RAW, "2019-10-22")) == 4

    # noinspection PyTypeChecker
    def test_good_new_dir(self):
        # for structures 1, 2
        assert len(pp._collect_schlieren_dirs(DIR_RAW, "2019-11-07")) == 4

    # noinspection PyTypeChecker
    def test_bad_dir(self):
        # make sure things don't break on a day with no tests
        assert len(pp._collect_schlieren_dirs(DIR_RAW, "2049-37-55")) == 0

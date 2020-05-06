import sys

import numpy as np
import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from funcs.gui.ui import calWindow


class CalWindow(QDialog, calWindow.Ui_MainWindow):
    def __init__(self, parent=None):
        super(CalWindow, self).__init__(parent)
        self.setupUi(self)

        # data storage
        self.store = None
        self.df = None
        self.date = None

        # don't let user check boxes
        # done this way because fully disabling them doesn't allow the GUI to
        # change checkbox values, which we want
        def gtfo():
            self.sender().setChecked(
                not self.sender().isChecked()
            )
        self.chkStoredCenterline.clicked.connect(gtfo)
        self.chkStoredNear.clicked.connect(gtfo)
        self.chkStoredFar.clicked.connect(gtfo)

        # DBAA
        for item in (self.inpStartShot, self.inpEndShot):
            item.setFocusPolicy(Qt.StrongFocus)
        for item in (
            self.btnLoadHDF5,
            self.btnExit,
            self.btnLoadAndCalibrate,
            self.btnStoreCalibration
        ):
            item.setFocusPolicy(Qt.NoFocus)

        # validate inputs
        re = QRegExp(r"(^[0-9]+$|^$|^\s$)")
        shot_num_validator = QRegExpValidator(re)
        self.inpStartShot.setValidator(shot_num_validator)
        self.inpEndShot.setValidator(shot_num_validator)

        # link buttons to functions
        self.btnExit.clicked.connect(sys.exit)
        self.btnLoadHDF5.clicked.connect(self.load_hdf5)
        self.cboxSelectDate.activated.connect(self.select_date)
        for item in (self.inpStartShot, self.inpEndShot):
            item.editingFinished.connect(self.check_stored_calibrations)

    def __del__(self):
        if hasattr(self.store, "close"):
            # noinspection PyCallingNonCallable
            self.store.close()

    def load_hdf5(self):
        self.btnLoadHDF5.clearFocus()
        self.inpStartShot.setFocus()
        default_path = "/d/Data/Processed/Data/"
        # noinspection PyArgumentList
        f = QFileDialog.getOpenFileName(
            parent=QFileDialog(),
            caption="Load Data Store",
            directory=default_path,
            filter="*.h5 *.hdf5",
        )[0]
        if len(f) > 0:
            self.store = pd.HDFStore(f)
            self.cboxSelectDate.clear()
            self.df = self.store["data"]
            self.cboxSelectDate.addItems(self.df["date"].unique())
            for item in (
                self.inpStartShot,
                self.inpEndShot,
                self.btnLoadAndCalibrate,
                self.btnStoreCalibration
            ):
                item.setEnabled(True)
            self.select_date()

    def select_date(self):
        self.check_stored_calibrations()
        self.date = self.cboxSelectDate.currentText()
        self.check_stored_calibrations()
        # set start and end shot

    def check_stored_calibrations(self):
        df_date = self.df[self.df["date"] == self.date]
        # TODO: this should be its own function
        try:
            start_shot = int(self.inpStartShot.text())
        except ValueError:
            start_shot = df_date["shot"].min()

        try:
            end_shot = int(self.inpEndShot.text())
        except ValueError:
            end_shot = df_date["shot"].max()

        pal = self.chkStoredCenterline.palette()
        if not np.any(np.isnan((start_shot, end_shot))):
            df_filtered = df_date[
                (df_date["shot"] >= start_shot) &
                (df_date["shot"] <= end_shot)
            ]
            self.chkStoredCenterline.setChecked(
                np.any(df_filtered["spatial_factor"].notna())
            )
            # pal.setColor(QPalette.Button, QColor(255, 0, 0))
            if len(df_filtered) == 0:
                set_color = "color: red"
            elif np.all(df_filtered["spatial_factor"].notna()):
                set_color = "color: green"
            elif np.any(df_filtered["spatial_factor"].notna()):
                set_color = "color: orange"
            else:
                set_color = None

            print(df_filtered["spatial_factor"])
            self.chkStoredCenterline.setStyleSheet(set_color)

        # else:
        #     pal.setColor(QPalette.Button, QColor(0, 255, 0))
        self.chkStoredCenterline.setPalette(pal)

    def load_spatial_file(self):
        pass

    def calibrate(self):
        pass

    def load_and_calibrate(self):
        self.load_spatial_file()
        self.calibrate()

    def check_overwrite_calibration(self):
        pass

    def store_calibration(self):
        pass

    def estimate_missing_calibration(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = CalWindow()
    form.show()
    app.exec_()

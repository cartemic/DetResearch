import os
import sys

import numpy as np
import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .ui import calWindow
from .._dev import d_drive
from ..schlieren import collect_spatial_calibration


class CalWindow(QDialog, calWindow.Ui_MainWindow):
    def __init__(self, parent=None):
        super(CalWindow, self).__init__(parent)
        self.setupUi(self)

        # data storage
        self.store = None
        self.df = None
        self.date = None
        self.calibration = None

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
        self.cboxSelectDate2.activated.connect(self.select_date)
        self.chkUseDate2.clicked.connect(self.use_second_date)
        self.btnLoadAndCalibrate.clicked.connect(self.load_and_calibrate)
        for item in (
                self.inpStartShot,
                self.inpEndShot,
                self.inpStartShot2,
                self.inpEndShot2,
        ):
            item.editingFinished.connect(self.check_stored_calibrations)

    def __del__(self):
        if hasattr(self.store, "close"):
            # noinspection PyCallingNonCallable
            self.store.close()

    def load_hdf5(self):
        self.btnLoadHDF5.clearFocus()
        self.inpStartShot.setFocus()
        default_path = os.path.join(
            d_drive,
            "Data",
            "Processed",
            "Data"
        )
        # noinspection PyArgumentList
        f = QFileDialog.getOpenFileName(
            parent=QFileDialog(),
            caption="Load Data Store",
            directory=default_path,
            filter="*.h5 *.hdf5",
            options=QFileDialog.DontUseNativeDialog
        )[0]
        if len(f) > 0:
            self.store = pd.HDFStore(f)
            self.df = self.store["data"]
            if "spatial_near" in self.df.keys():
                dates = self.df["date"].unique()
                self.cboxSelectDate.clear()
                for item in (
                    self.cboxSelectDate,
                    self.cboxSelectDate2
                ):
                    item.addItems(dates)
                for item in (
                    self.cboxSelectDate,
                    self.inpStartShot,
                    self.inpEndShot,
                    self.btnLoadAndCalibrate,
                    self.btnStoreCalibration,
                    self.lblStartShot,
                    self.lblEndShot,
                    self.lblSelectDate,
                    self.chkUseDate2,
                ):
                    item.setEnabled(True)
                self.select_date()
            else:
                warning_box = QMessageBox()
                warning_box.setIcon(QMessageBox.Warning)
                warning_box.setText("spatial_near not a valid key")
                warning_box.setWindowTitle("Invalid HDF5 Store!")
                warning_box.setStandardButtons(QMessageBox.Ok)
                # warning_box.setWindowIcon()
                warning_box.exec_()

    def select_date(self):
        date = self.cboxSelectDate.currentText()
        self.check_stored_calibrations()

        self.cboxSelectDate2.clear()
        self.cboxSelectDate2.addItems(self.df["date"].unique())
        for i in range(self.cboxSelectDate2.count()):
            if self.cboxSelectDate2.itemText(i) == date:
                self.cboxSelectDate2.removeItem(i)
                break
        self.date = date
        self.check_stored_calibrations()
        # set start and end shot

    def use_second_date(self):
        enable = self.chkUseDate2.isChecked()
        for item in (
            self.cboxSelectDate2,
            self.inpStartShot2,
            self.inpEndShot2,
            self.lblSelectDate2,
            self.lblStartShot2,
            self.lblEndShot2,
        ):
            item.setEnabled(enable)
            self.check_stored_calibrations()

    def check_stored_calibrations(self):
        # df_date = self.df[self.df["date"] == self.date]
        date_0 = self.cboxSelectDate.currentText()
        date_1 = self.cboxSelectDate2.currentText()

        # TODO: this should be its own function
        try:
            start_shot_0 = int(self.inpStartShot.text())
        except ValueError:
            start_shot_0 = self.df[self.df["date"] == date_0]["shot"].min()

        try:
            end_shot_0 = int(self.inpEndShot.text())
        except ValueError:
            end_shot_0 = self.df[self.df["date"] == date_0]["shot"].max()

        if not self.cboxSelectDate2.isEnabled():
            date_1 = date_0
            start_shot_1 = start_shot_0
            end_shot_1 = end_shot_0
        else:
            try:
                start_shot_1 = int(self.inpStartShot2.text())
            except ValueError:
                start_shot_1 = self.df[self.df["date"] == date_1]["shot"].min()

            try:
                end_shot_1 = int(self.inpEndShot2.text())
            except ValueError:
                end_shot_1 = self.df[self.df["date"] == date_1]["shot"].max()

        pal = self.chkStoredCenterline.palette()
        if not np.any(np.isnan((
                start_shot_0,
                end_shot_0,
                start_shot_1,
                end_shot_1,
        ))):
            df_filtered = self.filter_data(
                self.df,
                date_0,
                start_shot_0,
                end_shot_0,
                date_1,
                start_shot_1,
                end_shot_1
            )
            print(date_0, start_shot_0, end_shot_0)
            print(date_1, start_shot_1, end_shot_1)
            # print(df_filtered)
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

            # print(df_filtered["spatial_factor"])
            self.chkStoredCenterline.setStyleSheet(set_color)

        # else:
        #     pal.setColor(QPalette.Button, QColor(0, 255, 0))
        self.chkStoredCenterline.setPalette(pal)

    def load_spatial_file(self):
        date = self.cboxSelectDate.currentText()
        default_path = os.path.join(
            d_drive,
            "Data",
            "Raw",
            date
        )
        # noinspection PyArgumentList
        f = QFileDialog.getOpenFileName(
            parent=QFileDialog(),
            caption="Load Spatial Image",
            directory=default_path,
            filter="*.tif",
            options=QFileDialog.DontUseNativeDialog
        )[0]
        if len(f) > 0:
            return f

    @staticmethod
    def get_calibration(img_path):
        return collect_spatial_calibration(img_path)

    def load_and_calibrate(self):
        img_path = self.load_spatial_file()
        w = PlotWindow()
        print(
            collect_spatial_calibration(
                img_path,
                plot_window=w,
                msg_box=LineCountDialog
            )
        )

    def check_overwrite_calibration(self):
        pass

    def store_calibration(self):
        pass

    def estimate_missing_calibration(self):
        pass


class PlotWindow(QDialog):
    resized = pyqtSignal()

    def __init__(self):
        super().__init__()

        title = "Schlieren Spatial Calibration"
        self.setWindowTitle(title)
        self.canvas = Canvas(self)
        self.ax = self.canvas.axes
        self.canvas.mpl_connect(
            'button_press_event',
            self.button_press_callback
        )
        # noinspection PyUnresolvedReferences
        self.resized.connect(self.resize_canvas)

    def resizeEvent(self, event):
        # noinspection PyUnresolvedReferences
        self.resized.emit()
        return super(PlotWindow, self).resizeEvent(event)

    def resize_canvas(self):
        w = self.frameGeometry().width()
        h = self.frameGeometry().height()
        self.canvas.resize(w, h)

    def imshow(self, img):
        self.ax.imshow(img)
        self.ax.axis("off")

        # FigureCanvasQTAgg.setSizePolicy(
        #     self.canvas,
        #     QSizePolicy.Expanding,
        #     QSizePolicy.Expanding,
        # )
        FigureCanvasQTAgg.updateGeometry(self.canvas)
        self.showMaximized()

    def button_press_callback(self, event):
        if event.button == 2:
            # middle click
            self.close()


class Canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):

        fig = Figure()
        self.axes = fig.add_subplot(111)

        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)


class LineCountDialog(QInputDialog):
    def __init__(self):
        super().__init__()
        self.center()
        user_input = self.getInt(self, "Input number of boxes", "# Boxes:")
        if user_input[1] is False:
            self.num_boxes = np.NaN
        else:
            self.num_boxes = user_input[0]

    def center(self):
        resolution = QDesktopWidget().screenGeometry()
        self.move(
            (resolution.width() - self.frameSize().width()) / 2,
            (resolution.height() - self.frameSize().height()) / 2
        )


if __name__ == "__main__":
    pass
    # this doesn't work within this file for relative import reasons
    # app = QApplication(sys.argv)
    # form = CalWindow()
    # form.show()
    # app.exec_()

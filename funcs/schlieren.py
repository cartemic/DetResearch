import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import uncertainties as un
import uncertainties.unumpy as unp
from matplotlib import pyplot as plt
from matplotlib import widgets
from skimage import io
from skimage.filters import sobel_v

from ._dev import d_drive, convert_dir_to_local
from .uncertainty import add_uncertainty_terms, u_cell

u_cell = u_cell["schlieren"]


def get_spatial_dir(
        date,
        base_dir=os.path.join(
            d_drive,
            "Data",
            "Raw"
        )
):
    _dir_date = os.path.join(
        base_dir,
        date,
        "spatial"
    )
    if not os.path.exists(_dir_date):
        _dir_date = os.path.join(
            base_dir,
            date,
            "Camera",
            "spatial"
        )
        if not os.path.exists(_dir_date):
            warnings.warn("directory not found: %s" % _dir_date)
            _dir_date = np.NaN

    return _dir_date


def get_varied_spatial_dir(
        spatial_date_dir,
        spatial_dir_name,
        base_dir=os.path.join(
            d_drive,
            "Data",
            "Raw"
        )
):
    _dir_date = os.path.join(
        base_dir,
        spatial_date_dir,
        spatial_dir_name,
    )
    if not os.path.exists(_dir_date):
        warnings.warn("directory not found: %s" % _dir_date)
        _dir_date = np.NaN

    return _dir_date


def get_spatial_loc(
        date,
        which="near",
        base_dir=os.path.join(
            d_drive,
            "Data",
            "Raw"
        )
):
    _dir_date = get_spatial_dir(
        date,
        base_dir
    )
    _near = "near.tif"
    _far = "far.tif"

    if which == "near":
        return os.path.join(_dir_date, _near)
    elif which == "far":
        return os.path.join(_dir_date, _far)
    elif which == "both":
        return [os.path.join(_dir_date, _near), os.path.join(_dir_date, _far)]
    else:
        raise ValueError("bad value of `which`")


def _find_images_in_dir(
        directory,
        data_type=".tif"
):
    """
    Finds all files in a directory of the given file type. This function should
    be applied to either a `bg` or `frames` directory from a single day of
    testing.

    Parameters
    ----------
    directory : str
        Directory to search
    data_type : str
        File type to search for

    Returns
    -------
    list
    """
    last_n = -len(data_type)
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f[last_n:] == data_type
    ])


def find_shot_images(
        dir_shot,
        data_type=".tif"
):
    """
    Collects all background and frame images for a single shot directory. Shot
    directory should contain `bg` and `frames` sub-directories.

    Parameters
    ----------
    dir_shot : str
        Shot directory to collect images from
    data_type : str
        File type of schlieren images

    Returns
    -------
    list
        [[background image paths], [frame image paths]]
    """
    backgrounds = []
    frames = []
    for root, _, files in os.walk(dir_shot):
        curdir = os.path.split(root)[1]
        if curdir == "bg":
            backgrounds = _find_images_in_dir(root, data_type=data_type)
        elif curdir == "frames":
            frames = _find_images_in_dir(root, data_type=data_type)

    return [backgrounds, frames]


def average_frames(frame_paths):
    """
    Averages all frames contained within a list of paths

    Parameters
    ----------
    frame_paths : list
        Path to image frames to average

    Returns
    -------
    np.array
        Average image as a numpy array of float64 values
    """
    return np.array(
        [io.imread(frame) for frame in frame_paths],
        dtype='float64'
    ).mean(axis=0)


def bg_subtract_all_frames(dir_raw_shot):
    """
    Subtract the averaged background from all frames of schlieren data in a
    given shot.

    Parameters
    ----------
    dir_raw_shot : str
        Directory containing raw shot data output. Should have `bg` and
        `frames` sub-directories.

    Returns
    -------
    list
        List of background subtracted arrays
    """
    pth_list_bg, pth_list_frames = find_shot_images(dir_raw_shot)
    bg = average_frames(pth_list_bg)
    return [(io.imread(frame) - bg + 2**15) for frame in pth_list_frames]


def _maximize_window():
    # https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    plt_backend = plt.get_backend()
    mng = plt.get_current_fig_manager()
    if "Qt" in plt_backend:
        mng.window.showMaximized()
        return True
    elif "wx" in plt_backend:
        mng.frame.Maximize(True)
        return True
    elif "Tk" in plt_backend:
        mng.window_state('zoomed')
        return True
    else:
        print("figure out how to maximize for ", plt_backend)
        return False


def collect_spatial_calibration(
        spatial_file,
        line_color="r",
        marker_length_mm=0.2*25.4,
        save_output=False,
        px_only=False,
        apply_uncertainty=True,
        plot_window=None,
        msg_box=None
):  # pragma: no cover
    image = io.imread(spatial_file)

    if plot_window is not None:
        # called from Qt gui
        ax = plot_window.ax
        fig = plot_window.fig
    else:
        # not called form Qt gui
        fig, ax = plt.subplots(1, 1)

    fig.canvas.manager.window.move(0, 0)
    ax.axis("off")

    ax.imshow(image)
    cal_line = widgets.Line2D(
        [0, 100],
        [0, 100],
        c=line_color
    )
    ax.add_line(cal_line)

    # noinspection PyTypeChecker
    linebuilder = LineBuilder(cal_line)

    if plot_window is not None:
        # called from Qt gui
        plot_window.imshow(image)
        plot_window.exec_()
        if msg_box is None:
            raise ValueError("Lazy dev didn't error handle this! Aaahh!")
        num_boxes = msg_box().num_boxes
    else:
        # not called from Qt gui
        _maximize_window()
        plt.tight_layout()
        plt.show(block=True)
        num_boxes = float(input("number of markers: "))

    # I built the input to this in a bad way. The nominal value is the size of
    # an engineering paper box, and the std_dev is the resolution error of a
    # single line. The error should be applied at either end of the calibration
    # line, i.e. the error should be the same regardless of line length. To
    # make this happen, I am breaking out the components and applying them as
    # originally intended.
    line_length_mm = num_boxes * marker_length_mm
    if apply_uncertainty:
        line_length_mm = un.ufloat(
            line_length_mm,
            add_uncertainty_terms([
                u_cell["l_mm"]["b"],
                u_cell["l_mm"]["p"]
            ])
        )

    if px_only:
        return _get_cal_delta_px(linebuilder.xs, linebuilder.ys)
    else:
        mm_per_px = _calibrate(
            linebuilder.xs,
            linebuilder.ys,
            line_length_mm,
            apply_uncertainty=apply_uncertainty
        )

    if save_output:
        _save_spatial_calibration(
            mm_per_px=mm_per_px,
            spatial_file_path=spatial_file
        )

    return mm_per_px


def measure_single_frame(
        image_array,
        lc="r"
):
    m = MeasurementCollector(image_array, lc=lc)
    _maximize_window()
    data = m.get_data()
    del m
    return data


def _get_cal_delta_px(
        x_data,
        y_data
):
    return np.sqrt(
            np.square(np.diff(x_data)) +
            np.square(np.diff(y_data))
    )


def _calibrate(
        x_data,
        y_data,
        line_length_mm,
        apply_uncertainty=True
):
    """
    Calculates a calibration factor to convert pixels to mm by dividing
    the known line length in mm by the L2 norm between two pixels.

    Parameters
    ----------
    x_data : iterable
        X locations of two points
    y_data : iterable
        Y locations of two points
    line_length_mm : float
        Length, in mm, of the line between (x0, y0), (x1, y1)
    apply_uncertainty : bool
        Applies pixel uncertainty if True

    Returns
    -------
    float or un.ufloat
        Pixel linear pitch in mm/px
    """
    line_length_px = _get_cal_delta_px(x_data, y_data)

    if apply_uncertainty:
        line_length_px = un.ufloat(
            line_length_px,
            add_uncertainty_terms([
                u_cell["l_px"]["b"],
                u_cell["l_px"]["p"]
            ])
        )

    return line_length_mm / line_length_px


def _save_spatial_calibration(
        mm_per_px,
        spatial_file_path
):
    fname_saved_calibration = "spatial calibration.json"
    dir_spatial, fname_spatial_image = os.path.split(spatial_file_path)
    path_out = os.path.join(
        dir_spatial,
        fname_saved_calibration
    )
    save_statement = "Saved to {:s}!".format(path_out)

    current_info = {
            "in/px": mm_per_px,
            "cal date": str(datetime.now())
    }

    if not os.path.exists(path_out):
        with open(path_out, "w"):
            pass

    with open(path_out, "r+") as f:
        try:
            data = json.load(f)
            if fname_spatial_image in data.keys():
                user_overwrite = 'asdf'
                while user_overwrite.lower() not in {'y', 'n'}:
                    user_overwrite = input(
                        "Data already exists for {:s}. "
                        "Overwrite (y/n)? ".format(fname_spatial_image))
                if user_overwrite == "y":
                    data[fname_spatial_image] = current_info
                    f.seek(0)
                    json.dump(data, f)
                    f.truncate()
                    print(save_statement)
                else:
                    print("Calibration data not overwritten.")

            else:
                data[fname_spatial_image] = current_info
                f.seek(0)
                json.dump(data, f)
                f.truncate()
                print(save_statement)

        except json.decoder.JSONDecodeError:
            data = {
                fname_spatial_image: current_info
            }
            json.dump(data, f)
            print(save_statement)


class LineBuilder(object):  # pragma: no cover
    # I'm not sure how to automate tests on this, it works right now, and I
    # don't have time to figure out how, so I'm going to skip it for now.
    # modified version of code from
    # https://stackoverflow.com/questions/34855074/interactive-line-in-matplotlib
    def __init__(self, line, epsilon=10):
        canvas = line.figure.canvas
        line.set_alpha(0.7)
        self.canvas = canvas
        self.canvas.mpl_connect("key_press_event", self._button)
        self.line = line
        self.axes = line.axes
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.background = None
        self.epsilon = epsilon
        self.circles = [
            widgets.Circle(
                (self.xs[i], self.ys[i]),
                epsilon,
                color=line.get_c(),
                lw=line.get_linewidth(),
                fill=False,
                alpha=0.25
            )
            for i in range(len(self.xs))
        ]
        for c in self.circles:
            self.axes.add_artist(c)

        self._end_line_length = 2 * np.sqrt(
            sum([
                np.diff(self.axes.get_xlim())**2,
                np.diff(self.axes.get_ylim())**2
            ])
        )
        self._end_lines = [
            widgets.Line2D(
                [0, 1],
                [0, 1],
                c=line.get_c(),
                lw=line.get_linewidth(),
                alpha=0.5*line.get_alpha()
            )
            for _ in self.xs
        ]
        self.set_end_lines()
        for _line in self._end_lines:
            self.axes.add_artist(_line)

        self.items = (self.line, *self.circles, *self._end_lines)

        self.ind = None
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

    def _button(self, event):
        if event.key == "enter":
            plt.close(self.line.figure)

    def get_ind(self, event):
        if event.inaxes is not None:
            x = np.array(self.line.get_xdata())
            y = np.array(self.line.get_ydata())
            d = np.sqrt((x-event.xdata)**2 + (y - event.ydata)**2)
            if min(d) > self.epsilon:
                return None
            return int(d[0] > d[1])

    def button_press_callback(self, event):
        if event.button == 2:
            # middle click
            plt.close(self.axes.get_figure())
        elif event.button != 1:
            return
        self.ind = self.get_ind(event)

        for item in self.items:
            item.set_animated(True)

        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.line.axes.bbox)

        for item in self.items:
            self.axes.draw_artist(item)

        self.canvas.blit(self.axes.bbox)

    def button_release_callback(self, event):
        if event.button != 1:
            return
        self.ind = None
        for item in self.items:
            item.set_animated(False)

        self.background = None

        for item in self.items:
            item.figure.canvas.draw()

    def motion_notify_callback(self, event):
        if event.inaxes != self.line.axes:
            return
        if event.button != 1:
            return
        if self.ind is None:
            return
        self.xs[self.ind] = event.xdata
        self.ys[self.ind] = event.ydata
        self.line.set_data(self.xs, self.ys)
        self.set_end_lines()
        for c, x, y in zip(self.circles, self.xs, self.ys):
            # noinspection PyArgumentList
            c.set_center((x, y))

        self.canvas.restore_region(self.background)

        for item in self.items:
            self.axes.draw_artist(item)

        self.canvas.blit(self.axes.bbox)

    def get_line_angle(self):
        return np.arctan(np.diff(self.ys) / np.diff(self.xs))[0]

    def calculate_end_line_xy(self):
        angle = (self.get_line_angle() + np.pi / 2) % (2 * np.pi)
        dx = self._end_line_length / 2 * np.sqrt(1 / (1 + np.tan(angle)**2))
        dy = dx * np.tan(angle)
        x_points = [list(x + np.array([1, -1]) * dx) for x in self.xs]
        y_points = [list(y + np.array([1, -1]) * dy) for y in self.ys]
        return [x_points, y_points]

    def set_end_lines(self):
        end_line_points = self.calculate_end_line_xy()
        for _line, x, y in zip(self._end_lines, *end_line_points):
            _line.set_data(x, y)


class MeasurementCollector(object):  # pragma: no cover
    # also skipping tests for the same reason as LineBuilder
    class RemoveLine:
        button = 3

    class CloseIt:
        button = 2

    def __init__(self, image, lc="r"):
        self.locs = []
        self.cmap = "gray"
        fig, [ax, ax2] = plt.subplots(2, 1)
        self.lines = []
        self.fig = fig
        plt.get_current_fig_manager().window.setGeometry(0, 0, 640, 480)
        self.ax = ax
        self.lc = lc
        # remove_annotations(ax)
        ax.set_axis_off()
        ax.set_position([0, 0.07, 1, 0.9])
        ax2.set_position([0.375, 0.01, 0.25, 0.05])
        # plt.axis("off")
        # plt.axis("tight")
        self._help = False
        self._title_default = "press 'h' for help"
        self._title_help = \
            "HELP MENU\n\n"\
            "press 'r' to invert colors\n"\
            "press left mouse to identify a triple point\n"\
            "press right mouse to delete last measurement\n"\
            "press 'enter' or center mouse to end measurements\n"\
            "click and drag horizontally to adjust contrast to red area\n"\
            "click 'Reset Contrast' button to reset contrast\n"\
            "press 'h' to hide this dialog"
        self._set_title(self._title_default)
        canvas = ax.figure.canvas
        canvas.mpl_connect("key_press_event", self._button)
        canvas.mpl_connect('button_release_event', self.button_press_callback)
        self.image = self._sharpen(image)
        self.rect_select = widgets.SpanSelector(
            self.ax,
            self.slider_select,
            "horizontal"
        )
        # noinspection PyTypeChecker
        # ax2 = plt.axes((0.375, 0.025, 0.25, 0.04))
        # fig.add_axes(ax2)
        self.btn_reset = widgets.Button(
            ax2,
            "Reset Contrast"
        )
        self.btn_reset.on_clicked(self.reset_vlim)
        self.ax.imshow(self.image, cmap=self.cmap)
        self.fig.canvas.draw()

    @staticmethod
    def _sharpen(image):
        image /= image.max()
        filtered = 1 - sobel_v(image)
        filtered /= filtered.max()
        return filtered * image

    def _button(self, event):
        if event.key == "enter":
            self.button_press_callback(self.CloseIt)
        elif event.key == "r":
            if self.cmap == "gray":
                self.cmap = "gist_gray_r"
            else:
                self.cmap = "gray"
            self.ax.images[0].set_cmap(self.cmap)
            self.fig.canvas.draw()
        elif event.key == "h":
            if self._help:
                self._set_title(self._title_help, True)
            else:
                self._set_title(self._title_default)
            self._help = not self._help
            self.fig.canvas.draw()

    def _set_title(self, string, have_background=False):
        if have_background:
            bg_color = (1, 1, 1, 0.75)
            h_align = "left"
        else:
            bg_color = (0, 0, 0, 0)
            h_align = "right"
        t = self.fig.suptitle(
            string,
            size=10,
            y=0.99,
            ma=h_align,
        )
        t.set_backgroundcolor(bg_color)
        self.fig.canvas.draw()

    def slider_select(self, x_min, x_max):
        px_distance = abs(x_max - x_min)

        if px_distance <= 1:
            # this should have been a click
            pass
        else:
            # this was meant to be a drag
            x_min, x_max = int(x_min), int(x_max)
            img_in_range = self.image[:, x_min:x_max]
            self.ax.images[0].norm.vmin = np.min(img_in_range)
            self.ax.images[0].norm.vmax = np.max(img_in_range)
            self.fig.canvas.draw()
            self.button_press_callback(self.RemoveLine)

    def reset_vlim(self, _):
        self.ax.images[0].norm.vmin = np.min(self.image)
        self.ax.images[0].norm.vmax = np.max(self.image)
        self.button_press_callback(self.RemoveLine)
        self.fig.canvas.draw()

    def button_press_callback(self, event):
        if event.button == 1:
            # left click
            if any([d is None for d in [event.xdata, event.ydata]]):
                # ignore clicks outside of image
                pass
            else:
                self.lines.append(self.ax.axhline(event.ydata, color=self.lc))
                self.locs.append(event.ydata)
                self.fig.canvas.draw()

        elif event.button == 2:
            # middle click
            plt.close()
        elif event.button == 3:
            # right click
            if self.lines:
                # noinspection PyProtectedMember
                self.lines[-1]._visible = False
                del self.lines[-1], self.locs[-1]
                self.fig.canvas.draw()

    def get_data(self):
        plt.show(block=True)
        points = unp.uarray(
            sorted(np.array(self.locs)),
            add_uncertainty_terms([
                u_cell["delta_px"]["b"],
                u_cell["delta_px"]["p"]
            ])
        )
        return points


def get_cell_size_from_delta(
        delta,
        l_px_i,
        l_mm_i
):
    """
    Converts pixel triple point deltas to cell size

    Parameters
    ----------
    delta : un.ufloat
    l_px_i : float
        nominal value of spatial calibration factor (px)
    l_mm_i : float
        nominal value of spatial calibration factor (mm)

    Returns
    -------
    un.ufloat
        estimated cell size
    """
    l_px_i = un.ufloat(
        l_px_i,
        add_uncertainty_terms([
            u_cell["l_px"]["b"],
            u_cell["l_px"]["p"]
        ])
    )
    l_mm_i = un.ufloat(
        l_mm_i,
        add_uncertainty_terms([
            u_cell["l_mm"]["b"],
            u_cell["l_mm"]["p"]
        ])
    )
    return 2 * delta * l_mm_i / l_px_i


def _filter_df_day_shot(
        df,
        day_shot_list,
        return_mask=False
):
    """
    Filters a dataframe by date and shot number for an arbitrary number of
    date/shot combinations. Returns the indices (for masking) and the filtered
    dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to filter. Must have columns for "date" and "shot".
    day_shot_list : List[Tuple[Str, Int, Int]]
        List of tuples containing date, start shot, and end shot. Date should
        be a string in ISO-8601 format, and start/end shots numbers should be
        integers:
        [("YYYY-MM-DD", start_shot, end_shot)]
    return_mask : bool
        if true, mask will be returned as the second item, which can be used to
        update data (e.g. inserting a spatial calibration)

    Returns
    -------
    Union[Tuple[pd.DataFrame, np.array], Tuple[pd.DataFrame]]
        (filtered dataframe,) or (filtered dataframe, mask)
    """
    mask_list = [((df["date"] == date) &
                  (df["shot"] <= end_shot) &
                  (df["shot"] >= start_shot))
                 for (date, start_shot, end_shot) in day_shot_list]
    mask = [False for _ in range(len(df))]
    for m in mask_list:
        mask = m | mask
    if return_mask:
        return df[mask], mask
    else:
        return df[mask],


def _check_stored_calibrations(
        df
):
    """
    Check for stored calibrations within a filtered dataframe. All rows are
    checked for:
        * whether there are any stored spatial calibrations
        * whether there are stored calibrations for every date and shot
        * whether all of the stored calibrations are equal

    This function is meant to be applied to a schlieren dataframe, which must
    contain the columns:
        * spatial_near
        * spatial_far
        * spatial_centerline

    Parameters
    ----------
    df : pd.DataFrame
        filtered dataframe containing only the date/shot combinations of
        interest

    Returns
    -------
    Dict[String: Dict[String: Bool]]
        Outer keys:
            * near
            * far
            * centerline
        Inner keys:
            * any
            * all
            * equal
    """
    out = dict(
        near=dict(
            any=False,
            all=False,
            equal=False,
        ),
        far=dict(
            any=False,
            all=False,
            equal=False,
        ),
        centerline=dict(
            any=False,
            all=False,
            equal=False,
        ),
    )

    for location in out.keys():
        values = df["spatial_" + location].values.astype(float)
        not_nan = ~np.isnan(values)
        out[location]["any"] = np.any(not_nan)
        out[location]["all"] = np.all(not_nan)

        if len(values[not_nan]) == 0:
            # everything is NaN
            out[location]["equal"] = True
        else:
            # allclose will cause nanmedian check to fail for NaN as well as
            # for differing numerical values
            out[location]["equal"] = np.allclose(
                values,
                np.nanmedian(values)
            )

    return out


class SpatialCalibration:
    @staticmethod
    def collect(
            date,
            loc_processed_data,
            loc_schlieren_measurements,
            raise_if_no_measurements=True
    ):
        with pd.HDFStore(loc_processed_data, "r") as store_pp:
            # make sure date is in post-processed data
            if date not in store_pp.data["date"].unique():
                e_str = "date {:s} not in {:s}".format(
                    date,
                    loc_processed_data
                )
                raise ValueError(e_str)
            else:
                df_dirs = store_pp.data[
                    store_pp.data["date"] == date
                ][["shot", "spatial"]]
                df_dirs.columns = ["shot", "dir"]
                df_dirs["dir"] = df_dirs["dir"].apply(
                    convert_dir_to_local
                )

        with pd.HDFStore(loc_schlieren_measurements, "r+") as store_sc:
            df_sc = store_sc.data[
                store_sc.data["date"] == date
            ]
            if len(df_sc) == 0 and raise_if_no_measurements:
                e_str = "no measurements found for %s" % date
                raise ValueError(e_str)

            # collect calibrations
            df_daily_cal = pd.DataFrame([dict(
                dir=k,
                near=un.ufloat(np.NaN, np.NaN),
                far=un.ufloat(np.NaN, np.NaN),
            ) for k in df_dirs["dir"].unique()]).set_index("dir")
            desired_cals = ["near", "far"]
            successful_cals = []
            for d, row in df_daily_cal.iterrows():
                for which in desired_cals:
                    pth_tif = os.path.join(str(d), which + ".tif")
                    if os.path.exists(pth_tif):
                        df_daily_cal.at[
                            d,
                            which
                        ] = collect_spatial_calibration(pth_tif)
                        successful_cals.append(which)

            # apply calibrations
            for _, row in df_dirs.iterrows():
                row_mask = df_sc["shot"] == row["shot"]
                # set near and far spatial calibrations
                for which in successful_cals:
                    key = "spatial_" + which
                    df_sc[key] = np.where(
                        row_mask,
                        df_daily_cal.loc[row["dir"], which].nominal_value,
                        df_sc[key]
                    )
                    key = "u_" + key
                    df_sc[key] = np.where(
                        row_mask,
                        df_daily_cal.loc[row["dir"], which].std_dev,
                        df_sc[key]
                    )
                    df_sc["spatial_" + which + "_estimated"] = False

                # calculate and set centerline calibration
                centerline = np.mean(
                    [unp.uarray(df_sc["spatial_near"], df_sc["u_spatial_near"]),
                     unp.uarray(df_sc["spatial_far"], df_sc["u_spatial_far"])],
                    axis=0
                )
                df_sc["spatial_centerline"] = np.where(
                    row_mask,
                    unp.nominal_values(centerline),
                    df_sc["spatial_centerline"]
                )
                df_sc["u_spatial_centerline"] = np.where(
                    row_mask,
                    unp.std_devs(centerline),
                    df_sc["u_spatial_centerline"]
                )

            df_out = store_sc.data
            df_out.loc[df_out["date"] == date] = df_sc
            store_sc.put("data", df_out)

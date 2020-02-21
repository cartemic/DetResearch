# stdlib imports
import os
from datetime import datetime

# third party imports
import json
import numpy as np
import uncertainties as un
from matplotlib import pyplot as plt
from matplotlib import widgets
from skimage import io

# local imports
from .uncertainty import engr_line_thk as u_engr
u_engr = u_engr.nominal_value


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


def collect_shot_images(
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
    pth_list_bg, pth_list_frames = collect_shot_images(dir_raw_shot)
    bg = average_frames(pth_list_bg)
    return [(io.imread(frame) - bg + 2**15) for frame in pth_list_frames]


def remove_annotations(ax):  # pragma: no cover
    """
    Hide plot annotations

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot

    Returns
    -------
    None
    """
    ax.xaxis._visible = False
    ax.yaxis._visible = False
    for s in ax.spines:
        # noinspection PyProtectedMember
        ax.spines[s]._visible = False


def spatial_calibration(
        spatial_file,
        line_color="r",
        cmap="viridis",
        marker_length_inches=un.ufloat(0.2, u_engr/2),
        save_output=False,
        px_only=False
):  # pragma: no cover
    image = io.imread(spatial_file)
    fig, ax = plt.subplots(1, 1)

    ax.imshow(image, cmap=cmap)
    cal_line = widgets.Line2D(
        [0, 100],
        [0, 100],
        c=line_color
    )
    ax.add_line(cal_line)

    # noinspection PyTypeChecker
    linebuilder = LineBuilder(cal_line)

    # https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    plt_backend = plt.get_backend()
    mng = plt.get_current_fig_manager()
    if "Qt" in plt_backend:
        mng.window.showMaximized()
    elif "wx" in plt_backend:
        mng.frame.Maximize(True)
    elif "Tk" in plt_backend:
        mng.window_state('zoomed')
    else:
        print("figure out how to maximize for ", plt_backend)

    remove_annotations(ax)
    plt.tight_layout()
    plt.show(block=True)
    num_boxes = float(input("number of markers: "))

    # I built the input to this in a bad way. The nominal value is the size of
    # an engineering paper box, and the std_dev is the resolution error of a
    # single line. The error should be applied at either end of the calibration
    # line, i.e. the error should be the same regardless of line length. To
    # make this happen, I am breaking out the components and applying them as
    # originally intended.
    line_length_inches = un.ufloat(
        num_boxes * marker_length_inches.nominal_value,
        marker_length_inches.std_dev
    )

    # line_length_inches = line_length_px * marker_length_inches

    if px_only:
        # pixels only
        return np.linalg.norm([linebuilder.xs, linebuilder.ys], ord=2)
    else:
        inches_per_pixel = _calibrate(
            linebuilder.xs,
            linebuilder.ys,
            line_length_inches
        )

    if save_output:
        _save_spatial_calibration(
            inches_per_pixel=inches_per_pixel,
            spatial_file_path=spatial_file
        )

    return inches_per_pixel


def _calibrate(
        x_data,
        y_data,
        line_length_inches
):
    """
    Calculates a calibration factor to convert pixels to inches by dividing
    the known line length in inches by the L2 norm between two pixels.

    Parameters
    ----------
    x_data : iterable
        X locations of two points
    y_data : iterable
        Y locations of two points
    line_length_inches : float
        Length, in inches, of the line between (x0, y0), (x1, y1)

    Returns
    -------
    float
        Pixel linear pitch in in/px
    """
    line_length_px = un.ufloat(
        np.linalg.norm([x_data, y_data], ord=2),
        0.5  # +/- 1/2 pixel
    )

    return line_length_inches/line_length_px


def _save_spatial_calibration(
        inches_per_pixel,
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
            "in/px": inches_per_pixel,
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
        if event.button != 1:
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

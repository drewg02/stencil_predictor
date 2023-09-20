import os

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from enum import Enum


def makedirs(file):
    output_directory = os.path.dirname(file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


def create_uniform_random_array(rows, cols):
    arr = np.random.uniform(low=0, high=1, size=(rows, cols), dtype=float)
    return arr, np.copy(arr)

def create_test_array(rows, cols):
    arr = np.zeros((rows, cols), dtype=float)

    arr[:, 0] = 1
    arr[:, -1] = 1

    return arr, np.copy(arr)

class ArrayType(Enum):
    OUTLINE = 'outline'
    CENTER = 'center'
    PLUS = 'plus'

def update_array(array, rows, cols, array_type, thickness):
    match array_type:
        case ArrayType.OUTLINE:
            array[0:thickness, :] = 1
            array[-thickness:, :] = 1
            array[:, 0:thickness] = 1
            array[:, -thickness:] = 1
        case ArrayType.CENTER:
            array[rows // 2 - thickness:rows // 2 + thickness, cols // 2 - thickness:cols // 2 + thickness] = 1
        case ArrayType.PLUS:
            array[rows // 2 - thickness:rows // 2 + thickness, :] = 1
            array[:, cols // 2 - thickness:cols // 2 + thickness] = 1

def create_array(rows, cols, array_type, thickness=1, input_array=None, input_mask=None):
    arr = np.zeros((rows, cols), dtype=float) if input_array is None else input_array
    mask = np.zeros((rows, cols), dtype=float) if input_mask is None else input_mask

    for array in [arr, mask]:
        update_array(array, rows, cols, array_type, thickness)

    return arr, mask

def print_array(arr):
    df = pd.DataFrame(arr)
    print(df.to_string(index=None, header=False, float_format="%.10f"))

def save_array_to_file(arr, output_file):
    makedirs(output_file)

    np.save(output_file, arr, allow_pickle=False)

def read_array_from_file(input_file):
    return np.load(input_file, allow_pickle=False)

def max_diff(arr1, arr2):
    diff = np.abs(arr1 - arr2)
    return np.max(diff)

def avg_diff(arr1, arr2):
    diff = np.abs(arr1 - arr2)
    return np.average(diff)

def apply_stencil(arr, mask, iterations, save_history=True, max_diff_threshold=1e-10, avg_diff_threshold=1e-10):
    plate_history = np.copy(arr)
    new_array = np.copy(arr)

    max_diffs = np.empty(iterations, dtype=float)
    max_diffs.fill(np.nan)
    avg_diffs = np.empty(iterations, dtype=float)
    avg_diffs.fill(np.nan)
    for f in range(iterations):
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if mask[i][j] == 1:
                    continue

                total = 0
                count = 0

                for x in range(i - 1, i + 2):
                    for y in range(j - 1, j + 2):
                        if 0 <= x < len(arr) and 0 <= y < len(arr[i]):
                            total += arr[x][y]
                            count += 1

                new_array[i][j] = total / count

        arr, new_array = new_array, arr
        if save_history:
            plate_history = np.append(plate_history, np.copy(arr))

        max_diff_value = max_diff(arr, new_array)
        max_diff_delta = np.abs(max_diff_value - max_diffs[f - 1])
        if max_diff_threshold > 0 and max_diff_delta < max_diff_threshold:
            print('Stopped by max diff threshold of ' + str(max_diff_threshold) + ' at iteration ' + str(f))
            break

        max_diffs[f] = max_diff_value

        avg_diff_value = avg_diff(arr, new_array)
        avg_diff_delta = np.abs(avg_diff_value - avg_diffs[f - 1])
        if avg_diff_threshold > 0 and avg_diff_delta < avg_diff_threshold:
            print('Stopped by avg diff threshold of ' + str(avg_diff_threshold) + ' at iteration ' + str(f))
            break

        avg_diffs[f] = avg_diff_value


    if save_history:
        plate_history = np.reshape(plate_history, (-1, arr.shape[0], arr.shape[1]))

    return arr, plate_history, max_diffs, avg_diffs

def plot_diffs(max_diffs, max_squared_diffs, output_filename):
    makedirs(output_filename)

    plt.figure(figsize=(10, 5))
    plt.plot(max_diffs, label='Max Diff')
    plt.plot(max_squared_diffs, label='Average Diff')

    plt.xlabel('Iteration')
    plt.ylabel('Diff')
    plt.title('Max Diff and Average Diff Over Time')
    plt.legend()

    plt.savefig(output_filename)

def save_plate_history_to_file(plate_history, all_iterations):
    makedirs(all_iterations)

    np.save(all_iterations, plate_history)

def save_array_as_image(arr, output_file, graph_format=False, cmap='coolwarm'):
    makedirs(output_file)

    if graph_format:
        plt.imshow(arr, cmap='coolwarm', origin='upper')

        plt.xticks(np.arange(0, arr.shape[1], 10))
        plt.yticks(np.arange(0, arr.shape[0], 5))

        plt.tick_params(which='both', labelbottom=True, labeltop=True, top=True, labelright=True, right=True, labelsize=10)  # X-axis tick params

        if "." not in output_file:
            output_file = output_file + ".png"

        plt.savefig(output_file)
    else:
        fig = plt.figure(figsize=(arr.shape[1] / 100, arr.shape[0] / 100))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(arr, cmap=cmap, origin='upper')
        if "." not in output_file:
            output_file = output_file + ".png"
        fig.savefig(output_file, dpi=500)
        plt.close(fig)

def save_diff_as_image(arr1, arr2, output_file, graph_format=False):
    makedirs(output_file)

    arr = abs(arr1 - arr2)
    save_array_as_image(arr, output_file, graph_format=graph_format, cmap='gray')

def save_array_as_movie(arr, output_file, cmap="coolwarm"):
    makedirs(output_file)

    dpi = 200
    fps = 10

    base = plt.figure(figsize=(arr.shape[1] / 10, arr.shape[2] / 10), dpi=dpi)
    ax = plt.Axes(base, [0, 0, 1, 1])
    ax.set_axis_off()
    base.add_axes(ax)
    base.show()

    def animate(i):
        if arr.shape[0] < 50 or i % 10 == 0: print(i)
        return [ax.imshow(arr[i], cmap=cmap, origin='lower')]

    im_ani = animation.FuncAnimation(base, animate, frames=arr.shape[0] - 1, interval=1000 / fps, blit=True)
    im_ani.save(output_file, dpi=dpi)

class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format_usage(self):
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)
        usage = formatter.format_help()
        for action in self._actions:
            if len(action.option_strings) < 1:
                usage = usage.replace(f" {action.dest}", f" <{action.dest}>")

        return usage
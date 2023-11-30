import argparse
import time
from enum import Enum

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt, cm


def create_test_array(rows, cols):
    arr = np.zeros((rows, cols), dtype=float)

    arr[:, 0] = 1
    arr[:, -1] = 1

    return arr, np.copy(arr)


class ArrayType(Enum):
    OUTLINE = 'outline'
    CENTER = 'center'
    PLUS = 'plus'
    RANDOM = 'random'
    RANDOM_UNIFORM = 'random_uniform'


def update_array(array, rows, cols, array_type, thickness=1, chance=0.2):
    match array_type:
        case ArrayType.OUTLINE:
            array[0:thickness, :] = 1.0
            array[-thickness:, :] = 1.0
            array[:, 0:thickness] = 1.0
            array[:, -thickness:] = 1.0
        case ArrayType.CENTER:
            array[rows // 2 - thickness:rows // 2 + thickness, cols // 2 - thickness:cols // 2 + thickness] = 1.0
        case ArrayType.PLUS:
            array[rows // 2 - thickness:rows // 2 + thickness, :] = 1.0
            array[:, cols // 2 - thickness:cols // 2 + thickness] = 1.0
        case ArrayType.RANDOM:
            array[:] = np.random.choice([0.0, 1.0], size=(rows, cols), p=[1.0 - chance, chance])
        case ArrayType.RANDOM_UNIFORM:
            array[:] = np.random.uniform(low=0.0, high=1.0, size=(rows, cols))
        case _:
            raise ValueError(f"Invalid array type: {array_type}")

    return array


def create_array(rows, cols, array_type, thickness=1, chance=0.2, input_array=None, input_mask=None, mask_clone=True):
    arr = np.zeros((rows, cols), dtype=float) if input_array is None else input_array
    mask = np.zeros((rows, cols), dtype=float) if input_mask is None else input_mask

    try:
        arr = update_array(arr, rows, cols, array_type, thickness=thickness, chance=chance)
    except ValueError as e:
        print(f"Couldn't create the array: {e}")
        return None, None

    if mask_clone:
        mask = np.copy(arr)
    else:
        try:
            mask = update_array(mask, rows, cols, array_type, thickness=thickness, chance=chance)
        except ValueError as e:
            print(f"Couldn't create the mask: {e}")
            return None, None

    return arr, mask


def create_pairs(rows, cols, num_initials, num_iterations, array_type=ArrayType.RANDOM, thickness=1, chance=0.2, verbose=True):
    pairs = np.empty((num_initials * num_iterations, 2, rows, cols), dtype=float)
    last_progress = 0

    total_pairs = num_initials * num_iterations

    start_time = time.time()
    for i in range(num_initials):
        total_index = (i + 1) * num_iterations

        # Generate the input and mask arrays
        input_array, mask_array = create_array(rows, cols, array_type, thickness=thickness, chance=chance)

        # Run the stencil filter on the input array for 1 iteration
        _, plate_history, _, _ = apply_stencil(input_array, mask_array, num_iterations, verbose=False)

        # Save the input and output arrays to the pairs array
        for j in range(num_iterations):
            pairs[i * num_iterations + j][0] = plate_history[j]
            pairs[i * num_iterations + j][1] = plate_history[j + 1]

        progress = total_index / total_pairs * 100
        if progress - last_progress >= 1:
            last_progress = progress
            elapsed_time = time.time() - start_time
            if verbose: print(f"{progress:.2f}% done, generated {total_index}/{total_pairs} pairs, {elapsed_time:.2f} seconds elapsed")

    return pairs


def print_array(arr):
    df = pd.DataFrame(arr)
    print(df.to_string(index=False, header=False, float_format="%.10f"))


def save_array_to_file(arr, output_file):
    try:
        np.save(output_file, arr, allow_pickle=False)
        return True, None
    except Exception as e:
        return False, f"Couldn't save the array to {output_file}: {e}"


def read_array_from_file(input_file):
    try:
        return np.load(input_file, allow_pickle=False)
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        return None
    except Exception as e:
        print(f"Couldn't load the array from {input_file}: {e}")
        return None


def max_diff(arr1, arr2):
    diff = np.abs(arr1 - arr2)
    return np.max(diff)


def avg_diff(arr1, arr2):
    diff = np.abs(arr1 - arr2)
    return np.average(diff)


def apply_stencil(arr, mask, iterations, save_history=True, max_diff_threshold=None, avg_diff_threshold=None, verbose=True):
    start_time = time.time()

    plate_history = np.copy(arr)
    new_array = np.copy(arr)

    max_diffs = None
    avg_diffs = None
    if max_diff_threshold is not None:
        max_diffs = np.empty(iterations, dtype=float)
        max_diffs.fill(np.nan)
    if avg_diff_threshold is not None:
        avg_diffs = np.empty(iterations, dtype=float)
        avg_diffs.fill(np.nan)

    last_progress = 0
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

        if max_diff_threshold is not None:
            max_diff_value = max_diff(arr, new_array)
            max_diff_delta = np.abs(max_diff_value - max_diffs[f - 1])
            if max_diff_threshold > 0 and max_diff_delta < max_diff_threshold:
                if verbose: print(f'Stopped by max diff threshold of {max_diff_threshold} at iteration {f}')
                break

            max_diffs[f] = max_diff_value

        if max_diff_threshold is not None:
            avg_diff_value = avg_diff(arr, new_array)
            avg_diff_delta = np.abs(avg_diff_value - avg_diffs[f - 1])
            if avg_diff_threshold > 0 and avg_diff_delta < avg_diff_threshold:
                if verbose: print(f'Stopped by avg diff threshold of {avg_diff_threshold} at iteration {f}')
                break

            avg_diffs[f] = avg_diff_value

        progress = ((f + 1) / iterations) * 100
        if progress - last_progress >= 1:
            last_progress = progress
            elapsed_time = time.time() - start_time
            if verbose: print(f"{progress:.2f}% done, iteration {f + 1}/{iterations}, {elapsed_time:.2f} seconds elapsed")

    if save_history:
        plate_history = np.reshape(plate_history, (-1, arr.shape[0], arr.shape[1]))

    return arr, plate_history, max_diffs, avg_diffs


def plot_diffs(max_diffs, max_squared_diffs, output_filename):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(max_diffs, label='Max Diff')
        plt.plot(max_squared_diffs, label='Average Diff')

        plt.xlabel('Iteration')
        plt.ylabel('Diff')
        plt.title('Max Diff and Average Diff Over Time')
        plt.legend()

        plt.savefig(output_filename)

        plt.close()

        return True
    except Exception as e:
        print(f"Couldn't save the plot: {e}")
        return False


def save_array_as_image(arr, output_file, cmap="coolwarm", dpi=5000):
    try:
        # if the shape of array is less than 4, then we need to add dimensions to it until it is 4
        while len(arr.shape) < 4:
            arr = np.expand_dims(arr, axis=0)

        cmap_func = cm.get_cmap(cmap)

        groups = []
        for n in range(arr.shape[0]):
            group = []
            for m in range(arr.shape[1]):
                color_mapped = cmap_func(arr[n][m])
                img_bgr = (color_mapped * 255).astype(np.uint8)

                group.append(img_bgr)

            if len(group) > 1:
                group = np.vstack(group)
            else:
                group = group[0]

            groups.append(group)

        if len(groups) > 1:
            final_img = np.hstack(groups)
        else:
            final_img = groups[0]

        final_img_pil = Image.fromarray(final_img)
        final_img_pil.save(output_file, dpi=(dpi, dpi))

        return True
    except Exception as e:
        print(f"Couldn't save the image: {e}")
        return False


def save_diff_as_image(arr1, arr2, output_file):
    arr = abs(arr1 - arr2)
    return save_array_as_image(arr, output_file, cmap='gray')


def save_array_as_movie(arr, output_file, cmap="coolwarm", dpi=200, fps=10, fourcc_str='avc1'):
    start_time = time.time()
    try:
        height, width = arr.shape[1], arr.shape[2]
        scale_factor = dpi // 10

        height_scaled, width_scaled = height * scale_factor, width * scale_factor

        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(output_file, fourcc, fps, (width_scaled, height_scaled), isColor=True)

        cmap_func = cm.get_cmap(cmap)

        total_frames = arr.shape[0]
        last_progress = 0
        for i in range(total_frames):
            frame = arr[i]
            normalized_frame = frame / frame.max()
            rgba_frame = (cmap_func(normalized_frame) * 255).astype(np.uint8)
            bgr_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGR)

            bgr_frame_scaled = cv2.resize(bgr_frame, (width_scaled, height_scaled), interpolation=cv2.INTER_NEAREST)

            out.write(bgr_frame_scaled)

            progress = (i / total_frames) * 100
            if progress - last_progress >= 1:
                last_progress = progress
                elapsed_time = time.time() - start_time
                print(f"{progress:.2f}% done, frame {i} of {total_frames}, {elapsed_time:.2f} seconds elapsed")

        out.release()

        return True
    except Exception as e:
        print(f"Couldn't save the movie: {e}")
        return False


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


def MASE(target_images, predicted_images):
    mae = np.mean(np.abs(target_images - predicted_images))
    scale = np.mean(np.abs(target_images[1:] - target_images[:-1]))
    return mae / scale


def sMAPE(target_images, predicted_images):
    return 100 * np.mean(2 * np.abs(predicted_images - target_images) / (np.abs(predicted_images) + np.abs(target_images)))

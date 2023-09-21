import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import pygame
import matplotlib.pyplot as plt
from PIL import Image
import utilities as ut


def update_pixel(new_x, new_y, array, mask, mouse_button):
    # Only update the pixel if it is in bounds
    if 0 <= new_x < array.shape[1] and 0 <= new_y < array.shape[0]:
        # Update the pixel based on the mouse button, 1 is left click, 3 is right click
        if mouse_button == 1:
            array[new_y, new_x] = 1
            mask[new_y, new_x] = 1
        elif mouse_button == 3:
            array[new_y, new_x] = 0
            mask[new_y, new_x] = 0

    return array, mask


def render(array, screen, width, height):
    # Create the plot to render
    fig = plt.figure(figsize=(array.shape[1] / 100, array.shape[0] / 100))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(array, cmap='coolwarm', origin='upper')

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # Convert the plot to a pygame image to be displayed
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    img = Image.fromarray(buf)
    img = pygame.image.fromstring(img.tobytes(), img.size, "RGBA")

    # Scale the image to fit the screen
    scale_factor = min(width / array.shape[1], height / array.shape[0])
    new_width, new_height = int(array.shape[1] * scale_factor), int(array.shape[0] * scale_factor)
    img = pygame.transform.scale(img, (new_width, new_height))

    # Center the image on the screen
    x_offset = (width - new_width) // 2
    y_offset = (height - new_height) // 2

    # Render the image
    screen.fill((0, 0, 0))
    screen.blit(img, (x_offset, y_offset))
    pygame.display.flip()

    plt.close(fig)

    return x_offset, y_offset, new_width, new_height


# Saves the outputs and exits the program
def save_and_exit(array, mask, parsed_args):
    pygame.quit()

    ut.save_array_to_file(array, parsed_args.output_file)
    ut.save_array_to_file(mask, parsed_args.output_mask_file)
    if parsed_args.output_png_file is not None:
        ut.save_array_as_image(array, parsed_args.output_png_file)

    exit()


# Calculates the position of the mouse in the array
def calc_position(array, width, height, x_offset, y_offset):
    x, y = pygame.mouse.get_pos()
    scale_factor = min(width / array.shape[1], height / array.shape[0])
    x, y = int((x - x_offset) / scale_factor), int((y - y_offset) / scale_factor)

    return x, y


def main(args):
    parser = ut.ArgParser(description='Create a new array using a gui interface.')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('output_mask_file', type=str, help='Path to the output mask file')
    parser.add_argument('--rows', type=int, help='Number of rows for the array')
    parser.add_argument('--cols', type=int, help='Number of columns for the array')
    parser.add_argument('--input_file', type=str, help='File to start from')
    parser.add_argument('--input_mask_file', type=str, help='File to start from')
    parser.add_argument('--output_png_file', type=str, help='Path to the output png file')

    parsed_args = parser.parse_args(args)

    if (parsed_args.rows is None or parsed_args.cols is None) and parsed_args.input_file is None:
        print('You must specify either rows and cols or an input file.')
        return

    if (parsed_args.rows is not None or parsed_args.cols is not None) and parsed_args.input_file is not None:
        print('You cannot specify both rows and cols and an input file.')
        return

    pygame.init()
    width, height = 1000, 1000
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption(f'Draw Generator ({parsed_args.rows}x{parsed_args.cols})')

    # Handles the case where we are starting from an input file
    if parsed_args.input_file is not None:
        array = ut.read_array_from_file(parsed_args.input_file)
        mask = ut.read_array_from_file(parsed_args.input_mask_file)
    else:
        array = np.zeros((parsed_args.rows, parsed_args.cols), dtype=float)
        mask = np.zeros((parsed_args.rows, parsed_args.cols), dtype=float)

    mouse_down = False
    mouse_button = None
    last_x, last_y = None, None
    x_offset, y_offset = 0, 0

    clock = pygame.time.Clock()
    try:
        while True:
            for event in pygame.event.get():
                # Handles when the gui is closed
                if event.type == pygame.QUIT:
                    save_and_exit(array, mask, parsed_args)
                # Handles when the window is resized
                elif event.type == pygame.VIDEORESIZE:
                    width, height = event.w, event.h
                    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                # Handles when the mouse is clicked down
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_down = True
                    mouse_button = event.button

                    x, y = calc_position(array, width, height, x_offset, y_offset)
                    array, mask = update_pixel(x, y, array, mask, mouse_button)
                # Handles when the mouse button is released
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_down = False
                    mouse_button = None
                # Handles when the mouse is moved while the mouse button is down
                if event.type == pygame.MOUSEMOTION and mouse_down:
                    x, y = calc_position(array, width, height, x_offset, y_offset)
                    if x != last_x or y != last_y:
                        array, mask = update_pixel(x, y, array, mask, mouse_button)
                        last_x, last_y = x, y

            x_offset, y_offset, plot_width, plot_height = render(array, screen, width, height)

            clock.tick(120)
    # Handles when the user presses ctrl+c
    except KeyboardInterrupt:
        save_and_exit(array, parsed_args)


if __name__ == "__main__":
    main(sys.argv[1:])

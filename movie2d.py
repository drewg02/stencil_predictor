import sys
import time

import utilities as ut


def main(args):
    parser = ut.ArgParser(description='Read and print an array from a file.')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('--dpi', type=int, help='Frames per second', default=200)
    parser.add_argument('--fps', type=int, help='Frames per second', default=10)
    parser.add_argument('--fourcc', type=str, help='Frames per second', default='avc1')

    parsed_args = parser.parse_args(args)

    # Read the array of images from the input file
    array_to_display = ut.read_array_from_file(parsed_args.input_file)
    if array_to_display is None:
        return

    # Save the array as a movie to the output file
    start_time = time.time()
    ut.save_array_as_movie(array_to_display, parsed_args.output_file, dpi=parsed_args.dpi, fps=parsed_args.fps,
                           fourcc_str=parsed_args.fourcc)

    elapsed = time.time() - start_time
    print(f"Encoding as {parsed_args.fourcc} took {elapsed} seconds.")


if __name__ == "__main__":
    main(sys.argv[1:])

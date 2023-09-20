import sys
import utilities as ut
import time


def main(args):
    parser = ut.ArgParser(description='Read and print an array from a file.')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('output_file', type=str, help='Path to the output file')

    parsed_args = parser.parse_args(args)

    # Read the array of images from the input file
    array_to_display = ut.read_array_from_file(parsed_args.input_file)

    # Save the array as a movie to the output file
    start_time = time.time()
    ut.save_array_as_movie(array_to_display, parsed_args.output_file)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Encoding as mp4 took {elapsed} seconds.")


if __name__ == "__main__":
    main(sys.argv[1:])

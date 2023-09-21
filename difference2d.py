import argparse as ap
import sys

import utilities as ut


def main(args):
    # Argument parsing
    parser = ut.ArgParser(description='Read and print an array from a file.')
    parser.add_argument('input_file_1', type=str, help='Path to the input file')
    parser.add_argument('input_file_2', type=str, help='Path to the input file')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('--graph-format', action=ap.BooleanOptionalAction,
                        help='Make the output display in graph format')

    parsed_args = parser.parse_args(args)

    # Read the two arrays from the input files
    array_to_display_1 = ut.read_array_from_file(parsed_args.input_file_1)
    array_to_display_2 = ut.read_array_from_file(parsed_args.input_file_2)

    # Save their difference to the output file
    ut.save_diff_as_image(array_to_display_1, array_to_display_2, parsed_args.output_file)


if __name__ == "__main__":
    main(sys.argv[1:])

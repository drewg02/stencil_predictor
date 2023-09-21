import sys

import utilities as ut


def main(args):
    # Argument parsing
    parser = ut.ArgParser(description='Read and print an array from a file.')
    parser.add_argument('input_file', type=str, help='Path to the input file')

    parsed_args = parser.parse_args(args)

    # Read the array from the input file
    array_to_print = ut.read_array_from_file(parsed_args.input_file)

    # Print the array
    ut.print_array(array_to_print)


if __name__ == "__main__":
    main(sys.argv[1:])

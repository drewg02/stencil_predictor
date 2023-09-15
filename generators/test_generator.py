import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import utilities as ut


def main(args):
    # Argument parsing
    parser = ut.ArgParser(description='Generate a random array and save it to a file.')
    parser.add_argument('rows', type=int, help='Number of rows for the array')
    parser.add_argument('cols', type=int, help='Number of columns for the array')
    parser.add_argument('output_file', type=str, help='Path to the output file')

    parsed_args = parser.parse_args(args)

    # Apply the function to create the new array
    new_array = ut.create_test_array(parsed_args.rows, parsed_args.cols)

    # Save the output
    ut.save_array_to_file(new_array, parsed_args.output_file)


if __name__ == "__main__":
    main(sys.argv[1:])
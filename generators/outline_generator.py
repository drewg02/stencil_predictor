import inspect
import os
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import utilities as ut


def main(args):
    # Argument parsing
    parser = ut.ArgParser(description='Generate an array with an outline')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('output_mask_file', type=str, help='Path to the output mask file')
    parser.add_argument('--rows', type=int, help='Number of rows for the array')
    parser.add_argument('--cols', type=int, help='Number of columns for the array')
    parser.add_argument('--input_file', type=str, help='File to start from')
    parser.add_argument('--input_mask_file', type=str, help='File to start from')
    parser.add_argument('--thickness', type=int, help='Thickness of the outline', default=1)
    parser.add_argument('--output_png_file', type=str, help='Path to the output png file')

    parsed_args = parser.parse_args(args)

    if (parsed_args.rows is None or parsed_args.cols is None) and parsed_args.input_file is None:
        print('You must specify either rows and cols or an input file.')
        return

    if (parsed_args.rows is not None or parsed_args.cols is not None) and parsed_args.input_file is not None:
        print('You cannot specify both rows and cols and an input file.')
        return

    # Handle the case where we are starting from an input file
    input_array = None
    input_mask = None
    if parsed_args.input_file is not None:
        input_array = ut.read_array_from_file(parsed_args.input_file)
        input_mask = ut.read_array_from_file(parsed_args.input_mask_file)

        parsed_args.rows = len(input_array)
        parsed_args.cols = len(input_array[0])

    # Apply the function to create the new array
    array, mask = ut.create_array(parsed_args.rows, parsed_args.cols, ut.ArrayType.OUTLINE, parsed_args.thickness,
                                  input_array, input_mask)

    # Save the outputs
    ut.save_array_to_file(array, parsed_args.output_file)
    ut.save_array_to_file(mask, parsed_args.output_mask_file)
    if parsed_args.output_png_file is not None:
        ut.save_array_as_image(array, parsed_args.output_png_file)


if __name__ == "__main__":
    main(sys.argv[1:])

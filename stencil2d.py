import sys
import utilities as ut


def main(args):
    # Argument parsing
    parser = ut.ArgParser(description='Apply stencil filter to an array.')
    parser.add_argument('num_iterations', type=int, help='Number of iterations for stencil filter')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('all_iterations', type=str, help='All iterations parameter', nargs='?') # if left out just don't provide it

    parsed_args = parser.parse_args(args)
    # If the all_iterations parameter is provided, save the history
    save_history = parsed_args.all_iterations is not None

    # Read the array from the input file
    initial_array = ut.read_array_from_file(parsed_args.input_file)
    # Apply the stencil filter to the array
    new_array, plate_history, max_diffs, max_squared_diffs = ut.apply_stencil(initial_array, parsed_args.num_iterations, save_history=save_history)

    # Save the array to the output files
    ut.save_array_to_file(new_array, parsed_args.output_file)
    if save_history:
        ut.save_plate_history_to_file(plate_history, parsed_args.all_iterations)
    
    ut.plot_diffs(max_diffs, max_squared_diffs, './output/final/diffs.png')

if __name__ == "__main__":
    main(sys.argv[1:])
import sys
import time

import utilities as ut


def main(args):
    # Argument parsing
    parser = ut.ArgParser(description='Apply stencil filter to an array.')
    parser.add_argument('num_iterations', type=int, help='Number of iterations for stencil filter')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('input_mask_file', type=str, help='Path to the input mask file')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('--max_diff_threshold', type=float, help='Max difference threshold')
    parser.add_argument('--avg_diff_threshold', type=float, help='Average difference threshold')
    parser.add_argument('--all_iterations', type=str, help='Path to the output file for all iterations')
    parser.add_argument('--difference_graph_file', type=str, help='Path to the output file for the difference graph')

    parsed_args = parser.parse_args(args)
    # If the all_iterations parameter is provided, save the history
    save_history = parsed_args.all_iterations is not None

    # Read the array from the input file
    input_array = ut.read_array_from_file(parsed_args.input_file)
    if input_array is None:
        return

    # Read the array from the input file
    mask_array = ut.read_array_from_file(parsed_args.input_mask_file)
    if mask_array is None:
        return

    start_time = time.time()
    # Apply the stencil filter to the array
    new_array, plate_history, max_diffs, max_squared_diffs = ut.apply_stencil(input_array, mask_array,
                                                                              parsed_args.num_iterations,
                                                                              save_history=save_history,
                                                                              max_diff_threshold=parsed_args.max_diff_threshold,
                                                                              avg_diff_threshold=parsed_args.avg_diff_threshold)

    elapsed = time.time() - start_time
    print(f"Stenciling for {parsed_args.num_iterations} iterations took {elapsed} seconds.")

    # Save the array to the output files
    ut.save_array_to_file(new_array, parsed_args.output_file)
    if save_history:
        ut.save_array_to_file(plate_history, parsed_args.all_iterations)

    if parsed_args.difference_graph_file is not None:
        ut.plot_diffs(max_diffs, max_squared_diffs, parsed_args.difference_graph_file)


if __name__ == "__main__":
    main(sys.argv[1:])

import sys
import time

import utilities as ut


def main(args):
    parser = ut.ArgParser(description='Generate a mass amount of data, total number of pairs will be num_initials * num_iterations.')
    parser.add_argument('rows', type=int, help='Number of rows for the arrays')
    parser.add_argument('cols', type=int, help='Number of columns for the arrays')
    parser.add_argument('num_initials', type=int, help='How many initial states to generate')
    parser.add_argument('num_iterations', type=int, help='How many iterations to run')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('--chance', type=float, help='Chance of a pixel being on', default=0.2)

    parsed_args = parser.parse_args(args)

    start_time = time.time()
    pairs = ut.create_pairs(parsed_args.rows, parsed_args.cols, parsed_args.num_initials, parsed_args.num_iterations, chance=parsed_args.chance)

    elapsed = time.time() - start_time
    print(f"Generating {len(pairs)} pairs took {elapsed} seconds.")

    # Save the array to the output file
    ut.save_array_to_file(pairs, parsed_args.output_file)


if __name__ == "__main__":
    main(sys.argv[1:])

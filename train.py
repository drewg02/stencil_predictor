import time
import os
import sys

import torch

import utilities as ut
import training_utilities as tut

def main(args):
    # Argument parsing
    #parser = ut.ArgParser(description='Read and print an array from a file.')
    #parser.add_argument('input_file_1', type=str, help='Path to the input file')
    #parser.add_argument('input_file_2', type=str, help='Path to the input file')
    #parser.add_argument('output_file', type=str, help='Path to the output file')

    #parsed_args = parser.parse_args(args)
    # Configurable parameters
    input_file_path = './output/mass.npy'
    input_channels = 1
    batch_size = 512
    num_epochs = 10
    n_hidden = 100
    lr=1e-3
    weight_decay=1e-3
    checkpoint_interval = 1
    loss_plot_file_path = 'output/train/epoch_loss.png'
    mase_plot_file_path = 'output/train/epoch_mase.png'
    smape_plot_file_path = 'output/train/epoch_smape.png'
    model_file_path = 'output/train/model.test'
    data_loader_path = 'output/train/test_loader.pt'
    checkpoints_folder_path = 'output/train/checkpoints/'

    # Check that all the folders exist
    for path in [loss_plot_file_path, mase_plot_file_path, smape_plot_file_path, model_file_path, data_loader_path, checkpoints_folder_path]:
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            create = input(f"WARNING: {dirname} does not exist, create it? (y/n) ")
            if create.lower() != 'y':
                exit()

            os.makedirs(dirname)

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        con = input("WARNING: CUDA is not available, training will be very slow. Continue? (y/n) ")
        if con.lower() != 'y':
            exit()

    # Prepare the data and model
    train_loader, val_loader, test_loader, img_height, img_width = tut.prepare_data_split_pair(input_file_path, batch_size, train_ratio=0.7, val_ratio=0.15)
    pair_img_predictor = tut.PairImagePredictor(n_hidden, input_channels, img_height, img_width).to(device)

    # Train the model and get epoch losses
    start_time = time.time()
    train_losses, train_mase_scores, train_smape_scores, \
        val_losses, val_mase_scores, val_smape_scores = tut.train_model_pair(
        pair_img_predictor, num_epochs, train_loader, val_loader, device,
        lr=lr, weight_decay=weight_decay, checkpoint_interval=checkpoint_interval, checkpoints_folder_path=checkpoints_folder_path)

    elapsed = time.time() - start_time
    print(f"Training {len(train_loader)} batches for {num_epochs} epochs took {elapsed} seconds.")

    # Plot losses and save the model
    tut.plot_metrics(train_losses, val_losses, 'Loss', loss_plot_file_path, plot_intersections=0)
    tut.plot_metrics(train_mase_scores, val_mase_scores, 'MASE', mase_plot_file_path, plot_intersections=10)
    tut.plot_metrics(train_smape_scores, val_smape_scores, 'sMAPE', smape_plot_file_path, plot_intersections=10)
    tut.save_dataloader(test_loader, data_loader_path)
    tut.save_model(pair_img_predictor, model_file_path)

    print("Training process completed.")


if __name__ == "__main__":
    main(sys.argv[1:])

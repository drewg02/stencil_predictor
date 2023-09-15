import torch
import training_utilities as tut


# Configurable parameters
input_file_path = 'output/all/all-101x50x50.npy'
seq_length = 10
input_channels = 1
batch_size = 1
num_epochs = 500
plot_file_path = 'output/train/epoch_losses.png'
model_file_path = 'output/train/model.test'
data_loader_path = 'output/train/test_loader.pt'


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Prepare the data and model
train_loader, val_loader, test_loader, processed_img_area, img_height, img_width = tut.prepare_data_split(input_file_path, batch_size, seq_length, train_ratio=0.7, val_ratio=0.15)
indexed_img_predictor = tut.IndexedImgPredictor2(input_channels, img_height, img_width, processed_img_area).to(device)

# Train the model and get epoch losses
train_losses, val_losses = tut.train_model(indexed_img_predictor, num_epochs, train_loader, val_loader, device)

# Plot losses and save the model
tut.plot_training_losses(train_losses, val_losses, plot_file_path)
tut.save_dataloader(test_loader, data_loader_path)
tut.save_model(indexed_img_predictor, model_file_path)


print("Training process completed!")

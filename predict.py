import os
import time

import torch
import numpy as np

import training_utilities as tut
import utilities as ut

input_channels = 1
batch_size = 32
model_file_path = 'output/train/model.test'
data_loader_path = 'output/train/test_loader.pt'
predictions_file_path = 'output/predict/predicted.npy'

# Check that all the folders exist
for path in [predictions_file_path]:
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

# Load the test loader that was saved during training
test_loader = tut.load_dataloader(data_loader_path, batch_size)

# Load the model
image_predictor = tut.load_model('output/train/model.test').to(device)

# Predict the images
start_time = time.time()
predicted_images = tut.predict_pair(test_loader, image_predictor, device)

elapsed = time.time() - start_time
print(f"Predicting {len(test_loader)} batches took {elapsed} seconds.")

input_images = []
target_images = []
for i, batch in enumerate(test_loader):
    inputs = batch[:, 0, :, :]
    inputs = inputs.reshape(-1, 1, inputs.shape[1], inputs.shape[2])
    input_images.append(inputs)

    targets = batch[:, 1, :, :]
    targets = targets.reshape(-1, 1, targets.shape[1], targets.shape[2])
    target_images.append(targets)

input_images = np.concatenate(input_images, axis=0)
target_images = np.concatenate(target_images, axis=0)

mase = ut.MASE(target_images, predicted_images)
smape = ut.sMAPE(target_images, predicted_images)
max_diff = ut.max_diff(target_images, predicted_images)
avg_diff = ut.avg_diff(target_images, predicted_images)
print(f"MASE: {mase}")
print(f"sMAPE: {smape}")
print(f"Max diff: {max_diff}")
print(f"Avg diff: {avg_diff}")

# Save the images
output_array = np.hstack((input_images, target_images, predicted_images))
ut.save_array_to_file(output_array, predictions_file_path)



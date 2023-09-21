import numpy as np
import torch

import training_utilities as tut
import utilities as ut

seq_length = 10
input_channels = 1
batch_size = 1
model_file_path = 'output/train/model.test'
input_file_path = 'output/all/all-101x50x50.npy'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load all of the data from the input file
array_data, all_loader, processed_img_area, img_height, img_width = tut.prepare_data_all(input_file_path, batch_size,
                                                                                         seq_length)

# Load the model
image_predictor = tut.load_model('output/train/model.test').to(device)

# Predict the images
predicted_images = tut.predict(all_loader, image_predictor, device)
predicted_images = np.array(predicted_images)

# Save the predicted images as a movie
ut.save_array_as_movie(predicted_images, "output/predict/all-predicted.mp4")

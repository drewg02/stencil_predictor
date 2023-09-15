import torch
import training_utilities as tut

length = 10
input_channels = 1
img_height = 50
img_width = 50
visualize_file_path = "./output/visualize/visualize_model.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create a random tensor and index for the model visualization process
x = torch.randn(length, 1, img_height, img_width).to(device)
index = torch.randint(0, 100, (length,)).to(device)

# Create the model
indexed_img_predictor = tut.IndexedImgPredictor2(input_channels, img_height, img_width, img_height * img_width).to(device)

# Visualize the model and save output to a file
tut.visualize_model(x, index, indexed_img_predictor, visualize_file_path)

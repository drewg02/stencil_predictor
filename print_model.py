import training_utilities as tut

length = 10
input_channels = 1
img_height = 50
img_width = 50
visualize_file_path = "./output/visualize/visualize_model.png"

# Load the model
image_predictor = tut.load_model('output/train/model.test')

# Print the model directly to the console
print(image_predictor)
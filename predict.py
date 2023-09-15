import torch
import training_utilities as tut
import utilities as ut

seq_length = 10
input_channels = 1
batch_size = 1
model_file_path = 'output/train/model.test'
data_loader_path = 'output/train/test_loader.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the test loader that was saved during training
test_loader = tut.load_dataloader(data_loader_path, batch_size)

# Load the model
image_predictor = tut.load_model('output/train/model.test').to(device)

# Predict the images
predicted_images = tut.predict(test_loader, image_predictor, device)

# Save the predicted images
for i, (_, target_batch, _) in enumerate(test_loader):
    target_batch_size = len(target_batch)
    for j in range(target_batch_size):
        k = (i * target_batch_size) + j

        real_image = target_batch[j].squeeze(0)
        predicted_image = predicted_images[k]

        ut.save_array_as_image(predicted_image, f"output/predict/{k}-predicted.png")
        ut.save_diff_as_image(real_image, predicted_image, f"output/predict/{k}-predicted-diff.png")
        ut.save_array_as_image(real_image, f"output/predict/{k}-real.png")
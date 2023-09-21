import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchviz import make_dot

import utilities as ut


def sequential_split(dataset, lengths):
    if sum(lengths) == 1:
        subset_lengths = [int(len(dataset) * x) for x in lengths]
        remainder = len(dataset) - sum(subset_lengths)
        for i in range(remainder):
            idx = i % len(subset_lengths)
            subset_lengths[idx] += 1
        lengths = subset_lengths

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths must equal the length of the input dataset!")

    offsets = [sum(lengths[:i]) for i in range(len(lengths))]
    return [Subset(dataset, range(offset, offset + length)) for offset, length in zip(offsets, lengths)]


def _to_loader(array_data, size_of_batch, seq_length, img_height, img_width):
    tensor_data = torch.tensor(array_data, dtype=torch.float32).reshape(-1, 1, img_height, img_width)
    seq_data = []
    target_data = []
    index_data = []
    for i in range(len(tensor_data) - seq_length):
        seq_data.append(tensor_data[i:i + seq_length])
        target_data.append(tensor_data[i + seq_length])
        index_data.append(torch.arange(i, i + seq_length))

    seq_data = torch.stack(seq_data)
    target_data = torch.stack(target_data)
    index_data = torch.stack(index_data)

    dataset = TensorDataset(seq_data, target_data, index_data)

    return DataLoader(dataset, batch_size=size_of_batch)


def prepare_data_all(path_to_data, size_of_batch, seq_length):
    array_data = ut.read_array_from_file(path_to_data)
    img_height, img_width = array_data.shape[1], array_data.shape[2]
    img_area = img_height * img_width

    all_loader = _to_loader(array_data, size_of_batch, seq_length, img_height, img_width)

    return array_data, all_loader, img_area, img_height, img_width


def prepare_data_split(path_to_data, size_of_batch, seq_length, train_ratio=0.7, val_ratio=0.15):
    array_data = ut.read_array_from_file(path_to_data)
    img_height, img_width = array_data.shape[1], array_data.shape[2]
    img_area = img_height * img_width

    data_size = len(array_data)
    train_size = int(data_size * train_ratio)
    val_size = int(data_size * val_ratio)

    assert val_size > seq_length, f"Validation size({val_size}) must be greater than sequence length({seq_length})."

    train_loader = _to_loader(array_data[:train_size], size_of_batch, seq_length, img_height, img_width)
    val_loader = _to_loader(array_data[train_size:train_size + val_size], size_of_batch, seq_length, img_height,
                            img_width)
    test_loader = _to_loader(array_data[train_size + val_size:], size_of_batch, seq_length, img_height, img_width)

    return train_loader, val_loader, test_loader, img_area, img_height, img_width


def prepare_data_split_reverse(path_to_data, size_of_batch, seq_length, train_ratio=0.7, val_ratio=0.15):
    array_data = ut.read_array_from_file(path_to_data)
    img_height, img_width = array_data.shape[1], array_data.shape[2]
    img_area = img_height * img_width

    data_size = len(array_data)
    train_size = int(data_size * train_ratio)
    val_size = int(data_size * val_ratio)
    test_size = data_size - train_size - val_size

    assert val_size > seq_length, f"Validation size({val_size}) must be greater than sequence length({seq_length})."

    test_loader = _to_loader(array_data[:test_size], size_of_batch, seq_length, img_height, img_width)
    val_loader = _to_loader(array_data[test_size:test_size + val_size], size_of_batch, seq_length, img_height,
                            img_width)
    train_loader = _to_loader(array_data[test_size + val_size:], size_of_batch, seq_length, img_height, img_width)

    return train_loader, val_loader, test_loader, img_area, img_height, img_width


def prepare_data_split_split(path_to_data, size_of_batch, seq_length, train_ratio=0.7, val_ratio=0.15):
    array_data = ut.read_array_from_file(path_to_data)
    img_height, img_width = array_data.shape[1], array_data.shape[2]
    img_area = img_height * img_width

    data_size = len(array_data)
    train_size = int(data_size * train_ratio)
    val_size = int(data_size * val_ratio)
    test_size = data_size - train_size - val_size

    assert val_size > seq_length, f"Validation size({val_size}) must be greater than sequence length({seq_length})."

    test_loader = _to_loader(array_data[:test_size], size_of_batch, seq_length, img_height, img_width)
    val_loader = _to_loader(array_data[test_size + train_size:], size_of_batch, seq_length, img_height, img_width)
    train_loader = _to_loader(array_data[test_size:test_size + train_size], size_of_batch, seq_length, img_height,
                              img_width)

    return train_loader, val_loader, test_loader, img_area, img_height, img_width


def _process_batch(model, loader, device, loss_function, optimizer=None):
    total_loss = 0
    for i, (seq_batch, target_batch, index_batch) in enumerate(loader):
        for j in range(len(seq_batch)):
            single_seq = seq_batch[j:j + 1].squeeze(0).to(device)
            single_target = target_batch[j:j + 1].squeeze(0).to(device)
            index = index_batch[j:j + 1].to(device)

            single_prediction = model(single_seq, index)
            loss = loss_function(single_prediction, single_target)

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
    return total_loss / len(loader.dataset)


def train_model(model_to_train, epoch_count, train_loader, val_loader, computation_device):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model_to_train.parameters(), lr=0.001, weight_decay=0.00001)

    train_losses = []
    val_losses = []

    for epoch in range(epoch_count):
        model_to_train.train()
        train_loss = _process_batch(model_to_train, train_loader, computation_device, loss_function, optimizer)
        train_losses.append(train_loss)

        model_to_train.eval()
        with torch.no_grad():
            val_loss = _process_batch(model_to_train, val_loader, computation_device, loss_function)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{epoch_count}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


def plot_training_losses(train_losses, val_losses, output_filename):
    ut.makedirs(output_filename)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')

    intersections = 0
    for i, (train, val) in enumerate(zip(train_losses, val_losses)):
        if train <= val:
            plt.scatter(i, train, color='red', zorder=5)
            plt.text(i, train, 'X', color='red', fontsize=12, ha='center', va='center')
            intersections += 1

        if intersections >= 10:
            break

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Over Time')
    plt.legend()
    plt.savefig(output_filename)


class IndexedImgPredictor(nn.Module):
    def __init__(self, channels, img_height, img_width, img_area):
        super(IndexedImgPredictor, self).__init__()

        self.img_height = img_height
        self.img_width = img_width

        self.conv_layer = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.linear_layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=50, out_features=25),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=25, out_features=50),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(32 * img_area + 1, img_area),
            nn.ReLU(),
            nn.Linear(img_area, img_area // 10)
        )

    def forward(self, x, index):
        x = self.conv_layer(x)
        x = self.linear_layer(x)
        x = x.view(x.size(0), -1)

        index = index.view(-1, 1)
        x = torch.cat((x, index), dim=1)

        x = self.fc_layer(x)
        x = x.view(1, self.img_height, self.img_width)

        return x

    def __repr__(self):
        return super.__repr__(self)


class IndexedImgPredictor2(nn.Module):
    def __init__(self, channels, img_height, img_width, img_area):
        super(IndexedImgPredictor2, self).__init__()

        nh = 16  # number of hidden layers
        ks = 3  # kernel size
        ps = (ks - 1) // 2  # padding size

        self.channels = channels
        self.img_height = img_height
        self.img_width = img_width
        self.img_area = img_area

        self.pool0 = nn.AvgPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(in_channels=self.channels,
                               out_channels=nh,
                               kernel_size=ks,
                               padding=ps,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(nh)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=nh,
                               out_channels=nh,
                               kernel_size=ks,
                               padding=ps,
                               stride=1)
        self.bn2 = nn.BatchNorm2d(nh)

        self.conv3 = nn.Conv2d(in_channels=nh,
                               out_channels=nh,
                               kernel_size=ks,
                               padding=ps,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(nh)

        self.conv4 = nn.Conv2d(in_channels=nh,
                               out_channels=1,
                               kernel_size=ks,
                               padding=ps,
                               stride=1)

    def forward(self, x, index):
        x = self.conv1(x)
        x = self.bn1(x).relu()

        x = self.conv2(x)
        x = self.bn2(x).relu()

        x = self.conv3(x)
        x = self.bn3(x).relu()

        x = self.conv4(x)

        x = x.view(-1, self.img_area)
        x = torch.mean(x, dim=0)

        return x.reshape(self.channels, self.img_height, self.img_width)


def predict(loader, model, device):
    model.eval()

    predicted_images = []
    for i, (sequence_batch, _, index_batch) in enumerate(loader):
        for j in range(len(sequence_batch)):
            sequence_for_prediction = sequence_batch[j:j + 1].squeeze(0).to(device)
            indexes_for_prediction = index_batch[j:j + 1].squeeze(0).to(device)

            predicted_tensor = model(sequence_for_prediction, indexes_for_prediction)

            predicted_image = predicted_tensor.squeeze().detach().cpu().numpy()
            predicted_images.append(predicted_image)

    return predicted_images


def load_dataloader(dataloader_file_path, batch_size):
    data = torch.load(dataloader_file_path)
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def save_dataloader(dataloader, dataloader_file_path):
    ut.makedirs(dataloader_file_path)

    torch.save(dataloader.dataset, dataloader_file_path)


def load_model(model_file_path):
    return torch.load(model_file_path)


def save_model(model, model_file_path):
    ut.makedirs(model_file_path)

    torch.save(model, model_file_path)


def visualize_model(dummy_x, dummy_indexes, model, visualize_file_path):
    ut.makedirs(visualize_file_path)

    file_name, file_extension = os.path.splitext(visualize_file_path)

    if not file_extension:
        file_extension = 'png'
    else:
        file_extension = file_extension[1:]

    result = model(dummy_x, dummy_indexes)
    make_dot(result, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True).render(file_name,
                                                                                                           format=file_extension)

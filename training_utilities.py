import time
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchviz import make_dot

import utilities as ut


def _to_loader_sequence(array_data, size_of_batch, seq_length, img_height, img_width):
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


def prepare_data_all_sequence(path_to_data, size_of_batch, seq_length):
    array_data = ut.read_array_from_file(path_to_data)
    img_height, img_width = array_data.shape[1], array_data.shape[2]
    img_area = img_height * img_width

    all_loader = _to_loader_sequence(array_data, size_of_batch, seq_length, img_height, img_width)

    return array_data, all_loader, img_area, img_height, img_width


def prepare_data_split_sequence(path_to_data, size_of_batch, seq_length, train_ratio=0.7, val_ratio=0.15):
    array_data = ut.read_array_from_file(path_to_data)
    img_height, img_width = array_data.shape[1], array_data.shape[2]
    img_area = img_height * img_width

    data_size = len(array_data)
    train_size = int(data_size * train_ratio)
    val_size = int(data_size * val_ratio)

    assert val_size > seq_length, f"Validation size({val_size}) must be greater than sequence length({seq_length})."

    train_loader = _to_loader_sequence(array_data[:train_size], size_of_batch, seq_length, img_height, img_width)
    val_loader = _to_loader_sequence(array_data[train_size:train_size + val_size], size_of_batch, seq_length,
                                     img_height,
                                     img_width)
    test_loader = _to_loader_sequence(array_data[train_size + val_size:], size_of_batch, seq_length, img_height,
                                      img_width)

    return train_loader, val_loader, test_loader, img_area, img_height, img_width


def prepare_data_split_pair(path_to_data, size_of_batch, train_ratio=0.7, val_ratio=0.15):
    array_data = ut.read_array_from_file(path_to_data)
    img_height, img_width = array_data.shape[2], array_data.shape[3]

    tensor_data = torch.tensor(array_data, dtype=torch.float32).reshape(-1, 2, img_height, img_width)
    dataset = TensorDataset(tensor_data)

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size],
                                                   generator=torch.Generator().manual_seed(42))

    # Convert Subset to tensor
    train_data = tensor_data[torch.tensor(train_data.indices)]
    val_data = tensor_data[torch.tensor(val_data.indices)]
    test_data = tensor_data[torch.tensor(test_data.indices)]

    train_loader = DataLoader(train_data, size_of_batch, shuffle=True)
    val_loader = DataLoader(val_data, size_of_batch)
    test_loader = DataLoader(test_data, size_of_batch)

    return train_loader, val_loader, test_loader, img_height, img_width


def _process_batch_sequence(model, loader, device, loss_function, optimizer=None):
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


def train_model_sequence(model_to_train, epoch_count, train_loader, val_loader, computation_device):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model_to_train.parameters(), lr=0.001, weight_decay=0.00001)

    train_losses = []
    val_losses = []

    for epoch in range(epoch_count):
        model_to_train.train()
        train_loss = _process_batch_sequence(model_to_train, train_loader, computation_device, loss_function, optimizer)
        train_losses.append(train_loss)

        model_to_train.eval()
        with torch.no_grad():
            val_loss = _process_batch_sequence(model_to_train, val_loader, computation_device, loss_function)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{epoch_count}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


def _process_batch_pair(model, loader, device, loss_function, optimizer=None):
    total_loss = 0
    total_mase = 0
    total_smape = 0

    for i, batch in enumerate(loader):
        batch = batch.to(device)

        inputs = batch[:, 0, :, :].to(device)
        inputs = inputs.reshape(-1, 1, inputs.shape[1], inputs.shape[2])
        targets = batch[:, 1, :, :].to(device)
        targets = targets.reshape(-1, 1, targets.shape[1], targets.shape[2])

        predictions = model(inputs)

        loss = loss_function(predictions, targets)
        mase = MASE(targets, predictions)
        smape = sMAPE(targets, predictions)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_mase += mase.item()
        total_smape += smape.item()

    avg_loss = total_loss / len(loader.dataset)
    avg_mase = total_mase / len(loader.dataset)
    avg_smape = total_smape / len(loader.dataset)

    return avg_loss, avg_mase, avg_smape


def train_model_pair(model_to_train, epoch_count, train_loader, val_loader, computation_device, lr=1e-3,
                     weight_decay=1e-3, checkpoint_interval=50, checkpoints_folder_path=None, verbose=True):
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model_to_train.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    train_mase_scores = []
    train_smape_scores = []

    val_losses = []
    val_mase_scores = []
    val_smape_scores = []

    start_time = time.time()
    for epoch in range(epoch_count):
        model_to_train.train()
        train_loss, train_mase, train_smape = _process_batch_pair(model_to_train, train_loader, computation_device,
                                                                  loss_function, optimizer)
        train_losses.append(train_loss)
        train_mase_scores.append(train_mase)
        train_smape_scores.append(train_smape)

        model_to_train.eval()
        with torch.no_grad():
            val_loss, val_mase, val_smape = _process_batch_pair(model_to_train, val_loader, computation_device,
                                                                loss_function)
        val_losses.append(val_loss)
        val_mase_scores.append(val_mase)
        val_smape_scores.append(val_smape)

        if verbose:
            elapsed_time = time.time() - start_time
            print(
                f"Epoch [{epoch + 1}/{epoch_count}], Train Loss: {train_loss:.4}, Val Loss: {val_loss:.4}, {elapsed_time} seconds elapsed")

        if checkpoints_folder_path is not None and (epoch + 1) % checkpoint_interval == 0:
            save_model(model_to_train, os.path.join(checkpoints_folder_path,
                                                    f'checkpoint{int(epoch / checkpoint_interval)}_epoch{epoch + 1}.test'))
            if verbose:
                print(f"Checkpoint {int(epoch / checkpoint_interval)} saved at epoch {epoch + 1}")

    return train_losses, train_mase_scores, train_smape_scores, val_losses, val_mase_scores, val_smape_scores


def plot_metrics(train_metrics, val_metrics, metric_name, output_filename, plot_intersections=10):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_metrics, label=f'Train {metric_name}')
        plt.plot(val_metrics, label=f'Validation {metric_name}')

        if plot_intersections is not None and plot_intersections > 0:
            intersections = 0
            for i in range(1, len(train_metrics)):
                if (train_metrics[i] <= val_metrics[i] and train_metrics[i - 1] >= val_metrics[i - 1]) or \
                        (train_metrics[i] >= val_metrics[i] and train_metrics[i - 1] <= val_metrics[i - 1]):
                    plt.scatter(i, train_metrics[i], color='red', zorder=5)
                    plt.text(i, train_metrics[i], 'X', color='red', fontsize=12, ha='center', va='center')
                    intersections += 1

                if intersections >= plot_intersections:
                    break

        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'Train and Validation {metric_name} Over Time')
        plt.legend()
        plt.savefig(output_filename)

        plt.close()

        return True
    except Exception as e:
        print(f"Couldn't save the plot: {e}")
        return False


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


class PairImagePredictor(nn.Module):
    def __init__(self, n_hidden, channels, img_height, img_width):
        super(PairImagePredictor, self).__init__()

        nh = n_hidden
        ks = 3
        ps = ks

        self.channels = channels
        self.img_height = img_height
        self.img_width = img_width

        self.pool0 = nn.AvgPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(self.channels, nh, ks, padding=ps)
        self.bn1 = nn.BatchNorm2d(nh)

        self.conv1_1 = nn.Conv2d(nh, nh, ks, padding=ps)
        self.bn1_1 = nn.BatchNorm2d(nh)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(nh, nh * 2, ks, padding=(ps - 1))
        self.bn2 = nn.BatchNorm2d(nh * 2)

        self.conv3 = nn.Conv2d(nh * 2, nh * 2, ks, padding=(ps - 1))
        self.bn3 = nn.BatchNorm2d(nh * 2)

        self.dropout = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(nh * 2, channels, ks, padding=(ps - 2))

    def forward(self, x):
        x = x.reshape(-1, self.channels, self.img_height, self.img_width)
        # print(f"After reshape: {x.shape}")

        x = self.pool0(x)
        #  print(f"After pool0: {x.shape}")

        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"After conv1: {x.shape}")

        x = F.relu(self.bn1_1(self.conv1_1(x)))
        # print(f"After conv1_1: {x.shape}")

        x = self.pool1(x)
        # print(f"After pool1: {x.shape}")

        x = F.relu(self.bn2(self.conv2(x)))
        # print(f"After conv2: {x.shape}")

        x = F.relu(self.bn3(self.conv3(x)))
        # print(f"After conv3: {x.shape}")

        x = self.dropout(x)
        # print(f"After dropout: {x.shape}")

        x = self.conv4(x)
        # print(f"After conv4: {x.shape}")

        return x.reshape(-1, self.channels, self.img_height, self.img_width)


def predict_sequence(loader, model, device):
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


def predict_pair(loader, model, device):
    model.eval()

    predicted_images = []
    for i, batch in enumerate(loader):
        batch = batch.to(device)

        inputs = batch[:, 0, :, :].to(device)
        inputs = inputs.reshape(-1, 1, inputs.shape[1], inputs.shape[2])

        predictions = model(inputs)

        predicted_image = predictions.detach().cpu().numpy()
        predicted_images.append(predicted_image)

    return np.concatenate(predicted_images)


def load_dataloader(dataloader_file_path, batch_size):
    data = torch.load(dataloader_file_path)
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def save_dataloader(dataloader, dataloader_file_path):
    try:
        torch.save(dataloader.dataset, dataloader_file_path)
    except Exception as e:
        print(f"Couldn't save the dataloader: {e}")


def load_model(model_file_path):
    try:
        return torch.load(model_file_path)
    except FileNotFoundError:
        print("Model file not found.")
        return None
    except Exception as e:
        print(f"Couldn't load the model: {e}")
        return None


def save_model(model, model_file_path):
    try:
        torch.save(model, model_file_path)
        return True
    except Exception as e:
        print(f"Couldn't save the model: {e}")
        return False


def visualize_model(dummy_x, dummy_indexes, model, visualize_file_path):
    try:
        file_name, file_extension = os.path.splitext(visualize_file_path)

        if not file_extension:
            file_extension = 'png'
        else:
            file_extension = file_extension[1:]

        result = model(dummy_x, dummy_indexes)
        make_dot(result, params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True).render(
            file_name, format=file_extension)

        return True
    except Exception as e:
        print(f"Couldn't visualize the model: {e}")
        return False


def MASE(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred))
    scale = torch.mean(torch.abs(y_true[:, 1:] - y_true[:, :-1]))
    return mae / scale


def sMAPE(y_true, y_pred):
    return torch.mean(2 * torch.abs(y_pred - y_true) / (torch.abs(y_pred) + torch.abs(y_true))) * 100

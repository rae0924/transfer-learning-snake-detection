import torch
import numpy as np
from torch import optim
from torch import nn
from torch.utils import data
from dataset import SnakeDataset
from model import SnakeDetector
from torch.utils.data import DataLoader, SubsetRandomSampler
import pickle
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_val_split(dataset, p=0.5, seed=None, batch_size=1):
    split = int(np.floor(p*len(dataset)))
    indices = np.arange(len(dataset))
    if isinstance(seed, int): np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size, sampler=val_sampler)
    return train_loader, val_loader

def train(model, train_set, val_set, save_dir='save', epochs=1):
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    model.to(device)
    train_losses, train_accuracies = [],[]
    val_losses, val_accuracies = [], []
    for epoch in range(epochs):
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_losses, val_epoch_accuracies = [], []

        for batch in train_set:
            x, y = [i.to(device) for i in batch]
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()

        for batch in train_set:
            x, y = [i.to(device) for i in batch]
            is_correct = accuracy(x,y,model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        for batch in val_set:
            x, y = [i.to(device) for i in batch]
            batch_loss = val_batch(x, y, model, loss_fn)
            val_epoch_losses.append(batch_loss)
        val_epoch_loss = np.array(val_epoch_losses).mean()

        for batch in val_set:
            x, y = [i.to(device) for i in batch]
            is_correct = accuracy(x,y,model)
            val_epoch_accuracies.extend(is_correct)
        val_epoch_accuracy = np.mean(val_epoch_accuracies)

        print(f'epoch: {epoch+1} \n' + 
        f'train_loss: {train_epoch_loss} \t train_acc: {train_epoch_accuracy} \n' +
        f'val_loss: {val_epoch_loss} \t val_acc: {val_epoch_accuracy}')
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
    
    path = os.path.join(os.path.dirname(__file__), save_dir)
    if not os.path.exists(path): os.mkdir(path)
    history = [train_losses, train_accuracies,
    val_losses, val_accuracies]
    model_path = os.path.join(path, 'snake_detector.pt')
    history_path = os.path.join(path, 'train_history.pkl')
    torch.save(model.state_dict(), model_path)
    pickle.dump(history, open(history_path, 'wb'))
    

def train_batch(x, y, model, opt, loss_fn):
    model.train()
    opt.zero_grad()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    opt.step()
    return batch_loss.item()

@torch.no_grad()
def val_batch(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = prediction.round() == y
    return is_correct.cpu().numpy().tolist()


if __name__ == "__main__":
    dataset = SnakeDataset()
    model = SnakeDetector()
    train_set, val_set = train_val_split(dataset, 0.25, 123, 32)
    train(model, train_set, val_set, epochs=50)

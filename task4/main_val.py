"""This code is inspired from https://github.com/adambielski/siamese-triplet """

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import resnet50

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

#Criterion definintion using contrastive loss defined just above
def criterion(anchor, positive, negative):
    loss = nn.TripletMarginLoss(margin=0.7)

    return loss(anchor, positive, negative)

class Preprocessing(Dataset):
    def __init__(self, food_frame, mode, transform=None):
        self.dataset_path = '/scratch-second/vibars/task4_be9ai3nsdj/food'
        self.food_frame = pd.read_csv(food_frame, header=None, index_col=None, sep=" ", dtype=str)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.food_frame.shape[0]

    def __getitem__(self, idx):
        img_ref = self.food_frame.iloc[idx, 0] + '.jpg'
        img_1 = self.food_frame.iloc[idx, 1] + '.jpg'
        img_2 = self.food_frame.iloc[idx, 2] + '.jpg'

        for img in os.listdir(self.dataset_path):
            img_path = os.path.abspath(os.path.join(self.dataset_path, img))
            if img == img_ref:
                anchor = Image.open(img_path)
                anchor = self.transform(anchor)
            elif img == img_1:
                first = Image.open(img_path)
                first = self.transform(first)
            elif img == img_2:
                second = Image.open(img_path)
                second = self.transform(second)
            else:
                continue

        return anchor, first, second


def get_loader(mode, batch_size, shuffle=True, num_workers=2):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(300),
        #    transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(300),
        #    transforms.CenterCrop(280),
            transforms.ToTensor(),
        #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(300),
        #    transforms.CenterCrop(280),
            transforms.ToTensor(),
        #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    train_frame = '/scratch-second/vibars/task4_be9ai3nsdj/train_triplets.txt'
    test_frame = '/scratch-second/vibars/task4_be9ai3nsdj/test_triplets.txt'

    if mode == 'trainval':
        dataset = Preprocessing(train_frame, mode='train', transform=data_transforms['train'])
        L = len(dataset)
        L1 = int(0.7*L)
        L2 = L - L1
        print('-'*70)
        print('Length of training set: {}, Length of validation set: {}'.format(L1, L2))
        trainset, validset = random_split(dataset, lengths=[L1, L2])
        data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        data_loader_val = DataLoader(dataset=validset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return data_loader_train, data_loader_val

    elif mode == 'test':
        testset = Preprocessing(test_frame, mode='test', transform=data_transforms['test'])
        L3 = len(testset)
        print('-' * 70)
        print('Length of test set: {}'.format(L3))
        data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return data_loader_test

class EmbeddedNet(nn.Module):
    def __init__(self):
        super(EmbeddedNet, self).__init__()
        self.model_ft = resnet50(pretrained=True, progress=True)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 2)
        #print("zzzzz", self.model_ft.fc.out_features)
        c = 0
        for module in self.model_ft.children():
            c += 1
            if c < 8:
                for para in module.parameters():
                    para.requires_grad = False

    def forward(self, x):
        #print('XXXXX', x.shape)
        output = self.model_ft(x)
        #print('YYYY', output.shape)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module): 
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)



def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics): 
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    print(type(train_loader), len(train_loader))

    for batch_idx, data in enumerate(train_loader):

        if not type(data) in (tuple, list):
            data = (data,)

        data = tuple(d.to(device) for d in data)

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)

    return total_loss, metrics


def val_epoch(val_loader, model, loss_fn, device, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, data in enumerate(val_loader):
            if not type(data) in (tuple, list):
                data = (data,)

            data = tuple(d.to(device) for d in data)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, loss_outputs)

    return val_loss, metrics

    

def fit(train_loader, val_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval, metrics=[]):

    for epoch in range(0, n_epochs):
        scheduler.step()
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = val_epoch(val_loader, model, loss_fn, device, metrics)
        val_loss /= len(val_loader)

        scheduler.step()

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)

    return Prediction_epoch(test_loader, model, device)



def Prediction_epoch(test_loader, model, device):
    predictions = []
    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(test_loader):
            if not type(data) in (tuple, list):
                data = (data,)

            data = tuple(d.to(device) for d in data)

            outputs = model(*data)
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            anchor = outputs[0]
            first = outputs[1]
            second = outputs[2]

            distance_first = (anchor - first).pow(2).sum(1) #sum of power 2 terms == distance
            distance_second = (anchor - second).pow(2).sum(1)

            for i in range(len(distance_first)):
                if distance_first[i] <= distance_second[i]:
                    predictions.append(1)
                else:
                    predictions.append(0)

    return predictions


if __name__ == '__main__':

    model = TripletNet(EmbeddedNet())
    net_opt = torch.optim.AdamW(model.parameters(), lr=0.0001) #Test AdamW
    net_scheduler = lr_scheduler.StepLR(net_opt, step_size=3, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    queue_train, queue_val = get_loader(mode='trainval', batch_size=20, shuffle=True, num_workers=2)
    queue_test = get_loader(mode='test', batch_size=20, shuffle=False, num_workers=2)

    predictions = pd.DataFrame(fit(queue_train, queue_val, queue_test, model, criterion, net_opt, net_scheduler, 12, device, 100))
    predictions.to_csv('submission.txt', index=False, header=False)

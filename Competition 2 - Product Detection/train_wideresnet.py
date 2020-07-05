import os
os.environ['TORCH_HOME'] = 'models/wide_resnet' #setting the environment variable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import copy
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from torchsampler import ImbalancedDatasetSampler

import logging

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()


if __name__ == '__main__':
    
    script_start = datetime.datetime.now().strftime('%Y%m%d%H%M')
    
    logging.basicConfig(
        filename=f'{script_start}_wide_resnet_cutmix_train_log.txt',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filemode='w')
    
    model_path = None
    model_path = 'models/wideresnet_cutmix-30.pt'
    
    batch_size = 32

    transforms_train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
#             transforms.RandomErasing(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    transforms_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    train_dataset = datasets.ImageFolder(
        root='/mnt/processed/private/msds2020cpt12/shopee-code-league/product-detection/train/train',
        transform=transforms_train)
    sampler = ImbalancedDatasetSampler(train_dataset)
    train_dataset = CutMix(train_dataset, num_class=42, beta=1.0, prob=0.5, num_mix=2)
    
    
#     test_dataset = datasets.ImageFolder(
#         root='/mnt/processed/private/msds2020cpt12/shopee-code-league/product-detection/split_2/val',
#         transform=transforms_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    net = torchvision.models.wide_resnet101_2(pretrained=True)
    use_cuda = True
    criterion = CutMixCrossEntropyLoss(True)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 42)
    net.fc = net.fc.cuda() if use_cuda else net.fc
    
    if model_path:
        net.load_state_dict(torch.load(model_path))
    
    net.to(device)

    n_epochs = 40
    print_every = 10
    valid_loss_min = np.Inf
#     val_loss = np.zeros(n_epochs)
#     val_acc = np.zeros(n_epochs)
    train_loss = np.zeros(n_epochs)
    train_acc = np.zeros(n_epochs)
    total_step = len(train_dataloader)
    for epoch in range(31, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total=0
        logging.info(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()

            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            _,target__ = torch.max(target_, dim=1)
            correct += torch.sum(pred==target__).item()
            total += target_.size(0)
            if (batch_idx) % print_every == 0:
                logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc[epoch-1] = (100 * correct / total)
        train_loss[epoch-1] = (running_loss/total_step)
        logging.info(f'train-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
#         batch_loss = 0
#         total_t=0
#         correct_t=0
#         with torch.no_grad():
#             net.eval()
#             for data_t, target_t in (test_dataloader):
#                 data_t, target_t = data_t.to(device), target_t.to(device)
#                 outputs_t = net(data_t)
#                 loss_t = criterion(outputs_t, target_t)
#                 batch_loss += loss_t.item()
#                 _,pred_t = torch.max(outputs_t, dim=1)
#                 correct_t += torch.sum(pred_t==target_t).item()
#                 total_t += target_t.size(0)
#             val_acc[epoch-1] = (100 * correct_t/total_t)
#             val_loss[epoch-1] = (batch_loss/len(test_dataloader))
#             network_learned = batch_loss < valid_loss_min
#             logging.info(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

#             if network_learned:
#                 valid_loss_min = batch_loss
#                 torch.save(net.state_dict(), f'models/resnext_cutmix-{epoch:02}.pt')
#                 logging.info('Improvement-Detected, save-model')
        
#         fig, ax = plt.subplots(1, 2,dpi=150,figsize=(16,6))
#         ax[0].plot(range(1, n_epochs+1), train_loss, label='train')
#         ax[0].plot(range(1, n_epochs+1), val_loss, label='val')
#         ax[0].set(title='Train vs. Val Loss', ylabel='loss', xlabel='epoch')
#         ax[0].legend()                
#         ax[1].plot(range(1, n_epochs+1), train_acc, label='train')
#         ax[1].plot(range(1, n_epochs+1), val_acc, label='val')
#         ax[1].set(title='Train vs. Val Accuracy', ylabel='accuracy', xlabel='epoch')
#         ax[1].legend()

#         fig.savefig('20-epochs-train-cutout.png')
        
        if epoch%4==0 and epoch!=0:
            torch.save(net.state_dict(), f'models/wideresnet_cutmix-{epoch:02}.pt')

            holdout_dataset = ImageFolderWithPaths(
                root='/mnt/processed/private/msds2020cpt12/shopee-code-league/product-detection/test',
                transform=transforms_val)
            holdout_dataloader = DataLoader(holdout_dataset, batch_size=batch_size)

            inference = {}

            with torch.no_grad():
                net.eval()
                for data_t, _, filepaths in holdout_dataloader:
                    data_t = data_t.to(device)
                    outputs_t = net(data_t)

                    for img_path, pred in zip(filepaths, outputs_t):
                        inference[img_path.split('/')[-1]] = np.argmax(
                            pred.to('cpu').numpy()
                        )

            df_inference = pd.DataFrame(
                {
                    'filename':list(inference.keys()),
                    'category':list(inference.values())
                }
            )
            df_inference.category = df_inference.category.apply(lambda x: f'{x:02}')

            df_test = pd.read_csv(
                '/mnt/processed/private/msds2020cpt12/shopee-code-league/product-detection/test.csv')
            (df_test.drop('category', axis=1)
             .merge(df_inference, left_on='filename', right_on='filename')
             .to_csv(f"submission_wideresnet_cutmix-{epoch:02}.csv", index=False))
            
        net.train()
    torch.save(net.state_dict(), f'models/wideresnet_cutmix-{epoch:02}.pt')


    
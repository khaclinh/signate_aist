from Dataset import CustomImageDataset
from Sampler import ImbalancedDatasetSampler
from Model import AIST_model
from Loss import FocalLoss
from Run_model import model_generator
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import pandas as pd

# Variable
code_path = os.getcwd()
parent_path = os.path.abspath(os.path.join(code_path, os.pardir))
data_path = os.path.join(parent_path, 'data')
labels_path = os.path.join(data_path, 'train_master.tsv')
img_folders = ['train_1', 'train_2', 'train_3']
saving_weights_path = os.path.join(data_path, 'model_weights.pth')
saving_csv_path = os.path.join(data_path, 'results.csv')

ratio = 0.1  # positive class/total
BATCH_SIZE = 256
EPOCH = 10


# create transformation function
def transform(percent):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.RandomApply([transforms.RandomHorizontalFlip(p=percent),
                                                     transforms.RandomVerticalFlip(p=percent)])])
    return tf


# create training dataset
dataset = CustomImageDataset(labels_path, data_path, img_folders, transform, None)

# split to train and valid data
torch.manual_seed(0)
train, valid = random_split(dataset, [236800, 59382])  # 296182

# create dataloader
train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, num_workers=1,
                              sampler=ImbalancedDatasetSampler(train.dataset, train.indices, BATCH_SIZE, ratio))
valid_dataloader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True)

# declare optimization method and loss function
# class_weights = torch.tensor([ratio, 1-ratio]).float().cuda()
# loss_fn = nn.CrossEntropyLoss(class_weights)
loss_fn = FocalLoss(alpha=ratio, gamma=2)
optimizer = optim.Adam(AIST_model.parameters(), lr=1e-3, weight_decay=1e-5)

# run model
model = model_generator(train_dataloader, valid_dataloader, AIST_model, loss_fn, optimizer, num_epoch=EPOCH,
                        save_path=saving_weights_path)
train_loss, valid_loss, train_IoU, valid_IoU = model.run()

# saving to csv file
content = {'epoch': [(i + 1) for i in range(EPOCH)], 'training_loss': train_loss, 'testing_loss': valid_loss,
           'training_IoU': train_IoU, 'testing_IoU': valid_IoU}
df = pd.DataFrame(content)
df.to_csv(saving_csv_path)

# graph
plt.title('Training and Validation Loss')
plt.plot(train_loss, label="Training Loss")
plt.plot(valid_loss, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

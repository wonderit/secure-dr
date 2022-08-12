#!/usr/bin/env python
# coding: utf-8
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # Plotting
import seaborn as sns # Plotting

# Import Image Libraries - Pillow and OpenCV
from PIL import Image
import cv2

# Import PyTorch and useful functions
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torch.optim as optim

# Import useful sklearn functions
from sklearn.metrics import cohen_kappa_score, accuracy_score

from tqdm import tqdm
import alexnet
import lenet
import os
print(os.listdir("."))

model_name = 'lenet_224'
image_width = 224
base_dir = "./aptos2019-blindness-detection/"


# # Loading Data + EDA

train_csv = pd.read_csv('./aptos2019-blindness-detection/train.csv')
test_csv = pd.read_csv('./aptos2019-blindness-detection/test.csv')

print('Train Size = {}'.format(len(train_csv)))
print('Public Test Size = {}'.format(len(test_csv)))

print(train_csv.head())

counts = train_csv['diagnosis'].value_counts()
class_list = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']
for i,x in enumerate(class_list):
    counts[x] = counts.pop(i)

plt.figure(figsize=(10,5))
sns.barplot(counts.index, counts.values, alpha=0.8, palette='bright')
plt.title('Distribution of Output Classes')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Target Classes', fontsize=12)
plt.savefig('dist_class.png')


# # Data Processing
# Our own custom class for datasets
class CreateDataset(Dataset):
    def __init__(self, df_data, data_dir = './', transform=None):
        super().__init__()
        self.df = df_data.values
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_name,label = self.df[index]
        img_path = os.path.join(self.data_dir, img_name+'.png')
        image = cv2.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# In[ ]:
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_width, image_width)),
    transforms.RandomHorizontalFlip(p=0.4),
    #transforms.ColorJitter(brightness=2, contrast=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


train_path = "./aptos2019-blindness-detection/train_images/"
test_path = "./aptos2019-blindness-detection/test_images/"


train_data = CreateDataset(df_data=train_csv, data_dir=train_path, transform=transforms)

# Set Batch Size
batch_size = 64

# Percentage of training set to use as validation
valid_size = 0.2

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Create Samplers
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define Model Architecture
# model = vgg.__dict__['vgg16']()
# model_name = 'alexnet'
# model = alexnet.AlexNet().to(device)

model = lenet.LeNet().to(device)
print('model', model)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    model.cuda()

# Trainable Parameters
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: \n{}".format(pytorch_total_params))

# # Training (Fine Tuning) and Validation
# specify loss function (categorical cross-entropy loss)
criterion = nn.MSELoss()

# specify optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.00015)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 15

valid_loss_min = np.Inf

# keeping track of losses as it happen
train_losses = []
valid_losses = []
val_kappa = []
test_accuracies = []
valid_accuracies = []
kappa_epoch = []
batch = 0

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in tqdm(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda().float()
        target = target.view(-1, 1)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output.to(torch.float), target.to(torch.float))
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
        # Update Train loss and accuracies
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in tqdm(valid_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda().float()
        # forward pass: compute predicted outputs by passing inputs to the model
        target = target.view(-1, 1)
        with torch.set_grad_enabled(True):
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        #output = output.cohen_kappa_score_kappa_score)
        y_actual = target.data.cpu().numpy()
        y_pred = output[:,-1].detach().cpu().numpy()
        val_kappa.append(cohen_kappa_score(y_actual, y_pred.round()))        
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    valid_kappa = np.mean(val_kappa)
    kappa_epoch.append(np.mean(val_kappa))
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
        
    # print training/validation statistics 
    print('Epoch: {} | Training Loss: {:.6f} | Val. Loss: {:.6f} | Val. Kappa Score: {:.4f}'.format(
        epoch, train_loss, valid_loss, valid_kappa))
    
    ##################
    # Early Stopping #
    ##################
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), f'best_model_{model_name}.pt')
        valid_loss_min = valid_loss


# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.cla()
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.savefig(f"learning-curve-{model_name}.png")

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.cla()
plt.plot(kappa_epoch, label='Val Kappa Score/Epochs')
plt.legend("")
plt.xlabel("Epochs")
plt.ylabel("Kappa Score")
plt.legend(frameon=False)
plt.savefig(f"learning-curve-kappa-{model_name}.png")


# In[ ]:


model.load_state_dict(torch.load(f'best_model_{model_name}.pt'))


# # Inference

# In[ ]:


test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((image_width, image_width)),
    #torchvision.transforms.ColorJitter(brightness=2, contrast=2),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


# In[ ]:


test_csv['diagnosis'] = -1


# In[ ]:


test_data = CreateDataset(df_data=test_csv, data_dir=test_path, transform=test_transforms)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# In[ ]:


def round_off_preds(preds, coef=[0.5, 1.5, 2.5, 3.5]):
   for i, pred in enumerate(preds):
           if pred < coef[0]:
               preds[i] = 0
           elif pred >= coef[0] and pred < coef[1]:
               preds[i] = 1
           elif pred >= coef[1] and pred < coef[2]:
               preds[i] = 2
           elif pred >= coef[2] and pred < coef[3]:
               preds[i] = 3
           else:
               preds[i] = 4
   return preds


# In[ ]:


def predict(testloader):
    '''Function used to make predictions on the test set'''
    model.eval()
    preds = []
    for batch_i, (data, target) in enumerate(testloader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pr = output.detach().cpu().numpy()
        # print(pr.shape)
        for i in pr:
            preds.append(i.item())
            
    return preds


# ### TTA (Test Time Augmentation)

# In[ ]:

#
# preds1 = np.array(predict(testloader=test_loader))
# preds2 = np.array(predict(testloader=test_loader))
# preds3 = np.array(predict(testloader=test_loader))
# preds4 = np.array(predict(testloader=test_loader))
# preds5 = np.array(predict(testloader=test_loader))
# preds6 = np.array(predict(testloader=test_loader))
# preds7 = np.array(predict(testloader=test_loader))
# preds8 = np.array(predict(testloader=test_loader))
#
# preds = (preds1 + preds2 + preds3 + preds4 +
#          preds5 + preds6 + preds7 + preds8)/8.0

# preds = round_off_preds(preds)
preds = np.array(predict(testloader=test_loader))


# # Generating Submission File
sample_sub = pd.read_csv('./aptos2019-blindness-detection/sample_submission.csv')

sample_sub.diagnosis = preds
sample_sub.diagnosis = sample_sub['diagnosis'].astype(int)

sample_sub.head()

sample_sub.to_csv(f'submission_{model_name}.csv', index=False)

# #### Give this kernel an upvote if you found this helpful.
# 
# **Credits:** Abhishek's [Inference Kernel](https://www.kaggle.com/abhishek/pytorch-inference-kernel-lazy-tta)

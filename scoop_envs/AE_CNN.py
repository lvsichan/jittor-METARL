#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:51:45 2021

@author: msi
"""
import numpy as np
# from sklearn.model_selection import train_test_split
# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.utils.data as Data 
# import torch.optim as optim
# from torchsummary import summary
# import torch.utils.data as Data

import jittor as jt
from jittor import nn
import jittor.optim as optim
from jittor.dataset import VarDataset
import os

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(2, 64, 7, 2, 3),   # [-1, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 64, 7, 2, 3),  # [-1, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 64, 5, 2, 2),  # [-1, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 128, 5, 2, 2),  #  [-1, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, 128, 5, 2, 2), # [, 64, 4, 4]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, 128, 2, 2), # [, 64, 4, 4]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 64, 2, 1), # [, 64, 4, 4]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            # nn.Conv2d(128, 128, 4, 1, 0), # [, 64, 1, 1]
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.1),
        )
        
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 2 ,1, 0),   # [, 128, 24, 24]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(128, 128, 2, 2),   # [, 128, 48, 48]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(128, 128, 6, 2, 2),    # [, 64, 48, 48]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(128, 64, 6, 2, 2),    # [, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(64, 64, 6, 2, 2),      # [, 32, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(64, 64, 8, 2, 3),      # [, 32, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose2d(64, 2, 8, 2, 3)
            
        )
    
    def execute(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder
    def encode(self,x):
        return self.encoder(x)
def normalize(data):
    '''
    mu=np.mean(data)
    sigma=np.std(data)
    print(mu,sigma)
    '''
    mask=np.zeros([1000,2,128,128])
    mask[:,:,4:124,4:124]=1
    data[mask==0]=0
    
    mu=0
    sigma=1.1
    
    return (data-mu)/sigma
if __name__=='__main__':
    model=AutoEncoder()
    # model=model.cuda()
    criterion=nn.MSELoss()
    lr=1e-3
    # summary(model,(2,128,128))
    path='/home/msi/yxh/DATA/scoop_data'
    epoches=30
    optimizer=optim.Adam(model.parameters(),lr=lr)

    files = os.listdir(path)
    for epoch in range(epoches):
        print(epoch)
        if epoch in [20,28]:
            lr*=0.1
        
        for file in files:
            train_data=np.load(path+'/'+file).reshape([-1,2,128,128])
            train_data=normalize(train_data)
            train_data = VarDataset(train_data)
            train_loader = train_data.set_attrs(batch_size=128,shuffle = True,num_workers=4)
            # train_loader = Data.DataLoader(
            #     dataset=train_data,
            #     batch_size=128,
            #     shuffle=True,
            #     num_workers=4
            # )
            model.train()
            train_loss_epoch=0
            train_num=0
            
            for step, img in enumerate(train_loader):
                img=img.float()
                _, output = model(img)
                output=output.cuda()
                loss = criterion(output, img)
                optimizer.zero_grad()
                # loss.backward()
                optimizer.step(loss)
                train_loss_epoch += loss.item() * img.size(0)
                train_num += img.size(0)
                print('epoch:{} loss:{:7f}'.format(epoch,loss.item()))
                
        # torch.save(model, "./scoop_model/autoencoder%d.pkl"%epoch)
        model.save("./scoop_model/autoencoder%d.pkl"%epoch)
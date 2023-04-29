#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import gc
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from sys import argv
from torchvision import datasets, transforms

num_classes = 10
in_channels = 3
# scenario 19
drop_out = 0
lr = 0.0001
weight_decay = 0.00016052
grad_clip = 0.02576639
epochs = 700
epochs_to_display = 100

	
def conv_block(in_channels, out_channels, drop_out=0, pool=False):
	layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			  nn.BatchNorm2d(out_channels),
			  nn.ReLU(inplace=True), nn.Dropout(drop_out)
			  ]
	if pool: layers.append(nn.MaxPool2d(2))
	return nn.Sequential(*layers)

class NET(nn.Module):
	def __init__(self, in_channels, num_classes, drop_out):

		super().__init__()

		self.conv1 = conv_block(in_channels, 64, drop_out)
		self.conv2 = conv_block(64, 128, drop_out, pool=True)
		self.res1 = nn.Sequential(conv_block(128, 128, drop_out), conv_block(128, 128, drop_out))
		self.dropout = nn.Dropout(drop_out)
		self.conv3 = conv_block(128, 256, drop_out, pool=True)
		self.conv4 = conv_block(256, 512, drop_out, pool=True)
		self.res2 = nn.Sequential(conv_block(512, 512, drop_out), conv_block(512, 512, drop_out))
		self.conv5 = conv_block(512, 1028, drop_out, pool=True)
		self.res3 = nn.Sequential(conv_block(1028, 1028, drop_out), conv_block(1028, 1028, drop_out))

		self.classifier = nn.Sequential(nn.MaxPool2d(2),
										nn.Flatten(),
										nn.Linear(1028, num_classes))


	def forward(self, xb):
		out = self.conv1(xb)
		out = self.conv2(out)
		out = self.res1(out) + out
		out = self.conv3(out)
		out = self.dropout(out)
		out = self.conv4(out)
		out = self.dropout(out)
		out = self.res2(out) + out
		out = self.conv5(out)
		out = self.res3(out) + out
		out = self.classifier(out)
		return out



def train(model, device, train_loader, optimizer, epoch, grad_clip=None, sched=None, display=True):
	model.train()
	loss_function = nn.CosineEmbeddingLoss()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		#loss = F.cross_entropy(output, target)
		GT = torch.zeros((len(target),10)) # Ground truth tensor (One hot encoding)
		for idx in range(len(target)):
			GT[idx][target[idx]] = 1
		GT = GT.to(device)
		loss = loss_function(output, GT, torch.Tensor(output.size(0)).to(device).fill_(1.0))
		loss.backward()
		if grad_clip:
			nn.utils.clip_grad_value_(model.parameters(), grad_clip)
		optimizer.step()
		if sched:
			sched.step()
		
	if display and (batch_idx == 0 or batch_idx + 1 == len(train_loader)):
	  print('Train: Epoch {} [{}/{} ({:.0f}%)],\tLoss: {:.6f}'.format(
		  epoch + 1, batch_idx * len(data), len(train_loader.dataset),
		  100. * batch_idx / len(train_loader), loss.detach().item()))
	
	del loss, output
	gc.collect()
	if device == torch.device('cuda'):
		torch.cuda.empty_cache()


if __name__=="__main__":
	if len(argv)==1:
		input_dir = '.'
		output_dir = '.'
	else:
		input_dir = os.path.abspath(argv[1])
		output_dir = os.path.abspath(argv[2])

	print("Using input_dir: " + input_dir)
	print("Using output_dir: " + output_dir)


	##################### YOUR CODE GOES HERE
	### Preparation
	print("[Preparation] Start...")
	# select device
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	device_name = torch.cuda.get_device_name(0) if device == torch.device('cuda') else 'cpu'
	#device = torch.device("cpu")
	print(f'Running on {device_name}')
	#print(f'torch version: {torch.__version__}')
	# dataset: normalize and convert to tensor
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
		transforms.RandomGrayscale(),
		transforms.RandomHorizontalFlip(),
		torchvision.transforms.RandomAffine(degrees=30),
		transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), 
		transforms.ToTensor(),
		normalize]) #careful to keep this one same
	transform = transforms.Compose([transforms.ToTensor(), normalize])

	# dataset: load cifar10 data
	print(os.listdir(os.path.join(input_dir)))
	cifar_data = torchvision.datasets.ImageFolder(root=os.path.join(input_dir, 'train'), transform=transform_train)

	
	# dataset: initialize dataloaders for train and validation set
	train_loader = torch.utils.data.DataLoader(cifar_data, batch_size=128, shuffle=True)

	# model: initialize model
	model = NET(in_channels, num_classes, drop_out)
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(),
								lr=lr,
								weight_decay=weight_decay)
	
	sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, 
																	steps_per_epoch=len(train_loader))
																	
	scenario_description = 'Epochs: %d - lr: - %s - dropout: %s - Weight_decay: %s - Grad_clip: %s'%\
                                    (epochs, lr, drop_out, weight_decay, grad_clip)
	print(scenario_description)
	print("[Preparation] Done")

	### Training
	# model: training loop
	print("[Training] Start...\n")
	start_time = time.time()
	for epoch in range(epochs):
		train(model, device, train_loader, optimizer, epoch, grad_clip=grad_clip, sched=sched, display=epoch%epochs_to_display==0 or epoch==epochs - 1)
		#test(model, device, val_loader)
		#if epoch > epochs * 0.85:
			#break
	train_time = time.time() - start_time
	print(f"\n[Training] Done in {train_time:.3f} (s)")
	##################### END OF YOUR CODE


	### Saving Outputs
	print("[Saving Outputs] Start...")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# test evaluation: make predictions
	print("[Saving Outputs] Test set...")
	test_data = torchvision.datasets.ImageFolder(root=os.path.join(input_dir, 'test'), transform=transform)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
	test_predictions = []
	model.eval()
	with torch.no_grad():
		for data, _ in test_loader:
			data = data.to(device)
			output = model(data)
			pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
			test_predictions.extend(pred.squeeze().cpu().tolist())

	# test evaluation: save predictions
	test_str = '\n'.join(list(map(str, test_predictions)))
	with open(os.path.join(output_dir, 'answer_test.txt'), 'w') as result_file:
		result_file.write(test_str)
	print("[Saving Outputs] Done")

	print("All done!")


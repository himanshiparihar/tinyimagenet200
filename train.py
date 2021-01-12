import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import torch as T 
import torch.optim as optim
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from sys import argv, exit
import copy
import PIL
import time
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function 1
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.logsoft1 = nn.LogSoftmax(dim=1)
        
        # Linear function 2: 300 --> 300
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.logsoft2 = nn.LogSoftmax(dim=1)
        
        # Linear function 3: 300 --> 300
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.logsoft3 = nn.LogSoftmax(dim=1)
        
        # Linear function 4 (readout): 300 --> 200
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.logsoft1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.logsoft2(out)
        
        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.logsoft3(out)
        # Linear function 4 (readout)
        out = self.fc4(out)
        return out

def getTransform():
	customTransform = transforms.Compose([transforms.RandomRotation((-270,270)),
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

	return customTransform


def saveLosses(trainLosses, testLosses):
	xAxis = range(1, len(trainLosses) + 1)
	
	plt.plot(xAxis, trainLosses, label="Train Loss")
	plt.plot(xAxis, testLosses, label="Test Loss")
	plt.legend()
	plt.savefig("losses.png")
	plt.clf()


def train(model, device, optimizer, criterion, trainloader, testloader, weightName, epochs):
	miniBatch = 0
	runningLoss = 0.0
	printMiniBatch = 100
	bestAcc = 0.0
	bestWts = copy.deepcopy(model.state_dict())
	trainLosses, testLosses = [], []

	startTime = time.time()
	for epoch in range(epochs):
		for inputs, labels in trainloader:
			miniBatch += 1
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()
			# logps = model.forward(inputs)
			logps=model(inputs)
			loss = criterion(logps, labels)
			loss.backward()
			optimizer.step()
			runningLoss += loss.item()

			if(miniBatch%printMiniBatch == 0):
				testLoss, testAcc = evaluate(model, device, testloader, criterion)

				trainLoss = runningLoss/printMiniBatch 
				trainLosses.append(trainLoss)
				testLosses.append(testLoss)
				saveLosses(trainLosses, testLosses)

				print("Epoch: {}/{}, Minibatch: {}/{}, Train Loss: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}"
					.format(
					epoch+1,
					epochs,
					miniBatch,
					epochs*len(trainloader),
					trainLoss, 
					testLoss,
					testAcc))
				
				runningLoss = 0.0
				
				model.train()

				if(testAcc > bestAcc):
					bestAcc = testAcc
					bestWts = copy.deepcopy(model.state_dict())
					T.save(bestWts, weightName)

		epochWeight = "epoch{}.pth".format(epoch+1)
		bestWts = copy.deepcopy(model.state_dict())
		T.save(bestWts, epochWeight)

		scheduler.step()
	endTime = time.time()

	print("Training completed in {:.4f} seconds".format(endTime - startTime))
	
	model.load_state_dict(bestWts)
	# T.save(bestWts, weightName)


	return model

def evaluate(model, device, testloader, criterion):
	testLoss = 0
	testAcc = 0

	model.eval()
	with torch.no_grad():
		for inputs, labels in testloader:
			inputs, labels = inputs.to(device), labels.to(device)
			logps = model.forward(inputs)
			batchLoss = criterion(logps, labels)
			testLoss += batchLoss.item()

			ps = torch.exp(logps)
			topP, topClass = ps.topk(1, dim=1)
			equals = topClass == labels.view(*topClass.shape)
			testAcc += torch.mean(equals.type(torch.FloatTensor)).item()

	testAcc = 100 * testAcc/len(testloader)
	testLoss = testLoss/len(testloader)

	return testLoss, testAcc

def predictImage(img, model, device,testloader):
	
	testTransform = getTransform()

	model.eval()
	with torch.no_grad():
		imgTensor = testTransform(img)
		imgTensor = imgTensor.unsqueeze_(0)
		imgTensor = imgTensor.to(device)	
		predict = model(imgTensor)
		index = predict.data.cpu().numpy().argmax()


# calculating confusion matrix
		for i, data in enumerate(testloader, 0):
            # get the inputs
            t_image, mask = data
            t_image, mask = Variable(t_image.to(device)), Variable(mask.to(device))
        
            output = model(t_image)
            pred = torch.exp(output)
            conf_matrix = confusion_matrix(pred, mask)
            print(conf_matrix)


	return index, torch.exp(predict).data.cpu().numpy()

	
def evalImages(dataDir, model, device, classNames,testloader):
	classFolder = classNames[0]
	imgFiles = os.listdir(dataDir+classFolder)

	correctCount = 0

	for i, imgFile in enumerate(imgFiles):
		try:
			img = PIL.Image.open(os.path.join(dataDir, classFolder, imgFile))
		except IOError:
			continue

		index, probs = predictImage(img, model, device,testloader)
		# print("{}. Image belongs to class: {} | Probabilities: {}".format(
		# 	i, classNames[index], probs))

		# plt.imshow(np.asarray(img))
		# plt.show()

		if(classNames[index] == classFolder):
			correctCount += 1

	print("Accuracy for {} class is: {:.4f} | Correct Prediction: {} | Total Images: {} ".format(
		classFolder, correctCount*100/len(imgFiles),
		correctCount,
		len(imgFiles)))



if __name__ == '__main__':
	root_dir = 'tiny-imagenet-200/'
	dataDir = 'tiny-imagenet-200/train'
	dataDir2 = 'tiny-imagenet-200/test'
	weightName = "tinp.pth"
	image_transform = {
	'train': transforms.Compose([transforms.RandomRotation((-270,270)),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]),
	'test':  transforms.Compose([transforms.RandomRotation((-270,270)),transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])}
	
	class ImageFolderWithPaths(torchvision.datasets.ImageFolder):    
	    def __getitem__(self, index):
	        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
	        path = self.imgs[index][0]
	        tuple_with_path = (original_tuple + (path,))
	        return tuple_with_path


	trainData = datasets.ImageFolder(os.path.join(root_dir,'train'), transform=image_transform['train'])
	testData = ImageFolderWithPaths(root=dataDir2, transform=image_transform['test'])
	trainloader = T.utils.data.DataLoader(trainData,shuffle = True ,   batch_size=8, num_workers= 4)
	testloader = T.utils.data.DataLoader(testData, shuffle = True , batch_size=8, num_workers= 4)
	
	device = T.device("cuda" if T.cuda.is_available() else "cpu")
	# numclasses=200
	input_dim = 224*224*3
	hidden_dim = 300
	output_dim = 200
	epochs = 10
	classNames = trainData.classes


	# model = getModel2(device, numClasses)
	model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

	criterion = nn.NLLLoss()
	# optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
	optimizer = optim.Adam(model.parameters(), lr=0.003)
	expLrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	model = train(model, device, optimizer, criterion, trainloader, testloader, weightName, epochs)

	model.load_state_dict(torch.load(weightName))
	testLoss, testAcc = evaluate(model, device, testloader, criterion)
	print("Final Accuracy: {:.4f} and Loss: {:.4f}".format(testAcc, testLoss))

	evalImages(dataDir, model, device, classNames,testloader)


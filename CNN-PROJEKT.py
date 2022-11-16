from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')




# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,  sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']



# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    #print(images[idx].shape)
    ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])






# define the CNN architecture
class Net(nn.Module):
   

    def __init__(self):
        
        super(Net, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x

# create a complete CNN
model = Net()
#print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()


#model trained data is in model_cifar.pt
train_again = False

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=.01)


train_losslist = []
# number of epochs to train the model
n_epochs = [*range(30)] # CARE WHAT U WRITE HERE, 1 EPOCH IS 6 MINUTES ON AMD Ryzen 5 2600X Six-Core Processor  3.60 GHz

valid_loss_min = np.Inf # track change in validation loss
if train_again:
    for epoch in n_epochs:
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        counter = 0
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            if counter % 1000 == 0:
                print("epoch: {}, counter: {}, batch_loss: {}".format(epoch, counter, loss))
            counter +=20
            

            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
        
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        train_losslist.append(train_loss)
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model_cifar.pt')
            valid_loss_min = valid_loss



    plt.plot(n_epochs, train_losslist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Performance of Model 3")
    plt.show()
else:
    checkpoint = torch.load("model_cifar.pt")
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint)

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()

prvi = True
cifar_outputs = []
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)

    


    if prvi:
        cifar_outputs = output.detach().numpy()
        #print(cifar_outputs.shape)
        prvi = False
    else:
       cifar_outputs = np.append(cifar_outputs, output.detach().numpy(),axis = 0)
       #print(cifar_outputs.shape)
    #print(cifar_outputs)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

print(cifar_outputs.shape)

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


#batch size for svhn analisys
batch_size = 100

svhn_data = datasets.SVHN('data', split='train', download=True, transform=transform)
svhn_test_data = datasets.SVHN('data', split='train', download=True, transform=transform)

# prepare data loaders (combine dataset and sampler)

svhn_test_loader = torch.utils.data.DataLoader(svhn_test_data, batch_size=batch_size, num_workers=num_workers)

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
counter = 0
svhn_outputs = []
prvi = True

mostConfident = {}
labels = {}
data_keys = {}
# iterate over test data
for data, target in svhn_test_loader:
    if counter > 10000:
        break
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    for cntr,prob in enumerate(output):
        
        if len(mostConfident) < 20:
            data[cntr] = data[cntr]
            mostConfident[cntr] =max(torch.nn.functional.softmax(prob))
            labels[cntr] = np.argmax(prob.detach().numpy(), axis = 0)

        else:
            if(min(mostConfident.values()) < max(prob)):
                key = min(mostConfident.items(), key=lambda x: x[1])[0]
                data_keys[key] = data[cntr]
                mostConfident[key] = max(torch.nn.functional.softmax(prob))
                labels[key] = np.argmax(prob.detach().numpy(), axis = 0)

    if prvi:
        svhn_outputs = output.detach().numpy()
        #print(cifar_outputs.shape)
        prvi = False
    else:
       svhn_outputs = np.append(svhn_outputs, output.detach().numpy(),axis = 0)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
    if counter % 100 == 0:
        print(counter)
    counter += 100


#prints 20 images the clasifier is most confident about
fig2 = plt.figure(figsize=(25, 4))
print(len(data_keys))
for idx, key in enumerate(data_keys.keys()):
    ax = fig2.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    lab = labels[key]
    img = data_keys[key]
    conf = mostConfident[key]
    imshow(img.detach().numpy())
    ax.set_title(conf.detach().numpy())
    text = ax.text(0,-10,classes[lab], size=12)
plt.show()

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

classes = ["0","1","2","3", "4", "5", "6", "7", "8", "9"]
for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


def histo(dataCIFAR,dataSVHN):
    dataCIFAR = dataCIFAR.flatten()
    dataSVHN = dataSVHN.flatten()
    plt.figure(figsize=(20,10))
    plt.title("Trained on CIFAR10")
    plt.xlabel("log probs")
    plt.hist(-dataCIFAR, label="CIFAR10", density=True, bins=100) #log = True
    plt.hist(-dataSVHN, label="SVHN", density=True, bins=100) #log = True
    plt.legend()
    plt.show()

histo(np.amax(cifar_outputs, axis = 1),np.amax(svhn_outputs, axis = 1))
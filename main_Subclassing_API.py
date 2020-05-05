import torch 
import torch.nn as nn 
from torchvision import datasets, transforms
import torch.optim as optim
import collections
import datetime 
import os 


# _Start: change working directoy scope 
cwd = os.getcwd() 
os.chdir(cwd)
# _End: change working directory scope 


class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 8 * 8 * 8)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out


class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']


def CIFAR10_Loading(data_path:str): 

    cifar10 = datasets.CIFAR10(data_path, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4915, 0.4823, 0.4468),
                                                   (0.2470, 0.2435, 0.2616))
                          ]))

    cifar10_val = datasets.CIFAR10(data_path, train=False, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4915, 0.4823, 0.4468),
                                                   (0.2470, 0.2435, 0.2616))
                          ]))

    label_map = {0: 0, 2: 1} 
    class_names = ['airplane', 'bird']
    cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
    cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]

    return cifar2, cifar2_val 



def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, device:str):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)  # <1>
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch, loss_train / len(train_loader)))



def validate(model, train_loader, val_loader, device:str):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():  # <1>
            for imgs, labels in loader:
                imgs = imgs.to(device=device)  
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) # <2>
                total += labels.shape[0]  # <3>
                correct += int((predicted == labels).sum())  # <4>

        print("Accuracy {}: {:.2f}".format(name , correct / total))



def main():

    # _Start: data loading 
    cifar2 , cifar2_val = CIFAR10_Loading(data_path="./dataset")
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)
    # _End: data loading 


    # _Start: instantiate your model architecture 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNNnet().to(device=device)
    numel_list = [p.numel() for p in model.parameters()]
    print(sum(numel_list), numel_list)
    # _End: instantiate your model architecture 



    # _Start: training loop 
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()  

    training_loop(  
        n_epochs = 100,
        optimizer = optimizer,
        model = model,
        loss_fn = loss_fn,
        train_loader = train_loader,
        device=device
        )  
    # _End: training loop 


    # _Start: performance estimation 
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=False)
    val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

    for loader in [train_loader, val_loader]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)  
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy: %f" % (correct / total))
    # _End: performance estimation 


    # _Start: save your trained model 
    torch.save(model.state_dict(), "./model_list/" + 'birds_vs_airplanes.pt')
    # _End: save your trained model 



if __name__ == "__main__":
    main()
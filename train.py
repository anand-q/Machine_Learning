import os
from torchvision.transforms import ToPILImage
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

from architecture import Net

train_dir = "./cifar10/train"
test_dir = "./cifar10/test"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# ## Donwload cifar10 batches and convert to jpgs into class folders in ecah train and test folder

# # Ensure directories exist
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)

# # Function to save images
# def save_images(dataset, save_dir):
#     to_pil = ToPILImage()  # Transform to PIL image
#     for idx, (img, label) in enumerate(dataset):
#         # Convert tensor to PIL image
#         pil_img = to_pil(img)
        
#         # Save the image in class-labeled subfolders
#         class_dir = os.path.join(save_dir, str(label))
#         os.makedirs(class_dir, exist_ok=True)
#         pil_img.save(os.path.join(class_dir, f"{idx}.jpg"))



# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# save_images(trainset, train_dir)



# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# save_images(testset, test_dir)


## Load each jpg images from folders to tran and test data set of tensors.

def main():
    batch_size = 4

    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)


    net = Net()     

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(4):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    PATH = './cifar_net.pth'
    # Save both model and optimizer state_dict
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, PATH)
    print("model_saved")

if __name__ == '__main__':
    main()
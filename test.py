import torch
import torchvision.transforms as transforms
from architecture import Net
from PIL import Image

PATH = './cifar_net.pth'
net = Net()
checkpoint = torch.load(PATH)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

# OR
# net = Net()
# net.load_state_dict(torch.load(PATH, weights_only=True))
# net.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    # transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def preprocess_image(img_path):
    img = Image.open(img_path)
    img_tensor = transform(img)
    # img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

img_tensor = preprocess_image("./cifar10/test/0/3.jpg")

with torch.no_grad():
    outputs = net(img_tensor)
    _, predicted = torch.max(outputs, 1)

print("value = ", predicted.item())
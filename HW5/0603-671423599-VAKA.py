import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class InferenceDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if os.path.isfile(os.path.join(directory, fname))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')  # Convert grayscale images to RGB
        
        if self.transform:
            img = self.transform(img)
            
        return img, img_path

# class CustomImageDataset(Dataset):
#     def __init__(self, directory, transform=None):
#         self.directory = directory
#         self.image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if os.path.isfile(os.path.join(directory, fname))]
#         self.transform = transform

#         # Assuming the filenames are of the format "classname_XXXX.jpg"
#         # Extract unique class names and create a mapping to integer labels
#         self.class_names = sorted({fname.split('_')[0] for fname in os.listdir(directory)})
#         self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         img = Image.open(img_path).convert('RGB')  # Convert grayscale images to RGB
        
#         # Extract class name from the filename and convert to an integer label
#         class_name = os.path.basename(img_path).split('_')[0]
#         label = self.class_to_idx[class_name]
        
#         if self.transform:
#             img = self.transform(img)
            
#         return img, label
# Define the Net class just like you did in your training script
class Net(nn.Module):
    # ... [same as your definition above]
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 9)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load the saved model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
net.load_state_dict(torch.load('0602-671423599-VAKA.pth'))
net.eval()

# Transformation for inference (same as you used in training)
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image).unsqueeze(0).to(device)
    outputs = net(image)
    _, predicted = torch.max(outputs, 1)
    return classes[predicted[0]]

def get_true_label(filename, classes):
    for cls in classes:
        if cls in filename:
            return cls
    return None

if __name__ == '__main__':
    directory = os.path.dirname(os.path.abspath(__file__)) + '/geometry_dataset/output'  # same directory as training data
    dataset = InferenceDataset(directory, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    classes = sorted({fname.split('_')[0] for fname in os.listdir(directory)})
    int = 0
    results = {}
    correct = 0
    total = 0
    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            labels = [classes[idx] for idx in predicted]
            for p, label in zip(paths, labels):
                true_label = get_true_label(os.path.basename(p), classes)
                if true_label == label:
                    correct += 1
                total += 1
                results[p] = label
    
    # Print results
    for img_path, label in results.items():
        print(f"{os.path.basename(img_path)}: {label}")
    # for img_path, label in results.items():
    #     print(f"{os.path.basename(img_path)}: {label}")
    #         if filename.endswith(".png"):
    #             image_path = os.path.join(directory, filename)
    #             prediction = predict_image(image_path)
    #             print(f"{filename}: {prediction}")
    #             int+=1
    #             print(int)

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
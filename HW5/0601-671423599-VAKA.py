import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from PIL import Image
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split

classes = []

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if os.path.isfile(os.path.join(directory, fname))]
        self.transform = transform

        # Assuming the filenames are of the format "classname_XXXX.jpg"
        # Extract unique class names and create a mapping to integer labels
        self.class_names = sorted({fname.split('_')[0] for fname in os.listdir(directory)})
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')  # Convert grayscale images to RGB
        
        # Extract class name from the filename and convert to an integer label
        class_name = os.path.basename(img_path).split('_')[0]
        label = self.class_to_idx[class_name]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
def show_images_with_labels(loader, num_images=5):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images = images.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    fig, axs = plt.subplots(1, num_images, figsize=(12, 2))
    for i in range(num_images):
        image = images[i].cpu() / 2 + 0.5  # Unnormalize the image
        image = image.numpy().transpose((1, 2, 0))
        label = classes[labels[i]]
        pred_label = classes[predicted[i]]

        axs[i].imshow(image)
        axs[i].set_title(f'True: {label}\nPredicted: {pred_label}')
        axs[i].axis('off')
    plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
#Define the neural network
class Net(nn.Module):
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

    # def __init__(self, num_classes=9):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
    #     self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.fc1 = nn.Linear(32 * 32 * 32, 512)  # After two poolings, 128x128 becomes 32x32
    #     self.fc2 = nn.Linear(512, num_classes)
    #     self.dropout = nn.Dropout(0.25)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 32 * 32 * 32)
    #     x = F.relu(self.fc1(x))
    #     x = self.dropout(x)
    #     x = self.fc2(x)
    #     return x

    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    #     self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    #     self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    #     #self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     #self.fc1 = nn.Linear(256 * 8 * 8, 1024)
    #     self.fc1 = nn.Linear(128 * 16 * 16, 512)
    #     #self.fc2 = nn.Linear(1024, 512)
    #     self.fc2 = nn.Linear(512, 256)
    #     self.fc3 = nn.Linear(256, 9)
    #     self.dropout = nn.Dropout(0.5)
    #     self.batch_norm1 = nn.BatchNorm2d(32)
    #     self.batch_norm2 = nn.BatchNorm2d(64)
    #     self.batch_norm3 = nn.BatchNorm2d(128)
    #     #self.batch_norm4 = nn.BatchNorm2d(256)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
    #     x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
    #     x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
    #     #x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
    #     #x = x.view(-1, 256 * 8 * 8)
    #     x = x.view(-1, 128 * 16 * 16)
    #     x = F.relu(self.fc1(x))
    #     x = self.dropout(x)
    #     x = F.relu(self.fc2(x))
    #     x = self.dropout(x)
    #     # x = F.relu(self.fc3(x))
    #     # x = self.dropout(x)
    #     x = self.fc3(x)
    #     return x
    
def calculate_accuracy(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (100 * correct / total)

if __name__=='__main__':
    net = Net()

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # print(np.shape(np.array(Image.open('/home/vishalvaka/testing/geometry_dataset/output/Circle_0a0b51ca-2a86-11ea-8123-8363a7ec19e6.png'))))
    # train_data = np.random.rand(8000, 200, 200, 3)

    # Define transformations (you can modify as needed)
    # data_transforms = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # transforms.Resize((128, 128)),
    # transforms.ToTensor()])

    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Initialize the dataset and dataloader
    dataset = CustomImageDataset(os.path.dirname(os.path.abspath(__file__)) + '/geometry_dataset/output', transform=data_transforms)
    #dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    #print(dataloader)
    # Example: iterate through the DataLoader
    # for images, labels in dataloader:
    #     print(labels)
    #     class_names = [dataset.class_names[label] for label in labels]
    #     print(class_names)
    
    # train_len = int(0.8 * len(dataset))
    # test_len = len(dataset) - train_len

    # train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    labels = [dataset[i][1] for i in range(len(dataset))]

    # Create a stratified split
    train_indices, test_indices, _, _ = train_test_split(
        list(range(len(dataset))), 
        labels, 
        test_size=0.2, 
        stratify=labels
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers = 8)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers = 8)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = optim.RMSprop(net.parameters(), lr = 0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    classes = dataset.class_names
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(15):  # Change the number of epochs as needed
        star_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
            # scheduler.step(loss)

            running_loss += loss.item()

        with torch.no_grad():
            test_loss = 0.0
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss_val = criterion(outputs, labels)
                test_loss += loss_val.item()


        train_accuracy = calculate_accuracy(train_loader)
        test_accuracy = calculate_accuracy(test_loader)
        end_time = time.time()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}%, Time Taken: {end_time - star_time}')
        train_losses.append(running_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)


    print('Finished Training')
    torch.save(net.state_dict(), '0602-671423599-VAKA.pth')

    # Plot training and test loss
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Epoch vs Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and test accuracy
    plt.figure()
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()


    show_images_with_labels(test_loader, num_images=5)
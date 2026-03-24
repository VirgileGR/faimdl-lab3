import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.customnet import CustomNet
import wandb

wandb.init(
    project="faimdl-lab3",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 64,
        "architecture": "CustomNet"
    }
)


# Define transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to fit the input dimensions of the network
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load TinyImageNet
train_dataset = datasets.ImageFolder('./data/tiny-imagenet-200/train', transform=transform)
test_dataset = datasets.ImageFolder('./data/tiny-imagenet-200/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

# Test loop
# Modifie la définition de test pour accepter l'epoch
def test(epoch, model, test_loader, criterion): # Ajout de epoch ici
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss:.6f} Acc: {test_accuracy:.2f}%')
    
    # Correction des noms de variables ici
    wandb.log({"epoch": epoch, "test/accuracy": test_accuracy, "test/loss": test_loss})
    return test_accuracy

# --- LA BOUCLE DE RUN CORRIGÉE ---
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)
    # On teste à CHAQUE epoch pour avoir de beaux graphiques
    test_accuracy = test(epoch, model, test_loader, criterion) 
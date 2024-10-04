import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        
        # First hidden layer to second hidden layer
        self.layers.append(nn.Linear(hidden_sizes[0], hidden_sizes[1]))
        self.layers.append(nn.ReLU())
        
        # Second hidden layer to output layer
        self.layers.append(nn.Linear(hidden_sizes[1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Hyperparameters
input_size = 28 * 28  # MNIST images are 28x28
hidden_sizes = [128, 64]  # Sizes of hidden layers
output_size = 10  # 10 classes for digits 0-9
batch_size = 8
learning_rate = 0.001
num_epochs = 10
data_store = r"D:\git_repos\baseML\datastore"

# Data loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root=data_store, train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root=data_store, train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate model, loss function, and optimizer
model = MLP(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        images = images.view(-1, 28 * 28)  # Flatten the images

        # Forward pass
        outputs = model(images)
        print(f"outputs.shape= {outputs}")
        print(f"labels.shape= {labels}")
        
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # No gradients needed for evaluation
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')

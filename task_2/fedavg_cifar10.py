import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 5
NUM_ROUNDS = 5
CLIENT_EPOCHS = 2

server_model = models.resnet18(num_classes=10).to(device)

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

total_samples = len(trainset)
samples_per_client = total_samples // NUM_CLIENTS
client_datasets = []

indices = torch.randperm(total_samples).tolist()

for i in range(NUM_CLIENTS):
    start_idx = i*samples_per_client
    end_idx = start_idx+samples_per_client if i < NUM_CLIENTS-1 else total_samples
    client_indices = indices[start_idx:end_idx]
    client_datasets.append(Subset(trainset, client_indices))
    

client_loaders = [DataLoader(dataset, batch_size=64, shuffle=True) for dataset in client_datasets]
test_loader = DataLoader(testset, batch_size=64, shuffle=False)


def train_client(model, dataloader, epochs, lr):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"epoch {epoch+1}/{epochs}", leave=True, position=0)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="eval", leave=True, position=0):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    avg_loss = loss / len(dataloader)
    return accuracy, avg_loss


def fed_avg(server_model, client_models, client_sizes):
    """
       client_sizes: list of number of data points per client
       client_models: list of client models
    """
    server_dict = server_model.state_dict()
    
    for k in server_dict.keys():
        server_dict[k] = torch.zeros_like(server_dict[k], dtype=torch.float32) # fp32 long things
    
    total_data_points = sum(client_sizes)
    
    for i in range(len(client_models)):
        client_dict = client_models[i].state_dict()
        weight = client_sizes[i] / total_data_points
        
        for k in server_dict.keys():
            server_dict[k] += client_dict[k] * weight
            
    server_model.load_state_dict(server_dict)
    return server_model
    
def main():
    global server_model
    
    tqdm.write("clients:")
    for i, dataset in enumerate(client_datasets):
        tqdm.write(f"client {i}: {len(dataset)} samples")
        
    for round in tqdm(range(NUM_ROUNDS), desc="fl rounds"):
        tqdm.write(f"\nround {round+1}/{NUM_ROUNDS}")
        client_models = []
        client_sizes = []
        
        for i in range(NUM_CLIENTS):
            tqdm.write(f"training client {i}")
            
            client_model = models.resnet18(num_classes=10)
            client_model.load_state_dict(server_model.state_dict())
            client_model = client_model.to(device)
            
            train_loss = train_client(client_model, client_loaders[i], CLIENT_EPOCHS, lr=0.01)
            tqdm.write(f"client {i}: train loss: {train_loss:.4f}")
            
            client_models.append(client_model)
            client_sizes.append(len(client_datasets[i]))
        
        server_model = fed_avg(server_model, client_models, client_sizes)
        
        test_accuracy, test_loss = evaluate_model(server_model, test_loader)
        tqdm.write(f"test accuracy: {test_accuracy:.2f}, test loss: {test_loss:.4f}\n")
        
if __name__ == "__main__":
    main()
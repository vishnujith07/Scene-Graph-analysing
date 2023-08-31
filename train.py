import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.gnn_model import SceneGraphGNN
from utils.data_processing import SceneGraphDataset, collate_fn

def train():
    # Set up dataset, dataloader, and model
    train_dataset = SceneGraphDataset("data/train_data/")
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
    model = SceneGraphGNN(input_dim, hidden_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            graph_data, labels = batch
            optimizer.zero_grad()
            output = model(graph_data)
            loss = compute_loss(output, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

if __name__ == "__main__":
    train()

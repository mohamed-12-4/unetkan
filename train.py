from tqdm import tqdm
from loss import CombinedLoss
from torch import optim
import torch

def train_model(model, dataloader, val_dataloader, epochs, device):
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix({"Train Loss": loss.item()})

        # Validation
        val_loss = validate_model(model, val_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(dataloader):.4f}, Val Loss: {val_loss:.4f}")

# Validation Loop
def validate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    return val_loss / len(dataloader)

def save_model(model, path):
    torch.save(model.state_dict(), path)
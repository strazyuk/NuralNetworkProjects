import torch
import torch.nn as nn
import torch_optimizer as optim_look  # Using RAdam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------------
# Autoencoder with batch norm and deeper layers
# -------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def embed(self, x):
        return self.encoder(x)

# -------------------------------
# Hyperparameters
# -------------------------------
input_size = 784
latent_size = 64
learning_rate = 0.001
batch_size = 256
epochs = 20

# -------------------------------
# Dataset and Dataloader
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# Model, optimizer, scheduler
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(input_size, latent_size).to(device)
criterion = nn.SmoothL1Loss()  # Changed from MSELoss
optimizer = optim_look.RAdam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# -------------------------------
# Training Loop
# -------------------------------
best_loss = float('inf')
model.train()
for epoch in range(epochs):
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        recon = model(data)
        loss = criterion(recon, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_autoencoder.pth")

# -------------------------------
# Extract latent embeddings
# -------------------------------
model.load_state_dict(torch.load("best_autoencoder.pth"))
model.eval()

all_embeddings = []
all_labels = []
with torch.no_grad():
    for data, labels in train_loader:
        data = data.to(device)
        embeddings = model.embed(data)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

X = np.vstack(all_embeddings)
y = np.hstack(all_labels)

# -------------------------------
# Normalize embeddings
# -------------------------------
X_norm = normalize(X)

# -------------------------------
# Apply MiniBatch KMeans
# -------------------------------
kmeans = MiniBatchKMeans(n_clusters=10, batch_size=1024, random_state=42)
clusters = kmeans.fit_predict(X_norm)

# -------------------------------
# Evaluation: PDF style
# -------------------------------

# 1. Silhouette Score
sil_score = silhouette_score(X_norm, clusters)
print(f"\nSilhouette Score: {sil_score:.4f}")

# 2. Visual clustering in 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', s=10)
plt.title("Cluster Visualization (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()

# 3. Evaluate with Actual Labels (not as a metric, but visualization aid)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=10)
plt.title("Ground Truth Labels (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()

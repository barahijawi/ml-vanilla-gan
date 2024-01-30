import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the dataset
file_path = 'creditcard.csv'  # Update this with the path to your dataset
data = pd.read_csv(file_path)
X = data.drop('Class', axis=1)
y = data['Class']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled).float()
y_tensor = torch.tensor(y.values).float().view(-1, 1)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, X_tensor.shape[1]),  # Output size = number of features
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(X_tensor.shape[1], 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# Loss function
criterion = nn.BCELoss()

# Training loop
num_epochs = 100  # Number of epochs
for epoch in range(num_epochs):
    for i, (real_data, _) in enumerate(dataloader):
        batch_size = real_data.size(0)

        # Train Discriminator
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        discriminator.zero_grad()
        outputs_real = discriminator(real_data)
        loss_real = criterion(outputs_real, real_labels)

        z = torch.randn(batch_size, 100)
        fake_data = generator(z)
        outputs_fake = discriminator(fake_data.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        generator.zero_grad()
        outputs = discriminator(fake_data)
        loss_g = criterion(outputs, real_labels)
        loss_g.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

# Generate synthetic data
num_samples = 10000  # Number of synthetic samples to generate
z = torch.randn(num_samples, 100)
synthetic_data = generator(z).detach().numpy()

# Rescale the synthetic data to original scale
synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)

# Convert the generated data to a DataFrame
synthetic_df = pd.DataFrame(synthetic_data_rescaled, columns=X.columns)
synthetic_df.to_csv('generated_data.csv', index=False)
print("Generated data saved to generated_data_10k_1.csv")

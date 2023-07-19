from torch_utils import cluster_points
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
import os


print(f"Current dir: {os.getcwd()}")

if not os.getcwd().endswith('autoencoder-latent-space-visualization'):
    os.chdir('autoencoder-latent-space-visualization')
    print(f"Changed dir to: {os.getcwd()}")
dataset ='mnist' # 'mnist' or 'vessel'


# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x) # [bs, 2]
        assert encoded.dtype == torch.float32
        decoded = self.decoder(encoded)# [bs, 784]
        return decoded

# Hyperparameters
bottleneck_size = 2
learning_rate = 0.001
batch_size = 256
num_epochs = 50

early_stopping_epochs = 10
epoch_wo_improvement = 0
best_loss = np.inf

if dataset == 'mnist':
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
elif dataset == 'vessel':
    # Load the VESSEL patches dataset
    assert() # TODO
else:
    raise ValueError(f"Invalid dataset {dataset}. Choose 'mnist' or 'vessel'.")
    
print(f"Train dataset data type: {train_dataset.data.dtype}") # torch.uint8
print(f"Train dataset shape: {train_dataset.data.shape}") # [60000, 28, 28]
print(f"Test dataset shape: {test_dataset.data.shape}") # [10000, 28, 28]

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the autoencoder
autoencoder = Autoencoder(bottleneck_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Check if the model already exists and load it
print(f'Checking for existing model among in {os.getcwd()}...')
if os.path.exists(f'model_{dataset}.pth'):
    autoencoder.load_state_dict(torch.load(f'model_{dataset}.pth'))
else:
    # Train the autoencoder
    for epoch in range(num_epochs):
        print('\nTraining the autoencoder...')
        total_loss = 0
        for data, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            
            if dataset == 'mnist':
                #print(data.shape) # [256, 1, 28, 28]
                data = data.view(-1, 784) # Flatten to [256, 784]
            elif dataset == 'vessel':
                assert() # TODO
            
            recon = autoencoder(data)
            loss = criterion(recon, data)
            
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_tr_loss = total_loss / len(train_loader)

        with torch.no_grad():
            print('Validating the autoencoder...')
            total_loss = 0
            for data, _ in tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
                
                if dataset == 'mnist':
                    data = data.view(-1, 784)
                elif dataset == 'vessel':
                    assert()
                
                recon = autoencoder(data)
                total_loss += criterion(recon, data).item()
            
            avg_loss = total_loss / len(test_loader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                epoch_wo_improvement = 0
                torch.save(autoencoder.state_dict(), f'model_{dataset}.pth')
            else:
                epoch_wo_improvement += 1
            
            if epoch_wo_improvement == early_stopping_epochs:
                print(f'No improvement for {early_stopping_epochs} epochs. Training stopped.')
                break
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_tr_loss:.4f}, Validation Loss: {avg_loss:.4f}')

# Prepare the test data
if dataset == 'mnist':
    x_test = test_dataset.data.float() / 255.0 # [10000, 28, 28]
    x_test = x_test.view(x_test.size(0), -1) # Flatten to [10000, 784]
elif dataset == 'vessel':
    assert ()

# Encode the test images
with torch.no_grad():
    encoded_imgs = autoencoder.encoder(x_test).numpy()

#print(f'Encoded images shape: {encoded_imgs.shape}') # [10000, 2]


# Example usage
data = encoded_imgs

clustering_results = cluster_points(data, method='kmeans', max_clusters=5) # or method=['kmeans', 'dbscan', 'agglomerative']

print("Cluster labels:", clustering_results['labels'])
print("Optimal number of clusters:", clustering_results['optimal_clusters'])
print("Silhouette score:", clustering_results['silhouette_score'])

# Plot the encoded images
fig, ax = plt.subplots(1, 2)
ax[0].scatter(encoded_imgs[:, 0], encoded_imgs[:, 1],
              #c=test_dataset.targets,
              c=clustering_results['labels'],
              s=8, cmap='tab10')


# Create a grid of points in the latent space
grid_size = 15
grid_x = np.linspace(-2, 2, grid_size)
grid_y = np.linspace(-2, 2, grid_size)
grid = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)

# Decode the grid points
with torch.no_grad():
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    decoded_grid = autoencoder.decoder(grid_tensor).numpy() # [225, 784]

# Reshape and display the decoded grid
if dataset == 'mnist':
    decoded_grid = decoded_grid.reshape(-1, 28, 28)
elif dataset == 'vessel':
    assert()
    
ax[1].imshow(np.block(list(decoded_grid)), cmap='gray')


def onclick(event):
    if event.inaxes == ax[0]:
        # Get the clicked coordinates in the scatter plot
        ix, iy = event.xdata, event.ydata

        # Encode the clicked point
        with torch.no_grad():
            latent_vector = torch.tensor([[ix, iy]])
            #Convert to float 32
            latent_vector = latent_vector.type(torch.float32)
            
            #print(f"Latent vector: {latent_vector}") # Point coordinates in the latent space
            #print(f"Latent vector shape: {latent_vector.shape}") # [1, 2]
            #print(f"Latent vector type: {latent_vector.dtype}") # float32
            assert latent_vector.dtype == torch.float32
            if dataset == 'mnist':
                decoded_img = autoencoder.decoder(latent_vector).numpy().reshape(28, 28)
            elif dataset == 'vessel':
                assert()
        
        # Update the displayed decoded image
        ax[1].imshow(decoded_img, cmap='gray')
        plt.draw()

        # Save the figure
        plt.savefig('latent_space.png')

# Connect the onclick event
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()



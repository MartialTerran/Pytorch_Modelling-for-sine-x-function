# Model to generate Sine(x) values.  A Pytorch version of the original published by 
#   To do: add     self.dropout = nn.Dropout(0.1)  # Dropout layer
#Created on Thu Oct 28 12:18:37 2021
# @author: aman (translated to pytorch by Gemini) then Tested/Debugged/Evaluated by MartialTerran BSEE

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Set seed for experiment reproducibility
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

# Number of sample datapoints
SAMPLES = 400
# Generate a uniformly distributed set of random numbers in the range from
# 0 to 2π, which covers a complete sine wave oscillation
x_values = np.random.uniform(
    low=0, high=2*math.pi, size=SAMPLES).astype(np.float32)

print(x_values)
# Shuffle the values to guarantee they're not in order
np.random.shuffle(x_values)

# Calculate the corresponding sine values
y_values = np.sin(x_values).astype(np.float32)

# Plot our data. The 'b.' argument tells the library to print blue dots.
plt.plot(x_values, y_values, 'b.')
plt.show()

# Add a small random number to each y value
y_values += 0.01 * np.random.randn(*y_values.shape)

# Plot our data
plt.plot(x_values, y_values, 'b.')
plt.show()

# We'll use 60% of our data for training and 20% for testing. The remaining 20%
# will be used for validation. Calculate the indices of each section.
TRAIN_SPLIT =  int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

# Use np.split to chop our data into three parts.
# The second argument to np.split is an array of indices where the data will be
# split. We provide two indices, so the data will be divided into three chunks.
x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

# Double check that our splits add up correctly
assert (x_train.size + x_validate.size + x_test.size) ==  SAMPLES

# Plot the data in each partition in different colors:
plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_test, y_test, 'r.', label="Test")
plt.plot(x_validate, y_validate, 'y.', label="Validate")
plt.legend()
plt.show()

# Convert data to PyTorch tensors
x_train_tensor = torch.from_numpy(x_train).unsqueeze(1)
y_train_tensor = torch.from_numpy(y_train).unsqueeze(1)
x_test_tensor = torch.from_numpy(x_test).unsqueeze(1)
y_test_tensor = torch.from_numpy(y_test).unsqueeze(1)
x_validate_tensor = torch.from_numpy(x_validate).unsqueeze(1)
y_validate_tensor = torch.from_numpy(y_validate).unsqueeze(1)

# Define the PyTorch model
class SineModel(nn.Module):
    def __init__(self):
        super(SineModel, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = SineModel()

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001) # lr = learning rate
loss_fn = nn.MSELoss()

# Train the model
epochs = 500
batch_size = 64
train_losses = []
val_losses = []

for epoch in range(1, epochs + 1):
    # Training
    model.train()
    permutation = torch.randperm(x_train_tensor.size()[0])
    
    epoch_train_loss = 0.0
    for i in range(0, x_train_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        x_batch, y_batch = x_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * x_batch.size(0) 
    
    train_losses.append(epoch_train_loss/x_train_tensor.size(0))

    # Validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(x_validate_tensor)
        val_loss = loss_fn(y_val_pred, y_validate_tensor).item()
        val_losses.append(val_loss)

    if epoch % 100 == 0:
      print(f'Epoch {epoch}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}')

# Draw a graph of the loss, which is the distance between
# the predicted and actual values during training and validation.

epochs_range = range(1, len(train_losses) + 1)

# Exclude the first few epochs so the graph is easier to read
SKIP = 0

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)

plt.plot(epochs_range[SKIP:], train_losses[SKIP:], 'g.', label='Training loss')
plt.plot(epochs_range[SKIP:], val_losses[SKIP:], 'b.', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)

# Draw a graph of mean absolute error, which is another way of
# measuring the amount of error in the prediction.
# We need to recalculate MAE because it was not stored during training
model.eval()
with torch.no_grad():
  train_mae = torch.mean(torch.abs(model(x_train_tensor) - y_train_tensor)).item()
  val_mae = torch.mean(torch.abs(model(x_validate_tensor) - y_validate_tensor)).item()

plt.plot([epochs_range[-1]], [train_mae], 'g.', label='Training MAE')
plt.plot([epochs_range[-1]], [val_mae], 'b.', label='Validation MAE')
plt.title('Training and validation mean absolute error')
plt.xlabel('Epochs (only final epoch shown for MAE)')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate and print the loss on our test dataset
model.eval()
with torch.no_grad():
    y_test_pred_tensor = model(x_test_tensor)
    test_loss = loss_fn(y_test_pred_tensor, y_test_tensor).item()
    test_mae = torch.mean(torch.abs(y_test_pred_tensor - y_test_tensor)).item()

print(f'Test Loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')

# Make predictions based on our test dataset
# y_test_pred has already been computed above when calculating test loss/MAE

# Graph the predictions against the actual values
plt.clf()
plt.title('Comparison of predictions and actual values')
plt.plot(x_test, y_test, 'b.', label='Actual values')
plt.plot(x_test, y_test_pred_tensor.detach().numpy(), 'r.', label='PyTorch predicted')
plt.legend()
plt.show()

# PyTorch does not have a direct equivalent to TensorFlow Lite built-in.
# For mobile/embedded

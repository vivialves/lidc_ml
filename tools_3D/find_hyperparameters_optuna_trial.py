import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import plotly.io as pio


# --- 1. Define your PyTorch Model ---
class SimpleCNN(nn.Module):
    def __init__(self, trial):
        super(SimpleCNN, self).__init__()

        # --- Hyperparameters to tune for the CNN architecture ---
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
        channels = [trial.suggest_int(f'n_channels_l{i}', 16, 64, step=16) for i in range(n_conv_layers)]
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
        activation_fn_name = trial.suggest_categorical('activation_fn', ['ReLU', 'LeakyReLU'])

        # Select activation function
        if activation_fn_name == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_fn_name == 'LeakyReLU':
            self.activation = nn.LeakyReLU()

        self.conv_layers = nn.ModuleList()
        in_channels = 1 # FashionMNIST images are grayscale (1 channel)

        for i in range(n_conv_layers):
            self.conv_layers.append(
                nn.Conv2d(in_channels, channels[i], kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
            )
            self.conv_layers.append(self.activation) # Use the suggested activation
            self.conv_layers.append(nn.MaxPool2d(2))
            in_channels = channels[i]

        # Calculate input features for the first fully connected layer
        dummy_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            x = dummy_input
            for layer in self.conv_layers:
                x = layer(x)
        fc_input_features = x.view(x.size(0), -1).size(1)

        # Fully connected layer
        n_fc_units = trial.suggest_int('n_fc_units', 32, 256, step=32)
        self.fc = nn.Linear(fc_input_features, n_fc_units)
        self.relu_fc = nn.ReLU() # Keeping this ReLU for consistency in output layer path

        # Dropout layer (new parameter)
        self.dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
        self.dropout = nn.Dropout(self.dropout_rate)

        # Output layer
        self.output_layer = nn.Linear(n_fc_units, 10) # 10 classes for FashionMNIST

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1) # Flatten for FC layer
        x = self.relu_fc(self.fc(x))
        x = self.dropout(x) # Apply dropout
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

# --- Data Preparation ---
def get_fashion_mnist_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# --- 2. Optuna Objective Function ---
def objective(trial):
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameters to tune for training ---
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 5, 15) # Limit epochs for faster example

    # New training parameters
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True) # L2 regularization
    gradient_clipping_norm = trial.suggest_float('gradient_clipping_norm', 0.1, 5.0, step=0.1) # Gradient clipping

    # Get data loaders
    train_loader, valid_loader = get_fashion_mnist_loaders(batch_size)

    # Initialize model with suggested architecture hyperparameters
    model = SimpleCNN(trial).to(device)

    # Initialize optimizer with weight_decay
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # Apply gradient clipping (new)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)

            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = correct / total

        # Optuna Pruning: Report intermediate value and check if trial should be pruned
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

# --- 3. Run an Optuna Study ---
if __name__ == '__main__':
    # Create a study object and optimize the objective function.
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

    print("Starting Optuna optimization with new parameters...")
    study.optimize(objective, n_trials=50, timeout=900) # Increased timeout slightly

    # --- 4. Analyze Results ---
    print("\nOptimization finished!")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len(pruned_trials)}")
    print(f"Number of complete trials: {len(complete_trials)}")

    print("\nBest trial:")
    trial = study.best_trial

    print(f"  Value (validation accuracy): {trial.value:.4f}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # --- 5. Visualization (requires 'plotly' and 'kaleido' for static images) ---
    try:
        from optuna.visualization import plot_optimization_history
        from optuna.visualization import plot_param_importances
        from optuna.visualization import plot_slice
        
        pio.renderers.default = "browser"
        
        fig_history = plot_optimization_history(study)
        fig_history.show()

        fig_importances = plot_param_importances(study)
        fig_importances.show()

        fig_slice = plot_slice(study)
        fig_slice.show()

    except ImportError:
        print("\nPlotly and Kaleido are not installed. Skipping visualization.")
        print("Install them with: pip install plotly kaleido")
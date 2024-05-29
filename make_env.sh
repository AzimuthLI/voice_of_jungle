#!/bin/bash

# Ensure the script is executed with superuser privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 
   exit 1
fi

# Update pip to the latest version
echo "Updating pip..."
pip install --upgrade pip


# List of packages to be installed
packages=(
    matplotlib
    seaborn
    librosa
    transformers
    scikit-learn
    wandb
    tqdm
    ipykernel
    jupyter
    ipywidgets
    timm
)

# Install packages using pip
echo "Installing packages..."
for package in "${packages[@]}"; do
    echo "Installing $package..."
    pip install "$package" -q 
done

echo "Installation complete."
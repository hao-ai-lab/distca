#!/bin/bash

# Deploy all Modal scripts in the modal-scripts directory
for script in modal-scripts/*.py; do
    echo "Deploying $script..."
    modal deploy "$script"
done

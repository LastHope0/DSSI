#!/bin/bash

# Define the path to your virtual environment
venv_path=".venv"

# Check if the virtual environment exists
if [ -d "$venv_path" ]; then
    echo "Activating virtual environment..."
    source "$venv_path/bin/activate"
else
    echo "Virtual environment not found at $venv_path."
    echo "Please create and activate your virtual environment first."
    exit 1
fi

# Run your Python script (main.py)
echo "Running main.py..."
XDG_SESSION_TYPE=x11 python3 main.py

# Deactivate the virtual environment
echo "Deactivating virtual environment..."
deactivate

echo "Script execution complete."

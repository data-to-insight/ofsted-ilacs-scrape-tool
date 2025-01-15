#!/bin/bash

# Install additional system-level dependencies/packages
sudo apt-get update    
python3 -m pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt


                                     
# Install the Python extension for Visual Studio Code
code --install-extension ms-python.python --force


#!/bin/bash

# Install additional system-level dependencies/packages
sudo apt-get update    

# Install Python dependencies
pip install -r requirements.txt


                                     
# Install the Python extension for Visual Studio Code
code --install-extension ms-python.python --force


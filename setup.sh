#!/bin/bash

echo "Setting up Whisper Transcriber for Ubuntu..."

# Update package list
echo "Updating package list..."
sudo apt update

# Install system dependencies for audio
echo "Installing system dependencies..."
sudo apt install -y python3-pip python3-tk portaudio19-dev python3-pyaudio

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user -r requirements.txt

echo ""
echo "Setup complete! You can now run the application with:"
echo "python3 whisper_transcriber.py"
echo ""
echo "Note: The first time you load the model, it will download the Whisper tiny model (~39MB)" 
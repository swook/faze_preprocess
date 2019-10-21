#!/bin/bash

# Install Python Dependencies
pip3 install --user -r requirements.txt

# Download required additional labels
wget -N https://ait.ethz.ch/projects/2019/faze/downloads/preprocessing/MPIIFaceGaze_supplementary.h5
wget -N https://ait.ethz.ch/projects/2019/faze/downloads/preprocessing/GazeCapture_supplementary.h5

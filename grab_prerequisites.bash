#!/bin/bash

# Install Python Dependencies
pip3 install --user -r requirements.txt

# Download required additional labels
wget -N https://files.ait.ethz.ch/projects/faze/preprocessing/MPIIFaceGaze_supplementary.h5
wget -N https://files.ait.ethz.ch/projects/faze/preprocessing/GazeCapture_supplementary.h5

# Download eos files for pre-processing for ST-ED
mkdir -p ./eos/
cd ./eos/
wget -N https://github.com/patrikhuber/eos/raw/master/share/sfm_shape_3448.bin

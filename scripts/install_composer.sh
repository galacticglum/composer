#!/bin/bash
#
#
# An install script for composer (https://github.com/galacticglum/composer).
# Run with: bash install_composer.sh

# Exit on error
set -e

finish() {
    if (( $? != 0)); then
        echo ""
        echo "=================================================="
        echo "Installation did not finish successfully."
        echo "For assistance, post an issue on the project page."
        echo "https://github.com/galacticglum/composer"
        echo "=================================================="
        echo ""
    fi
}
trap finish EXIT

# For printing error messages
print_error() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
    exit 1
}

# Check which operating system
if [[ "$(uname)" == "Darwin" ]]; then
	echo 'Mac OS Detected'
	readonly OS='MAC'
elif [[ "$(uname)" == "Linux" ]]; then
	echo 'Linux OS Detected'
	readonly OS='LINUX'
else
	print_error 'Detected neither OSX or Linux Operating System'
fi

# Check if conda already installed
if [[ ! $(which conda) ]]; then
	print_error "- conda not detected! Composer requires conda."
else
	echo "- conda detected"
fi

# Set up the composer environment
echo "- setting up composer environment"

if [[ $(conda info --envs | grep "composer" | awk '{print $1}') != "composer" ]]; then
    read -p "conda environment of name 'composer' already exists! Would you like to remove this environment and continue (Y/[N])? " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] then
        exit
    fi

    echo "- removing old conda environment"
    if [[ $(conda info --envs | grep "*" | awk '{print $1}') == "composer" ]]; then
        conda deactivate
    fi 

    conda remove --name composer --all -y -q
fi

conda env create -f environment.yml -q

set +e
conda activate "composer"
set -e
if [[ $(conda info --envs | grep "*" | awk '{print $1}') != "composer" ]]; then
    echo "$(conda info --envs | grep "*" | awk '{print $1}') # conda output!"
    print_error 'Did not successfully activate the composer conda environment'
fi

pip install --editable .

echo ""
echo "=============================="
echo "Composer Install Success!"
echo ""
echo "NOTE:"
echo "For changes to become active, you will need to open a new terminal."
echo ""
echo "To just uninstall the environment run:"
echo "  conda remove -n composer --all"
echo ""
echo "To run composer, activate your environment:"
echo "  source activate composer"
echo ""
echo "You can deactivate when you're done:"
echo "  source deactivate"
echo "=============================="
echo ""
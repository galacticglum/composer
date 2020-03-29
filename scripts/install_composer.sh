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
    readonly MINICONDA_SCRIPT='Miniconda3-latest-MacOSX-x86_64.sh'
elif [[ "$(uname)" == "Linux" ]]; then
    echo 'Linux OS Detected'
    readonly OS='LINUX'
    readonly MINICONDA_SCRIPT='Miniconda3-latest-Linux-x86_64.sh'
else
    print_error 'Detected neither OSX or Linux Operating System'
fi

# Check if anaconda already installed
if [[ ! $(which conda) ]]; then
    echo ""
    echo "==========================================="
    echo "anaconda not detected, installing miniconda"
    echo "==========================================="
    echo ""
    readonly CONDA_INSTALL="/tmp/${MINICONDA_SCRIPT}"
    readonly CONDA_PREFIX="${HOME}/miniconda3"
    curl "https://repo.continuum.io/miniconda/${MINICONDA_SCRIPT}" > "${CONDA_INSTALL}"
    bash "${CONDA_INSTALL}" -p "${CONDA_PREFIX}"
    # Modify the path manually rather than sourcing .bashrc because some .bashrc
    # files refuse to execute if run in a non-interactive environment.
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    if [[ ! $(which conda) ]]; then
      print_error 'Could not find conda command. conda binary was not properly added to PATH'
    fi
else
    echo ""
    echo "========================================="
    echo "anaconda detected, skipping conda install"
    echo "========================================="
    echo ""
fi

# Set up the composer environment
echo ""
echo "================================="
echo "setting up composer environment"
echo "================================="
echo ""

conda create -n "composer" python=3.7

set +e
conda activate "composer"
set -e
if [[ $(conda info --envs | grep "*" | awk '{print $1}') != "composer" ]]; then
  print_error 'Did not successfully activate the composer conda environment'
fi

pip install --editable ../

echo ""
echo "=============================="
echo "Composer Install Success!"
echo ""
echo "NOTE:"
echo "For changes to become active, you will need to open a new terminal."
echo ""
echo "For complete uninstall, remove the installed anaconda directory:"
echo "rm -r ~/miniconda2"
echo ""
echo "To just uninstall the environment run:"
echo "conda remove -n composer --all"
echo ""
echo "To run composer, activate your environment:"
echo "source activate composer"
echo ""
echo "You can deactivate when you're done:"
echo "source deactivate"
echo "=============================="
echo ""
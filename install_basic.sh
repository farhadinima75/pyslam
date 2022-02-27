#!/usr/bin/env bash

#N.B: this install script allows you to run main_vo.py and the test scripts 
# echo "usage: ./${0##*/} <INSTALL_PIP3_PACKAGES> <INSTALL_CPP>"   # the arguments are optional 

#set -e

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

export INSTALL_PIP3_PACKAGES=1   # install pip3 packages by default 
if [ $# -ge 1 ]; then
    # check optional argument 
    INSTALL_PIP3_PACKAGES=$1
    echo INSTALL_PIP3_PACKAGES: $INSTALL_PIP3_PACKAGES
fi

export INSTALL_CPP=1   # install cpp by default 
if [ $# -ge 2 ]; then
    # check optional argument 
    INSTALL_CPP=$2
    echo INSTALL_CPP: $INSTALL_CPP
fi

# ====================================================

echo `pwd`

# install system packages 
. install_system_packages.sh     # use . in order to inherit python env configuration 

# install pip3 packages 
# N.B.: install_pip3_packages script can be skipped if you intend to use a virtual python environment 
if [ $INSTALL_PIP3_PACKAGES -eq 1 ]; then
    echo 'installing pip3 packages'
    ./install_pip3_packages.sh   
fi 

# set up git submodules  
./install_git_modules.sh 

# build and install cpp stuff 
# N.B.: install_cpp script can be skipped here if you intend to use a virtual python environment 
#       but it must be then called within your virtual python environment in order to properly install libs 
if [ $INSTALL_CPP -eq 1 ]; then
    ./install_cpp.sh                
fi 




#set -e
. bash_utils.sh 
# ====================================================
print_blue '================================================'
print_blue "Building Thirdparty"
print_blue '================================================'
STARTING_DIR=`pwd`  # this should be the main folder directory of the repo
# ====================================================
# N.B.: this script requires that you have first run:
#./install_basic.sh 
# ====================================================
if [[ -z "${USE_PYSLAM_ENV}" ]]; then
    USE_PYSLAM_ENV=0
fi
if [ $USE_PYSLAM_ENV -eq 1 ]; then
    . pyenv-activate.sh
fi  
# ====================================================
# check if we have external options
EXTERNAL_OPTION=$1
if [[ -n "$EXTERNAL_OPTION" ]]; then
    echo "external option: $EXTERNAL_OPTION" 
fi
if [[ -n "$WITH_PYTHON_INTERP_CHECK" ]]; then
    echo "WITH_PYTHON_INTERP_CHECK: $WITH_PYTHON_INTERP_CHECK " 
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DWITH_PYTHON_INTERP_CHECK=$WITH_PYTHON_INTERP_CHECK"
fi
# ====================================================
CURRENT_USED_PYENV=$(get_virtualenv_name)
print_blue "currently used pyenv: $CURRENT_USED_PYENV"

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/orbslam2_features ..."
cd thirdparty/orbslam2_features
. build.sh $EXTERNAL_OPTION
cd $STARTING_DIR

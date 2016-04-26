# Dependencies and installation: LS5

Install python3:

    cd ~
    wget https://www.python.org/ftp/python/3.5.1/Python-3.5.1.tar.xz
    unxz Python-3.5.1.tar.xz
    tar -xvf Python-3.5.1.tar
    cd Python-3.5.1
    ./configure --prefix=$HOME/python3.5
    make
    make install

Add to .bashrc:

    export PATH=$HOME/python3.5/bin
    export PYTHONPATH=$HOME/python3.5/lib/python3.5/site-packages:$PYTHONPATH
    export CPATH=$HOME/local:$CPATH
    export LIBRARY_PATH=$HOME/local:$LIBRARY_PATH

Restart bash. Get virtualenv (not required due to installation in $HOME/python3.5):

    easy_install-3.5 virtualenv

Get libyaml and make:

    cd ~
    wget http://pyyaml.org/download/libyaml/yaml-0.1.5.tar.gz
    cd yaml-0.1.5
    ./configure --prefix=$HOME/local
    make
    make install

Set up virtualenv (not required due to installation in $HOME/python3.5):

    cd ~/tmd
    virtualenv -p python3 tmd_env
    source tmd_env/bin/activate

When finished with tmd, can use `deactivate` to leave virtualenv.
Install tmd:

    python3 setup.py develop

# Dependencies and installation: local

Uses pyyaml which requires libyaml:

    sudo apt-get install libyaml-dev

Uses [ASE](https://wiki.fysik.dtu.dk/ase/index.html) and the [2D materials repository](https://cmr.fysik.dtu.dk/c2dm/c2dm.html).

Create a virtualenv:

    sudo apt-get install python-virtualenv
    virtualenv -p python3 tmd_env
    source tmd_env/bin/activate

Install:

    python3 setup.py install

or:

    python3 setup.py develop

For making plots, need cwannier:

    git submodule init
    git submodule update
    cd cwannier
    git submodule init
    git submodule update
    cd ctetra
    make
    cd ..
    make



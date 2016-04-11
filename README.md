# Dependencies and installation

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

# Brewster Development Branch
A spectral inversion code for retrieval analysis of emission spectra from brown dwarfs and giant planets.
## How to Use this Branch
1. Clone the repository
2. Use `git checkout dev` to check out this branch
   
If you want to **create a fork** of Brewster_V2 to then use this branch, create a fork and **uncheck** "Copy the `main` branch only" 
1. `git clone` forked_repository_url
2. `cd brewster_v2`
3. `git add upstream https://github.com/fwang23nh/brewster_v2.git` (allows you to fetch any updates)
4. `git checkout dev`


## What's needed to run the code:
- Everything in this repository
- Line lists (Dir: /Linelists/)
- Clouds (.mieff files, Dir: /Clouds/)
- Atmospheric Models (Dir: Brewster/Data *Recommended, can live anywhere on your machine)

Environment Requirements:
- gFortran compiler (https://gcc.gnu.org/wiki/GFortranBinaries or https://hpc.sourceforge.net)

## Current Installation Process for Brewster
1. If you do not have a fortran compiler on your machine, get the proper version of the gfortran compiler from https://gcc.gnu.org/wiki/GFortranBinaries or https://hpc.sourceforge.net.
2. Git clone or fork the directory and place wherever your code lives. (If you plan to make any changes to the code you might like to incorporate into the master branch via a pull request, you should probably fork it.)
3. To keep various package versions clean and organized, you should create a Python 3 environment, ideally Python <=3.11, as that is the version that will be needed to run the code on a cluster.
4. `pip install --upgrade pip && pip install /path/to/brewster`
5. You will need to get access to the Linelist and Clouds folders that are shared via Dropbox. Place the Clouds and Linelists folders one directoy above the data directory from this repository 
```bash
├── Clouds
├── brewster_v2
│   └── data
└── Linelists
 ```
6. To test that everything is properly installed, run the check-brewster command, by typing `check-brewster` in the terminal making sure you are in the brewster working directory specified above

Please post any problems to the issue tracker.


## Developer notes from @blackwer
Since this is a hybrid python-fortran code, getting editable pip installs is somewhat nuanced. See the discussion at
https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#editable-installs

I have had good luck with the experimental editable-rebuild option. On a fresh python instance, it will rebuild
the fortran objects automatically on import of any of the modules.

```
pip install --upgrade pip
pip install scikit-build-core numpy
cd /path/to/brewster
pip install --no-build-isolation --config-settings=editable.rebuild=true -Cbuild-dir=build -ve .
```

## Nick's Notes

**Environment setup:**

```
conda create -n brewster -c conda-forge python numpy scikit-build cmake ninja pip astropy corner emcee scipy jupyter schwimmbad
conda activate brewster
```

**Option 1:**

```
python -m pip install --no-deps --no-build-isolation . -v
```

**Option 2:**

```
mkdir build
cd build
cmake ..
cmake --build . -j && cmake --install .
```

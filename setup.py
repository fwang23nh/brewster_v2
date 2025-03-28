from skbuild import setup

setup(
    name="brewster",
    packages=['brewster'],
    python_requires='>=3.6',
    version="2.0.0",
    install_requires=['numpy'], 
    cmake_args=[]
)

==========================
Demo of a novel controller
==========================

Overview
--------

The controller should be able to learn how to optimally control a given system with an unknown and also slowly adapting second controller.

Keywords
--------

Optimal Control, Differential Games, Identification, Neural Networks, DDPG

Installation and Usage
----------------------

Installation:

- Install Miniconda (Python 3.6, 64-bit) from official website: https://conda.io/miniconda.html
- Use Anaconda Prompt to:
- Use the command "cd" to go into the folder which contains environment.yml
- Type "conda env create" and Enter ->  Creates conda environment and installs all dependencies including python

Usage:

- Use the command "cd" to go into the folder which contains environment.yml (if not already there)
- Type "conda activate pal" and Enter -> So that executing a python script uses the dependencies that you just installed
- Type "python -m pal.main" -> Executes the file test_n_seconds.py in folder test (as a module so it finds the other Classes and Files inside the package)

What it does (when executing main):

- Simulates the training of two PALs
- Plots the result
- Saves the result into a file as soon as the plot is closed

Hints:

- To change code, an editor/IDE like Notepad++, PyCharm or Sublime Text is recommended so the line breaks will be interpreted correctly
- To Configure the Python Interpreter in PyCharm: Open the folder "controller" with PyCharm -> Settings -> Project-Interpreter -> Gear Wheel icon -> Add -> Conda Environment -> Existing -> Path to environment specific python.exe (see next bullet point)
- If you installed Miniconda for "Just this User" the environment specific python.exe a.k.a. Interpreter are located in the folder C:\Users\XXXXX\AppData\Local\Continuum\Miniconda3\envs\ENVNAME or ...\Local\conda\envs\ENVNAME. The Hidden Folder AppData can be opened by typing %AppData% into the navigation field of the Explorer and going one level of folders back up. This path can then be copied into the "File Open" Dialog for choosing the interpreter.
- The Warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use...." can be ignored if performance is good enough. Otherwise: See next point
- To get better performance you can compile tensorflow from Source to use all advanced Processor instructions that are supported like AVX, SSE etc. You then have to do "conda uninstall tensorflow tensorflow_base" before installing tensorflow with pip as mentioned in the following guide: (https://www.tensorflow.org/install/install_sources) -> takes at least 1-2 hours and only works good on Linux for now.
- Link to community pre-compiled tensorflow versions with different combinations of those instructions on windows: https://github.com/fo40225/tensorflow-windows-wheel
- The performance with GPU tensorflow was worse, because of the frequent and short calls to tf. CPU tf is recommended
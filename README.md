# A climate simulation output analysis tool

## Introduction

This project implements a method for detecing spatio-temporal consistent bias patterns in climate simulation output. Paper introducing the method can be found on <https://doi.org/10.5194/gmd-2019-107.>

## File Organization

- analyze.py is the main file. The core analysis logic is implemented in this file.
- BiasInstance.py and BiasFamily.py defines data structure for bias instance and bias family.
- PlotHeatMap.py defines a function to output a figure for a bias family, like Figure 3 in the paper.

## How to run it

The project can be run with python 3.5 and you may need to install some python packages, including Basemap, matplotlib, mlpy and netCDF4.

After the running environment has been set up, you can run the project with "python analyze.py".

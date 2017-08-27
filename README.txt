This file contains the instructions for how to run the code for Assignment 3

Reports:
1) jtay6-analysis.pdf - Assignment 3 report
2) jtay6-analysis1.pdf - Assignment 1 report

Code Files: 
1) parse.py - code to parse raw data files for this project
2) benchmark.py - Code to establish performance benchmark of neural network on these datasets
3) helpers.py - Miscellaneous helper functions	
4) clustering.py - Code for Clustering experiments
5) PCA.py, ICA.py, RP.py, RF.py - Code for PCA, ICA, Random Projections and Random Forest Feature Selection respectively	
6) Madelon tricks.py - Code for Part 7 of the assignment, to maximise accuracy on Madelon data set
7) plotter.py - Code to do plots

There are also a number of folders
1) BASE - Output folder for clustering on the original features
2) PCA - Output folder for experiments with PCA
3) ICA - Output folder for experiments with ICA
4) RP  - Output folder for experiments with Random Projections
5) RF - Output folder for experiments with Random Forest Feature Selection

Data Files
1) madelon_train.data, madelon_train.labels, madelon_valid.data,madelon_valid.labels - original Madelon data from the UCI ML repository

To run the experiments:
1) Generate the data files from the original data by running the parse.py code. This will also generate the appropriate directory structure for the rest of the experiments. 
2) Run clustering.py at the command line with the argument "BASE". eg: python clustering.py BASE
3) Run each of the scripts ICA, PCA, RP and RF in turn
4) Run clustering.py at the command line with the arguments ICA, PCA, RP or RF depending on the desired result set. The run.bat file will do this.

The data file code was written in Python 3.5, using Pandas 0.18.0 and sklearn 0.19.1

Plotting code was  written in Python 3.5, using Pandas 0.18.0 and matplotlib 1.5.1, Seaborn 0.7.1

Within the output folders, the data files are csv files. They are labelled by dataset and experiment type.



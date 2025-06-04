# alba_funcs
This is the python package containing the functions and imports necessary for the analysis in the paper.

## analysis.py
Functions related to the signal analysis and manipulation.

## io.py
Functions related to reading and writing data to and from files, such as reading roi.txt files.

## macros.py
Several python functions that create and run imageJ macros to view and manipulate scans and rois.

## paths.py
Functions for manipulating and querying paths.

## plotting.py
Funcions for plotting, one for plotting data colored by a sequence of values, and one for plotting an image stack.

## utils.py
Functions for general use.

# data
Contains aggregated data and some sample  files to allow running the notebook.

## alba_experiment_2022_roi_paths.json
Paths of the files containing energies of rois on scans from 2022 experiments.

## alba_experiment_2023_paths.json
Paths of the directories containing files of energies of rois on scans from 2023 experiments.

## full_data_{date}.json
Aggregated data collected from roi.txt files for analysis. Includes energies, absorptions and processed stepsn such as bgd, which is after background reduction, and right, left and sigma peak intensities.

## nqd{date}.csv
Manually annotated list of roi.txt files, indicating if the original is flipped (negative value), irrelevant (0) or too noisy (2).

# figures
Some of the figures as produced from the notebook

# notebooks

## analysis.ipynb
This is the notebook containing the necessary scripts to reproduce the resulting figures in the paper.

## Analysis Sections

1. Full Radiation Scans
    1. Crystal / vesicle pi prepeak ratio comparison
    2. Radiaiton damage and tilting effect on absrption spectra
    3. Sample crystal and vesicle absorption wave graphs
    4. Peak width comparison between crystals and vesicles
2. Minimal Radiation Scans
    1. More tilting effects, more dense degrees
    2. Minimal radiation - showing that the chosen scans do not damage the results
    3. Radiation damage, continuing the previous analysis with more data.
    

## Requirements

Besides the requirements in the requirements.txt file, another requirement is access to the data itself. without this, the analysis cannot run. The code itself is built fitting to the directory structure the data is currently in. if it is changed/copied/moved, this might need to be updated.

## License

[MIT](https://choosealicense.com/licenses/mit/)

# How to run
## Installation
    1. clone the repository using
```bash
git clone https://github.com/Gur-Lab-WIS/ALBA_2025_analysis
```
    2. make sure you have an environment with a jupyter kernel linked to it. 
    3. activate the environment on which to run, e.g.
```bash
conda activate myenv
```
    4. navigate to the cloned repo and run
```bash
pip install .
```
    5. open the notebook in a jupyter server (e.g. by running `jupyter notebook`), change kernel to the desired kernel.
    6. configure the constants in the top cell to match your preference, most likely being:
```python
SAVE_FIGs = False
REWRITE_DF = False
LOAD_DF = True
LOAD_ROIS = True
REWRITE_ROIS = False
```
    with this configuration, the existing aggregated data files are loaded and used, and no new files are saved.
    6. you may run the whole notebook, cell by cell or simply survey the code. 
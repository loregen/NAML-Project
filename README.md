# NAML-Project
Project for 2023-2024 Numerical Analysis for Machine Learning Course - Urban Sounds Classification 

## Setting up the environment
The project was developed on a MacOS M1 machine, for which a conda environment yaml file is provided (environment_AppleSilicon.yaml). If you are using a different machine, install the packages found in the yaml manually. 

## Downloading the datasets
The datasets are necessary to run the code.
The ESC-50 dataset can be downloaded and extracted by running the following commands in the root of the project:
```bash
wget https://github.com/karoldvl/ESC-50/archive/master.zip
unzip master.zip
rm master.zip
```
The UrbanSound8K dataset can be downloaded from the following link: https://urbansounddataset.weebly.com/urbansound8k.html. You will need to fill in a form to get access to the dataset. After downloading the dataset, extract it and move the UrbanSound8K folder to the root of the project.

## Structure of the project
There are 4 notebooks you can run:
- `ESC-10_analysis.ipynb` and `UrbanSound8K_analysis.ipynb` analyze some properties of the datasets, such as the distribution of sampling rates and the time durations of the audio files.
- `training.ipynb` is the main notebook where the models are trained. Here you can choose a wide range of parameters, such as the dataset, the preprocessing steps, the model architecture and so on.
- `inference.ipynb` is the notebook where you can load a trained model and analyze its performance on the test set. Confusion matrices and intermediate activations can be visualized.




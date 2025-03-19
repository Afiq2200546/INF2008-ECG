# INF2008 ECG Classification Project

## Overview
This repository contains scripts and Jupyter notebooks for processing and analyzing ECG data from the PTB Diagnostic ECG Database. The project involves extracting labels, processing ECG files, and training machine learning models to classify ECG signals.

## Prerequisites
Ensure you have the following dependencies installed before running the scripts:
- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- TensorFlow/Keras (for neural network models)

## Dataset
Download the PTB Diagnostic ECG Database from PhysioNet:
[PTB Diagnostic ECG Database](https://www.physionet.org/static/published-projects/ptbdb/ptb-diagnostic-ecg-database-1.0.0.zip)

## Installation and Setup

### Step 1: Download the ECG Dataset
Download the dataset from PhysioNet and extract it to a folder named `ptb-diagnostic-ecg-database-1.0.0`.

### Step 2: Clone the Repository
Clone this GitHub repository to your local machine:
```sh
git clone https://github.com/Afiq2200546/INF2008-ECG.git
```

### Step 3: Move the Dataset
Navigate to the cloned repository and extract the dataset into it:
```sh
cd INF2008-ECG
mv /path/to/ptb-diagnostic-ecg-database-1.0.0 .
```

### Step 4: Run in Jupyter Notebook or VS Code
You have the option to run the scripts in either Jupyter Notebook or VS Code.

#### Option 1: Run in Jupyter Notebook
Ensure Jupyter Notebook is installed and execute the following:
```sh
jupyter notebook
```
Then open and run the following notebooks in sequence:
1. `label_extractor.ipynb` (outputs `new_labels.csv`)
2. `File_processing.ipynb` (outputs `new_labels_processed.csv` + `ptb_processed_new`)
3. Machine learning models:
   - `Machine_Learning_RF_NN.ipynb`
   - `Machine_Learning_SVM_NN.ipynb`

#### Option 2: Run in VS Code
1. Install the Jupyter extension in VS Code.
2. Open the repository folder in VS Code:
   ```sh
   code INF2008-ECG
   ```
3. Open the notebooks and run them using the built-in Jupyter notebook feature.

## Results and Analysis
The output of the machine learning models includes trained models, performance metrics, and visualizations of ECG classification.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Special thanks to PhysioNet for providing the PTB Diagnostic ECG Database.

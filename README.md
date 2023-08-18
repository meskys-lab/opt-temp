# Model to predict optimal temperature

"Algoritmas, skirto fermento optimalios veikimo temperatūros nustatymui pagal duotą fermento seką, kūrimas"

## Description

This repository contains code for a model that predict optimal enzyme temperature. This algorithm is designed to be 
easily trained on data obtain from high throughput testing. On testing data (random split), the performance of the model
can be seen in the table below:


| MAE | Spearman R | Accuracy (meso vs thermo) |
|-----|------------|---------------------------|
| 6.7 | 0.617      | 97.4%                     |



## Installation

All dependencies and libraries are in environment file. To run and train GPU is required. This algorithm should work 
with any modern Nvidia GPU (tested on RTX 2080). To create conda environment run: 

```
conda env create -f environment.yml
```

## Train model

In order to train model on your data create csv files which contains two columns `sequence` and `temperature`.

Then run the command as shown below

```
python -m otm.train --train_csv train_split.csv --val_csv val_split.csv
```

## Make predictions

In order to make prediction you need to provide a fasta file and then run command by specifying pretrained model
weights as well as path to fasta file.

Note: predictions are run on GPU.

```
python -m otm.predict --model models/EsmOptTempRep_2023_08_18_10_32.pth --fasta example/to_predict.fasta

```

## Interpretation of the results
The default output of the program is a csv file which contains sequence id and sequence itself from fasta file.
The last column is the prediction of the model.
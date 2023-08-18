# Model to predict optimal temperature

"Algoritmas, skirto fermento optimalios veikimo temperatūros nustatymui pagal duotą fermento seką, kūrimas"

## Installation

First, clone environment and the create conda environment. To create conda environment run: 

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

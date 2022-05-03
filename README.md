# Reproducing DeepDTA: Deep Drug–Target Binding Affinity Prediction
This repository contains the codes and data for the final project of Deep Learning for Healthcare course at UIUC in Spring 2022. This project attempts to reproduce the major results of the DeepDTA paper, and includes some additional experiments beyond the paper.

## Contents
- [DeepDTA](#deepdta)
- [Dependencies](#dependencies)
- [Data](#data)
- [Codes](#codes)
  * [Preprocessing](#preprocessing)
  * [Training](#training)
  * [Evaluation](#evaluation)
- [Pretrained Models](#pretrained-models)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## DeepDTA
DeepDTA is a deep learning-based model that predicts the level of interaction, or binding affinity, between a drug and a target chemical. DeepDTA uses convolutional neural networks (CNNs) to learn representations from raw sequences of proteins and ligands.

<p align="center">
  <img src="https://github.com/hkmztrk/DeepDTA/blob/master/docs/figures/deepdta.PNG" alt="drawing" width="750"/>
</p>

**Citation to the paper**: Hakime Öztürk, Arzucan Özgür, Elif Ozkirimli, DeepDTA: deep drug–target binding affinity prediction, Bioinformatics, Volume 34, Issue 17, 01 September 2018, Pages i821–i829, [https://doi.org/10.1093/bioinformatics/bty593](https://doi.org/10.1093/bioinformatics/bty593)

**Code repository of the paper**: [https://github.com/hkmztrk/DeepDTA](https://github.com/hkmztrk/DeepDTA)

**Code repository of the baseline SimBoost**: [https://github.com/hetong007/SimBoost](https://github.com/hetong007/SimBoost)

**Code repository of the baseline KronRLS**: [https://github.com/aatapa/RLScore](https://github.com/aatapa/RLScore)

## Dependencies
**DeepDTA** is written in Python 3 and has the following dependencies. 
- [Python 3.4+](https://www.python.org)
- [Keras 2.x](https://pypi.org/project/keras/)
- [Tensorflow 1.x](https://www.tensorflow.org/install/)
- [Numpy](https://numpy.org)
- [Matplotlib](https://matplotlib.org)

It is important to note that most Tensorflow and Keras commands used in DeepDTA code are deprecated in the new version of Tensorflow, and therefore Tensorflow 1.x should necessarily be used to run the code. Google Colab loads Tensorflow 2.x by default, and the 1.x version could be loaded with the following command:
```%tensorflow_version 1.x```

**SimBoost** is written in R and has the following dependencies.
- [xgboost 1.6.0](https://cran.r-project.org/web/packages/xgboost/index.html)
- [igraph 1.3.0](https://igraph.org/r/)
- [recosystem 0.11.0](https://cran.r-project.org/web/packages/recosystem/index.html)
- [ROCR 1.0](https://cran.r-project.org/web/packages/ROCR/index.html)

**KronRLS** is written in Python 2 and has the following dependencies.
- [Python 2.7+](https://www.python.org)
- [Numpy](https://numpy.org)
- [C-compiler](https://gcc.gnu.org)

To compute the Area Under Precision Recall curve (AUPR) as performance measure, all methods use the `auc.jar` package available for download at [http://mark.goadrich.com/programs/AUC/](http://mark.goadrich.com/programs/AUC/) and located at [auc.jar](DeepDTA/source/auc.jar) in the repository.

## Data
The paper uses the Davis Kinase binding affinity dataset [(Davis et al., 2011)](https://www.nature.com/articles/nbt.1990), containing 442 proteins and 68 compounds with overall 30,056 interactions, and the KIBA large-scale kinase inhibitors bioactivity dataset [(Tang et al., 2014)](https://pubs.acs.org/doi/10.1021/ci400709d), containing 229 proteins and 2111 compounds with overall 118,254 interactions. 

**Raw Data Download**: The raw datasets are available for download from the following links:
- Davis: [http://staff.cs.utu.fi/~aatapa/data/DrugTarget/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt](http://staff.cs.utu.fi/~aatapa/data/DrugTarget/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt)
- KIBA: [http://pubs.acs.org/doi/suppl/10.1021/ci400709d](http://pubs.acs.org/doi/suppl/10.1021/ci400709d)

**Preprocessed Data**: The preprocessed datasets are located under [DeepDTA/data](DeepDTA/data) directory as `kiba` and `davis`. Each dataset directory contains several files named as follows:
- `proteins.txt`: This file contains raw amino-acid sequences of proteins.
- `ligands_can.txt`: This file continas the raw SMILES sequences of ligands (compounds) in canonical form.
- `Y`: This file contains binding affinity values between proteins and ligands.
- `target-target_similarities_WS.txt`: This file contains the Smith-Waterman (SW) matrices of similarity between target pairs.
- `drug-drug_similarities_2D.txt`: This file contains the Pubchem Sim matrices of similarity between drug pairs.

**Data Statistics**: The [data_statistics.ipynb](./data_statistics.ipynb) file demonstrates some statistics of the datasets, including distribution of the binding affinity scores and distribution of the protein and SMILES sequence lengths for both Davis and KIBA datasets.

## Pretrained Models

*Note*: Since the Keras and Tensorflow versions used in DeepDTA are old and now deprecated, the recent `h5py` packages can not be used to load the pretrained models. You will need to reinstall the package using the following command:
```
pip install 'h5py==2.10.0' --force-reinstall
```
The following code can then be used to load a pretrained model:
```python
from keras.models import load_model
model = load_model('combined_davis.h5', custom_objects={"cindex_score": cindex_score})
model.summary()
```
Based on where you run the code, you may also need to have the `cindex_score` function, which is available at [DeepDTA/source/run_experiments.py](DeepDTA/source/run_experiments.py).

## Results
For each experiment on the Davis dataset, the total number of training samples was 20036 and the total number of test samples was 5010.
### Davis
| Method      | Proteins    |Compounds    | CI-Index | MSE Loss |Avg Training Time (per epoch) |
| ----------- | ----------- | ----------- | ----------- | ----------- |----------- |
| KronRLS     |  S–W       | Pubchem Sim |  |  |  |
| SimBoost    |  S–W       | Pubchem Sim |  |  |  |
| DeepDTA     |  S–W       | Pubchem Sim |  |  |  |
| DeepDTA     |   CNN      | Pubchem Sim |  |  |  |
| DeepDTA     |  S–W       | CNN         |  |  |  |
| DeepDTA     |    CNN     | CNN         |  |  |  |

### KIBA
For each experiment on the KIBA dataset, the total number of training samples was 78836 and the total number of test samples was 19709.
| Method      | Proteins    |Compounds    | CI-Index | MSE Loss |Avg Training Time (per epoch) |
| ----------- | ----------- | ----------- | ----------- | ----------- |----------- |
| KronRLS     |  S–W       | Pubchem Sim |  |  |  |
| SimBoost    |  S–W       | Pubchem Sim |  |  |  |
| DeepDTA     |  S–W       | Pubchem Sim |  |  |  |
| DeepDTA     |   CNN      | Pubchem Sim |  |  |  |
| DeepDTA     |  S–W       | CNN         |  |  |  |
| DeepDTA     |    CNN     | CNN         |  |  |  |
## Acknowledgements
Please cite the original paper if you are using this code in your work.
```
@article{ozturk2018deepdta,
  title={DeepDTA: deep drug--target binding affinity prediction},
  author={{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzucan and Ozkirimli, Elif},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i821--i829},
  year={2018},
  publisher={Oxford University Press}
}
```


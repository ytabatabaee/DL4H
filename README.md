# Reproducing DeepDTA: Deep Drug–Target Binding Affinity Prediction
This repository contains the codes and data for the final project of Deep Learning for Healthcare course at UIUC in Spring 2022. This project attempts to reproduce the major results of the [DeepDTA paper (Ozturk et al.,2018)](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245), and includes some additional experiments beyond the paper.

## Contents
- [DeepDTA](#deepdta)
- [Dependencies](#dependencies)
- [Data](#data)
- [Codes](#codes)
  * [Preprocessing](#preprocessing)
  * [Training](#training-and-evaluation)
  * [Evaluation](#training-and-evaluation)
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

**Code repository of the baseline SimBoost (official - in R)**: [https://github.com/hetong007/SimBoost](https://github.com/hetong007/SimBoost)

**Code repository of the baseline SimBoost (unofficial - in Python)**: [https://github.com/mahtaz/Simboost-ML_project-](https://github.com/mahtaz/Simboost-ML_project-)

**Code repository of the baseline KronRLS**: [https://github.com/aatapa/RLScore](https://github.com/aatapa/RLScore)

## Dependencies
**DeepDTA** is written in Python 3 and has the following dependencies.
- [Python 3.4+](https://www.python.org)
- [Keras 2.x](https://pypi.org/project/keras/)
- [Tensorflow 1.x](https://www.tensorflow.org/install/)
- [Numpy](https://numpy.org)
- [Matplotlib](https://matplotlib.org)

It is important to note that most Tensorflow and Keras commands used in DeepDTA code are deprecated in the new version of Tensorflow, and therefore Tensorflow 1.x should necessarily be used to run the code. Google Colab loads Tensorflow 2.x by default, and the 1.x version can be loaded with the following command:
```%tensorflow_version 1.x```

**SimBoost** (official code) is written in R and has the following dependencies.
- [xgboost 1.6.0](https://cran.r-project.org/web/packages/xgboost/index.html)
- [igraph 1.3.0](https://igraph.org/r/)
- [recosystem 0.11.0](https://cran.r-project.org/web/packages/recosystem/index.html)
- [ROCR 1.0](https://cran.r-project.org/web/packages/ROCR/index.html)

**SimBoost** (unofficial code) is written in Python 3 and has the following dependencies.
- [Python 3.4+](https://www.python.org)
- [scikit-learn](https://scikit-learn.org/stable/)
- [python-igraph](https://igraph.org/python/)
- [Tensorflow 1.x](https://www.tensorflow.org/install/)
- [Numpy](https://numpy.org)
- [Matplotlib](https://matplotlib.org)

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

## Codes
### Preprocessing
The preprocessed data is already available at [DeepDTA/data](DeepDTA/data). *All the experiments were done on this preprocessed data, and no extra preprocessing is required*. The [data_statistics.ipynb](./data_statistics.ipynb) jupyter notebook shows how the data can be loaded and used in any code, and demonstrates a reproduction of Figure 1 in the paper. 

#### Additional Explanation of the Data 
This section is partially taken from the original README of the paper's repository.

**Similarity files**

For each dataset, there are two similarity files, drug-drug and target-target similarities.
*  Drug-drug similarities obtained via Pubchem structure clustering.
*  Target-target similarities are obtained via S-W similarity.

These files were used to re-produce the results of two other methods [(Pahikkala et al., 2017)](https://academic.oup.com/bib/article/16/2/325/246479) and [(He et al., 2017)](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z), and also for some experiments in DeepDTA model, please refer to [paper](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245). 

**Binding affinity files**

*  For davis dataset, standard value is Kd in nM. In the article, the following transformation was used:

<a href="https://www.codecogs.com/eqnedit.php?latex=pK_{d}=-log_{10}\frac{K_d}{1e9}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?pK_{d}=-log_{10}\frac{K_d}{1e9}" title="pK_{d}=-log_{10}\frac{K_d}{1e9}" /></a>

* For KIBA dataset, standard value is KIBA score. Two versions of the binding affinity value txt files correspond the original values and transformed values ([more information here](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z)). In the article the tranformed form was used. 

* nan values indicate there is no experimental value for that drug-target pair.

**Train and test folds**
There are two files for each dataset: train fold and test fold. Both of these files keep the position information for the binding affinity value given in binding affinity matrices in the text files. 
*  Since the authors performed 5-fold cv, each fold file contains five different set of positions.
*  Test set is same for all five training sets.

**For using the folds**
*   Load affinity matrix Y 

```python
import pickle
import numpy as np

Y = pickle.load(open("Y", "rb"))  # Y = pickle.load(open("Y", "rb"), encoding='latin1')
# log transformation for davis 
if log_space:
        Y = -(np.log10(Y/(math.pow(10,9))))
label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
```

*  label_row_inds: drug indices for the corresponding affinity matrix positions (flattened)  
    e.g. 36275th point in the KIBA Y matrix indicates the 364th drug (same order in the SMILES file) 
    ```python
    label_row_inds[36275]
    ```

*  label_col_inds: protein indices for the corresponding affinity matrix positions (flattened)

    e.g.  36275th point in the KIBA Y matrix indicates the 120th protein (same order in the protein sequence file) 
    ```python
    label_col_inds[36275]
    ```
    
*   You can then load the fold files as follows:
    ```python
    import json
    test_fold = json.load(open(yourdir + "folds/test_fold_setting1.txt"))
    train_folds = json.load(open(yourdir + "folds/train_fold_setting1.txt"))
    
    test_drug_indices = label_row_inds[test_fold]
    test_protein_indices = label_col_inds[test_fold]
    
    ```
    
    Remember that, ```train_folds``` contain an array of 5 lists, each of which correspond to a training set.

#### Preprocessing for SimBoost
A code for preprocessing the raw data and generating the similarity matrices for SimBoost is available in the [simboost_R/preprocessing/](simboost_R/preprocessing/) directory and can be run using the commands below, which will generate `.Rda` files in the [simboost_R/data/](simboost_R/data/) directory.
```r
Rscript preprocess_metz_davis.R
Rscript preprocess_Kiba.R
```
Note that the kiba dataset should be manually downloaded, but davis will be downloaded automatically in the code. 

### Training and Evaluation
The training and evaluation can be done with a single command in all methods, but the evaluation could be done separately by loading the pretrained models as well. We will bring the commands used for running the experiments for each method below:
#### DeepDTA
DeepDTA can be run with the following command.
```bash
$ cd DeepDTA/source/
$ python3 run_experiments.py --num_windows 32 \
                          --seq_window_lengths 8 \
                          --smi_window_lengths 6 \
                          --batch_size 256 \
                          --num_epoch 100 \
                          --max_seq_len 1000 \
                          --max_smi_len 100 \
                          --dataset_path 'data/kiba/' \
                          --problem_type 1 \
                          --log_dir 'logs/'
```
We explain some of the non-trivial parameters below:
* `--num_windows`: The number of filters for the first convlutional layer.
* `--seq_window_lengths`, `--smi_window_lengths`: fixed length of windows for protein (seq) and compound (smiles) sequences. Could be provided as a range, such as `4 8 12`.
* `--max_seq_len`, `--max_smi_len`: fixed lengths of protein (seq) and compound (smi) sequences. These were set as 1000 and 100 respectively for kiba and 1200 and 85 for davis in the paper.
* `--problem_type`: 1 for kiba and 0 for davis, indicates whether a log transformation is needed 

To use a different performance measure or run one of the DeepDTA baselines, you can change the following two lines at the end of `run_experiments.py`:
```python
perfmeasure = get_cindex # specify performance measure, e.g. mse loss
deepmethod = build_combined_categorical # specify model type, e.g. baseline, combined, etc
```
#### SimBoost
We only ran the python version of SimBoost in this project. The python codes for training and evaluation of SimBoost are available in the jupyter notebooks [simboost_python/SimBoost_kiba.ipynb](simboost_python/SimBoost_kiba.ipynb) and [simboost_python/SimBoost_davis.ipynb](simboost_python/SimBoost_davis.ipynb).

* Command for running the R version:
```bash
$ cd simboost_R/xgboost/
$ Rscript Sequential.feature.*.R
$ Rscript Sequential.cv.xgb.quantile.exec.R
$ Rscript Sequential.cv.xgb.exec.R
```
where `*` is the name of the dataset (kiba or davis).
#### KronRLS
KronRLS can be run with the following commands. Note that `setup.py` should only be run to install for the first time.
```bash
$ cd KronRLS/
$ python2 setup.py # use only in the first run
$ python2 kronecker_experiments.py
```
The dataset and problem type (regression or classification) could be specified in the main function in the `kronecker_experiments.py`, you can just uncomment the function you want to run, such as `kiba_regression()` or `davis_regression()`.
## Pretrained Models
Most of the pretrained models are provided in the [pretrained_models](./pretrained_models) directory in Github, but the ones that were larger than 100MB are provided in Google Drive, with links available below:
- KronRLS davis model: [pretrained_models/davis_kronrls.pkl](./pretrained_models/davis_kronrls.pkl)
- KronRLS KIBA model: [kiba_kronrls.pkl - Google Drive link](https://drive.google.com/drive/folders/1W9iw1pddJd3y52l56Ac1eIDWplLAtJWP?usp=sharing)
- SimBoost davis model: [pretrained_models/davis_simboost.pkl](./pretrained_models/davis_simboost.pkl)
- SimBoost KIBA model: [pretrained_models/kiba_simboost.pkl](./pretrained_models/kiba_simboost.pkl)
- DeepDTA (CNN, CNN) davis model: [pretrained_models/combined_davis.h5](./pretrained_models/combined_davis.h5)
- DeepDTA (CNN, CNN) KIBA model: [pretrained_models/combined_kiba.h5](./pretrained_models/combined_kiba.h5)
- DeepDTA (SW, CNN) davis model: [pretrained_models/single_drug_davis.h5](./pretrained_models/single_drug_davis.h5)
- DeepDTA (SW, CNN) kiba model: [pretrained_models/single_drug_kiba.h5](./pretrained_models/single_drug_kiba.h5)
- DeepDTA (CNN, Pubchem Sim) davis model: [pretrained_models/single_prot_davis.h5](./pretrained_models/single_prot_davis.h5)
- DeepDTA (CNN, Pubchem Sim) kiba model: [pretrained_models/single_prot_kiba.h5](./pretrained_models/single_prot_kiba.h5)
- DeepDTA (SW, Pubchem Sim) davis model: [pretrained_models/baseline_davis.h5](./pretrained_models/baseline_davis.h5)
- DeepDTA (SW, Pubchem Sim) kiba model: [pretrained_models/baseline_kiba.h5](./pretrained_models/baseline_kiba.h5)

#### Loading DeepDTA pretrained models
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

#### Evaluating the pretrained models
The evaluation and training codes are not separate for any of the methods, however, evaluation can be easily done using the pretrained models as well. A complete example is available at the end of [simboost_python/SimBoost_davis.ipynb](simboost_python/SimBoost_davis.ipynb). All the `.pkl` models can be loaded with pickle as below.
```python
import pickle
loaded_model = pickle.load(open('davis_simboost.pkl', 'rb'))
Y_pred = loaded_model.predict(X_test)

print("Davis Test CI-Index: %.3f" % cindex_score(Y_test, Y_pred))
print("Davis Test MSE: %.3f" % mean_squared_error(Y_test, Y_pred))
```

## Results
For each experiment on the Davis dataset, the total number of training samples was 20036 and the total number of test samples was 5010.
### Davis
| Method      | Proteins    |Compounds    | CI-Index | MSE Loss |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| KronRLS     |  S–W       | Pubchem Sim | 0.867 | 0.376 |
| SimBoost    |  S–W       | Pubchem Sim | 0.862 | 0.298 |
| DeepDTA     |  S–W       | Pubchem Sim | 0.771 | 0.685 |
| DeepDTA     |   CNN      | Pubchem Sim | 0.810 | 0.490 |
| DeepDTA     |  S–W       | CNN         | 0.823 | 0.462 |
| DeepDTA     |    CNN     | CNN         | 0.876 | 0.255 |

### KIBA
For each experiment on the KIBA dataset, the total number of training samples was 78836 and the total number of test samples was 19709.
| Method      | Proteins    |Compounds    | CI-Index | MSE Loss |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| KronRLS     |  S–W       | Pubchem Sim | 0.794 | 0.373 |
| SimBoost    |  S–W       | Pubchem Sim | 0.824 | 0.279 |
| DeepDTA     |  S–W       | Pubchem Sim | 0.704 | 1.59 |
| DeepDTA     |   CNN      | Pubchem Sim | 0.702 | 0.541 |
| DeepDTA     |  S–W       | CNN         | 0.759 | 0.355 |
| DeepDTA     |    CNN     | CNN         | 0.857 | 0.211 |
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

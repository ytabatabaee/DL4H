# Reproducing DeepDTA: Deep Drug–Target Binding Affinity Prediction
This repository contains the codes and data for the final project of Deep Learning for Healthcare course at UIUC in Spring 2022. This project attempts to reproduce the major results of the DeepDTA paper, and includes some additional experiments beyond the paper.

## Contents
- [Original Paper](#original-paper)
- [Dependencies](#dependencies)
- [Data](#data)
- [Codes](#codes)
  * [Preprocessing](#preprocessing)
  * [Training](#training)
  * [Evaluation](#evaluation)
- [Pretrained Models](#pretrained-models)
- [Results](#results)

## Original Paper
**Citation to the paper**: Hakime Öztürk, Arzucan Özgür, Elif Ozkirimli, DeepDTA: deep drug–target binding affinity prediction, Bioinformatics, Volume 34, Issue 17, 01 September 2018, Pages i821–i829, [https://doi.org/10.1093/bioinformatics/bty593](https://doi.org/10.1093/bioinformatics/bty593)

**Code repository of the paper**: [https://github.com/hkmztrk/DeepDTA](https://github.com/hkmztrk/DeepDTA)

**Code repository of the baseline SimbBoost**: [https://github.com/hetong007/SimBoost](https://github.com/hetong007/SimBoost)

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

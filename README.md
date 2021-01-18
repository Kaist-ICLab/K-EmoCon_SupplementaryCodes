# K-EmoCon Supplementary Codes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3931963.svg)](https://doi.org/10.5281/zenodo.3931963)
This repository contains supplementary codes for the **K-EmoCon Dataset**.

For the detailed description of the dataset, please refer to:
> Park, C.Y., Cha, N., Kang, S. *et al.* K-EmoCon, a multimodal sensor dataset for continuous emotion recognition in naturalistic conversations. *Sci Data* **7**, 293 (2020). https://doi.org/10.1038/s41597-020-00630-y
---

## Usage
### Installation
```console
$ git clone https://github.com/cheulyop/K-EmoCon_SupplementaryCodes.git
$ cd K-EmoCon_SupplementaryCodes
```

### Preprocessing
```console
$ python preprocess.py --root '/path/to/kemocon_root'
```
Running [preprocess.py](https://github.com/cheulyop/K-EmoCon_SupplementaryCodes/blob/master/preprocess.py) will create 5-second segments containing 4 types of biosignals `['bvp', 'eda', 'temp', 'ecg']` acquired during debates as JSON files to `'/path/to/kemocon_root/segments/'`, under subdirectories corresponding to each participant.

JSON files for biosignal segments will have names with the following pattern: for example, `p01-017-243333.json` indicates that the file is a 17th 5-second biosignal segment for participant 1.

The last 6 digits are multiperspective emotion annotations associated with the segment, in the order of 1) self-arousal, 2) self-valence, 3) partner-arousal, 4) partner-valence, 5) external-arousal, and 6) external-valence.

### Baseline Classification
[baseline.py](https://github.com/cheulyop/K-EmoCon_SupplementaryCodes/blob/master/baseline.py) implements a stratified k-fold and leave-one-subject-out (LOSO) cross-validation for the binary classification of low and high classes (with low < 3 and high >= 3), with three simple voters (random, majority, and class ratio), [Gaussian NB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn-naive-bayes-gaussiannb), and [XGBoost](https://github.com/dmlc/xgboost).

\* `baseline.py` requires [PyTEAP](https://pypi.org/project/PyTEAP/) for feature extraction. You can install it by running: `pip install PyTEAP` in your terminal.

```console
$ python baseline.py --help
usage: baseline.py [-h] -r ROOT [-tz TIMEZONE] [-s SEED] [-t TARGET] [-l LENGTH] [-y LABEL] [--majority] [--rolling] [--cv CV] [--splits SPLITS] [--shuffle]

Preprocess K-EmoCon dataset and get baseline classification results.

optional arguments:
  -h, --help            show this help message and exit
  -r ROOT, --root ROOT  path to the dataset directory
  -tz TIMEZONE, --timezone TIMEZONE
                        a pytz timezone string for logger, default is UTC
  -s SEED, --seed SEED  seed for random number generation, default is 0
  -t TARGET, --target TARGET
                        target label for classification, must be either "valence" or "arousal"
  -l LENGTH, --length LENGTH
                        number of consecutive 5s-signals in one segment, default is 5
  -y LABEL, --label LABEL
                        type of label to use for classification, must be either "s"=self, "p"=partner, "e"=external, or "sp"=self+partner
  --majority            set majority label for segments, default is last
  --rolling             get segments with rolling: e.g., s1=[0:n], s2=[1:n+1], ..., default is no rolling: e.g., s1=[0:n], s2=[n:2n], ...
  --cv CV               type of cross-validation to perform, must be either "kfold" or "loso"
  --splits SPLITS       number of folds for k-fold stratified classification, default is 5
  --shuffle             shuffle data before splitting to folds, default is no shuffle
```
---

## Results

Below are sample commands and corresponding classification results for arousal and valence, with seed = 1, segment length = 25s (5*5s, using annotation of the last segment as the segment label), label = self, rolling, 5-fold CV, and shuffle.

### Arousal (5-fold CV)

```console
$ python baseline.py -r /path/to/kemocon_root/segments/ -s 1 -t arousal -l 5 -y s --rolling --cv kfold --splits 5 --shuffle --gpu
```
| Metric   |   Random |   Majority |   Class ratio |   Gaussian NB |   XGBoost |
|:---------|---------:|-----------:|--------------:|--------------:|----------:|
| acc.     | 0.503671 |   0.684578 |      0.560299 |       0.61854 |  0.80108  |
| auroc    | 0.5      |   0.5      |      0.491283 |       0.54747 |  0.851438 |
| bacc.    | 0.500061 |   0.5      |      0.491283 |       0.51454 |  0.729572 |
| f1       | 0.584102 |   0.812759 |      0.67853  |       0.74067 |  0.864128 |

### Valence (5-fold CV)

```console
$ python baseline.py -r /path/to/kemocon_root/segments/ -s 1 -t valence -l 5 -y s --rolling --cv kfold --splits 5 --shuffle --gpu
```
| Metric   |   Random |   Majority |   Class ratio |   Gaussian NB |   XGBoost |
|:---------|---------:|-----------:|--------------:|--------------:|----------:|
| acc.     | 0.520099 |   0.804348 |      0.667756 |      0.748167 |  0.851923 |
| auroc    | 0.5      |   0.5      |      0.474616 |      0.568466 |  0.853776 |
| bacc.    | 0.522477 |   0.5      |      0.474616 |      0.527758 |  0.646984 |
| f1       | 0.634711 |   0.891566 |      0.793022 |      0.850216 |  0.914462 |

\* For the results of arousal and valence **LOSO CV**, see csv files in the `results` folder. AUROC is not calculated for participants who only have one class in y_true, as the [ROC AUC score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score) is not defined in that case.

* [`arousal-loso.csv`](https://github.com/cheulyop/K-EmoCon_SupplementaryCodes/blob/master/results/arousal-loso.csv)
* [`valence-loso.csv`](https://github.com/cheulyop/K-EmoCon_SupplementaryCodes/blob/master/results/valence-loso.csv)
---

## Supplementary Codes for the K-EmoCon Dataset Descriptor
* [chauvenet.py](https://github.com/Kaist-ICLab/K-EmoCon_SupplementaryCodes/blob/master/chauvenet.py) - an implementation of [Chauvenet's criterion](https://en.wikipedia.org/wiki/Chauvenet%27s_criterion) for detecting outliers.
* [vote_majority.py](https://github.com/Kaist-ICLab/K-EmoCon_SupplementaryCodes/blob/master/vote_majority.py) - implements a majority voting to get a consensus between external annotations.
* [plotting.py](https://github.com/cheulyop/K-EmoCon_SupplementaryCodes/blob/master/utils/plotting.py) - includes functions to produce the IRR heatmap (Fig. 4) in the K-EmoCon dataset descriptor.
  * `get_annotations` - loads annotations saved as CSV files.
  * `subtract_mode_from_values` - implements mode subtraction.
  * `compute_krippendorff_alpha` - computes Krippendorff's alpha (IRR).
  * `plot_heatmaps` - plots the IRR heatmap.

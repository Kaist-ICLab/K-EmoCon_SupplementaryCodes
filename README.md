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
[baseline.py](https://github.com/cheulyop/K-EmoCon_SupplementaryCodes/blob/master/baseline.py) implements a stratified k-fold baseline classification with four simple classifiers, which are [Gaussian NB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn-naive-bayes-gaussiannb), random voter, majority voter, and class ratio voter.

```console
$ python baseline.py --help
usage: baseline.py [-h] --root ROOT [--seed SEED] [--target TARGET] [--length LENGTH] [--which WHICH] [--rolling] [--no_rolling] [--splits SPLITS] [--shuffle] [--no_shuffle]

Preprocess K-EmoCon dataset and get baseline classification results.

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT, -r ROOT  path to the dataset directory
  --seed SEED, -s SEED  seed for random number generation
  --target TARGET, -t TARGET
                        target label for classification, must be either "valence" or "arousal"
  --length LENGTH, -l LENGTH
                        number of consecutive 5s-signals in one segment
  --which WHICH, -w WHICH
                        which label to set for segments, must be either "last" or "majority"
  --rolling             get segments with rolling: e.g., s1=[0:n], s2=[1:n+1], ...
  --no_rolling          get segments without rolling: e.g., s1=[0:n], s2=[n:2n], ...
  --splits SPLITS, -k SPLITS
                        number of fold in k-fold stratified classification
  --shuffle             shuffle data before splitting to folds
  --no_shuffle          don't shuffle data before splitting to folds
```

Below is a sample command to perform a *5-fold* stratified classification with *valence* as a target, for *25s rolling-based* segments, with *last labels* as segment labels, with *shuffle*, and *seed=1*:

```console
$ python baseline.py --root '/path/to/kemocon_root/segments/' -seed 1 --target 'valence' --length 5 --rolling --which 'last' --shuffle
```

Here are sample commands and corresponding classification results:

#### Arousal
```console
$ python baseline.py -r '/path/to/kemocon/segments/' --target "arousal" --which "last" --rolling --splits 5 --shuffle --seed 1
```
| Metric   |   Gaussian NB |   Random |   Majority |   Class ratio |
|:---------|--------------:|---------:|-----------:|--------------:|
| acc.     |      0.61691  | 0.512718 |   0.662429 |      0.544286 |
| bacc.    |      0.531167 | 0.517761 |   0.5      |      0.487093 |
| f1       |      0.319523 | 0.424699 |   0        |      0.315257 |

#### Valence
```console
$ python baseline.py -r '/path/to/kemocon/segments/' --target "valence" --which "last" --rolling --splits 5 --shuffle --seed 1
```
| Metric   |   Gaussian NB |   Random |   Majority |   Class ratio |
|:---------|--------------:|---------:|-----------:|--------------:|
| acc.     |      0.729704 | 0.490557 |   0.760049 |      0.634928 |
| bacc.    |      0.523319 | 0.494089 |   0.5      |      0.494309 |
| f1       |      0.183313 | 0.32008  |   0        |      0.227544 |

## Supplementary Codes for the K-EmoCon Dataset Descriptor
* [chauvenet.py](https://github.com/Kaist-ICLab/K-EmoCon_SupplementaryCodes/blob/master/chauvenet.py) - an implementation of [Chauvenet's criterion](https://en.wikipedia.org/wiki/Chauvenet%27s_criterion) for detecting outliers.
* [vote_majority.py](https://github.com/Kaist-ICLab/K-EmoCon_SupplementaryCodes/blob/master/vote_majority.py) - implements a majority voting to get a consensus between external annotations.
* [plotting.py](https://github.com/cheulyop/K-EmoCon_SupplementaryCodes/blob/master/utils/plotting.py) - includes functions to produce the IRR heatmap (Fig. 4) in the K-EmoCon dataset descriptor.
  * `get_annotations` - loads annotations saved as CSV files.
  * `subtract_mode_from_values` - implements mode subtraction.
  * `compute_krippendorff_alpha` - computes Krippendorff's alpha (IRR).
  * `plot_heatmaps` - plots the IRR heatmap.
---

## Changelog

@Sep 23, 2020: added `baseline.py` for baseline classification.

@Sep 16, 2020: added `preprocess.py` and `logging.py` for preprocessing the dataset.

@Jul 7, 2020: updated `vote_majority.py`
* Updated aggregate_by_majority_voting to support easier aggregation of external rater annotations.

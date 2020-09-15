# K-EmoCon Supplementary Codes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3931963.svg)](https://doi.org/10.5281/zenodo.3931963)
This repository contains supplementary codes for the K-EmoCon dataset.

For the detailed description of the dataset, please refer to: [K-EmoCon, a multimodal sensor dataset for continuous emotion recognition in naturalistic conversations](https://www.nature.com/articles/s41597-020-00630-y)


## Usage



### utils
* [Chauvenet's criterion](https://github.com/Kaist-ICLab/K-EmoCon_SupplementaryCodes/blob/master/chauvenet.py)
* [Majority voting](https://github.com/Kaist-ICLab/K-EmoCon_SupplementaryCodes/blob/master/vote_majority.py)
* [Plotting](https://github.com/Kaist-ICLab/K-EmoCon_SupplementaryCodes/blob/master/utils.py)
  * Loading annotations saved as CSV files
  * Computing Krippendorff's alpha (IRR)
  * Plotting heatmaps


## Changelog

@Jul 7, 2020: Updated vote_majority.py
* Updated aggregate_by_majority_voting to support easier aggregation of external rater annotations.

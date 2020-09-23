import os
import json
import logging
import argparse
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import OrderedDict
from numpy.random import default_rng
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from pyteap.signals.bvp import acquire_bvp, get_bvp_features
from pyteap.signals.gsr import acquire_gsr, get_gsr_features
from pyteap.signals.hst import acquire_hst, get_hst_features
from pyteap.signals.ecg import get_ecg_features
from pyteap.utils.logging import init_logger


def load_segments(segments_dir):
    segments = {}

    # for each participant
    for pid in os.listdir(segments_dir):
        segments.setdefault(int(pid), [])
        froot = os.path.join(segments_dir, pid)

        # for segments for a participant
        for fname in os.listdir(froot):
            # get labels, segment index, and path to json file
            labels = fname.split('-')[-1].split('.')[0]
            idx = int(fname.split('-')[1])
            fpath = os.path.join(froot, fname)

            # load json file and save to dict of pid: [segments = (idx, segment, labels)]
            with open(fpath) as f:
                seg = json.load(f)
                segments[int(pid)].append((idx, seg, labels))

    # return dict sorted by pid
    return OrderedDict(sorted(segments.items(), key=lambda x: x[0]))


def get_features(sig, sr, sigtype):
    if sigtype == 'bvp':
        features = get_bvp_features(acquire_bvp(sig, sr), sr)
    elif sigtype == 'eda':
        features = get_gsr_features(acquire_gsr(sig, sr, conversion=1e6), sr)
    elif sigtype == 'temp':
        features = get_hst_features(acquire_hst(sig, sr), sr)
    elif sigtype == 'ecg':
        features = get_ecg_features(sig)
    return features


def get_data_rolling(segments, n, which_label):
    X, y = {}, {}

    # for each participant
    for pid, segs in segments.items():
        segs = sorted(segs, key=lambda x: x[0])
        pbar = tqdm(range(len(segs) - n), desc=f'Subject {pid:02d}', ascii=True, dynamic_ncols=True)

        curr_X, curr_y = [], []
        for i in pbar:
            # get n consecutive segments from i-th segment
            curr_segs = segs[i:i + n]

            features = []
            # get features
            for sigtype, sr in [('bvp', 64), ('eda', 4), ('temp', 4), ('ecg', 1)]:
                sig = np.concatenate([sigs[sigtype] for _, sigs, _ in curr_segs])
                features.extend(get_features(sig, sr, sigtype))

            # skip if one or more feature is NaN
            if np.isnan(features).any():
                logging.getLogger('default').warning('One or more feature is NaN, skipped.')
                continue

            # take label of the last segment
            if which_label == 'last':
                labels = curr_segs[-1][-1]
                a_val, v_val = int(labels[0]), int(labels[1])
            # or take majority label
            elif which_label == 'majority':
                a_values, a_counts = np.unique([int(labels[0]) for _, _, labels in curr_segs], return_counts=True)
                v_values, v_counts = np.unique([int(labels[1]) for _, _, labels in curr_segs], return_counts=True)
                a_val = a_values[np.argmax(a_counts)]
                v_val = v_values[np.argmax(v_counts)]

            curr_X.append(features)
            curr_y.append([int(a_val > 3), int(v_val > 3)])

        # stack features for current participant and apply min-max scaling
        X[pid] = StandardScaler().fit_transform(np.stack(curr_X))
        y[pid] = np.stack(curr_y)

    return X, y


def get_data_discrete(segments, n, which_label):
    X, y = {}, {}

    # for each participant
    for pid, segs in segments.items():
        segs = sorted(segs, key=lambda x: x[0])
        pbar = tqdm(segs, desc=f'For subject {pid:02d}', ascii=True, dynamic_ncols=True)

        curr_X, curr_y, curr_segs = [], [], {}
        # for each segment
        for idx, signals, labels in pbar:
            # get labels and add to buffer
            s_a, s_v = int(labels[0]), int(labels[1])
            curr_segs.setdefault('a', []).append(s_a)
            curr_segs.setdefault('v', []).append(s_v)

            # get signals and add to buffer
            for sigtype, sr in [('bvp', 64), ('eda', 4), ('temp', 4), ('ecg', 1)]:
                curr_segs.setdefault(sigtype, []).append(signals[sigtype])

                # if n segments are in buffer
                if len(curr_segs[sigtype]) == n:
                    # concat signals and get features
                    sig = np.concatenate(curr_segs.pop(sigtype))
                    features = get_features(sig, sr, sigtype)
                    curr_segs.setdefault('features', []).append(features)

            # if features are in the buffer, pop features and labels
            if 'features' in curr_segs:
                features = np.concatenate(curr_segs.pop('features'))
                # skip if one or more feature is NaN
                if np.isnan(features).any():
                    logging.getLogger('default').warning('One or more feature is NaN, skipped.')
                    continue

                # take label of the last segment
                if which_label == 'last':
                    a_val = curr_segs.pop('a')[-1]
                    v_val = curr_segs.pop('v')[-1]
                # or take majority label
                elif which_label == 'majority':
                    a_values, a_counts = np.unique(curr_segs.pop('a'), return_counts=True)
                    v_values, v_counts = np.unique(curr_segs.pop('v'), return_counts=True)
                    a_val = a_values[np.argmax(a_counts)]
                    v_val = v_values[np.argmax(v_counts)]

                curr_X.append(features)
                curr_y.append([int(a_val > 3), int(v_val > 3)])
                pbar.set_postfix({'processed': idx // n})

        # stack features for current participant and apply min-max scaling
        X[pid] = StandardScaler().fit_transform(np.stack(curr_X))
        y[pid] = np.stack(curr_y)

    return X, y


def prepare_kemocon(segments_dir, n, which_label, rolling):
    # load segments
    pid_to_segments = load_segments(segments_dir)

    # extract features and labels
    if rolling:
        X, y = get_data_rolling(pid_to_segments, n, which_label)
    else:
        X, y = get_data_discrete(pid_to_segments, n, which_label)

    features = np.concatenate(list(X.values()))
    labels = np.concatenate(list(y.values()))
    return features, labels


def pred_majority(majority, y_test):
    preds = np.repeat(majority, y_test.size)
    res = {
        'acc.': accuracy_score(y_test, preds),
        'bacc.': balanced_accuracy_score(y_test, preds, adjusted=False),
        'f1': f1_score(y_test, preds)
    }
    return res


def pred_random(y_classes, y_test, seed, rng, ratios=None):
    preds = rng.choice(y_classes, y_test.size, replace=True, p=ratios)
    res = {
        'acc.': accuracy_score(y_test, preds),
        'bacc.': balanced_accuracy_score(y_test, preds, adjusted=False),
        'f1': f1_score(y_test, preds)
    }
    return res


def pred_GaussianNB(X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    preds = clf.fit(X_train, y_train).predict(X_test)
    res = {
        'acc.': clf.score(X_test, y_test),
        'bacc.': balanced_accuracy_score(y_test, preds, adjusted=False),
        'f1': f1_score(y_test, preds)
    }
    return res


def get_baseline_kfold(X, y, seed, target_class, n_splits, shuffle):
    # get labels corresponding to target class
    if target_class == 'arousal':
        y = y[:, 0]
    elif target_class == 'valence':
        y = y[:, 1]

    # initialize random number generator and fold generator
    rng = default_rng(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    results = {}
    # for each fold, split train & test and get classification results
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_classes, y_counts = np.unique(y_train, return_counts=True)
        majority = y_classes[np.argmax(y_counts)]
        class_ratios = y_counts / y_train.size

        results[i+1] = {
            'Gaussian NB': pred_GaussianNB(X_train, y_train, X_test, y_test),
            'Random': pred_random(y_classes, y_test, seed, rng),
            'Majority': pred_majority(majority, y_test),
            'Class ratio': pred_random(y_classes, y_test, seed, rng, ratios=class_ratios)
        }

    # return results as table
    results = {(fold, classifier): values for (fold, _results) in results.items() for (classifier, values) in _results.items()}
    results_table = pd.DataFrame.from_dict(results, orient='index').stack().unstack(level=1).rename_axis(['Fold', 'Metric'])
    return results_table[['Gaussian NB', 'Random', 'Majority', 'Class ratio']]


if __name__ == "__main__":
    # initialize parser
    parser = argparse.ArgumentParser(description='Preprocess K-EmoCon dataset and get baseline classification results.')
    parser.add_argument('--root', '-r', type=str, required=True, help='path to the dataset directory')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed for random number generation')
    parser.add_argument('--target', '-t', type=str, default='valence', help='target label for classification, must be either "valence" or "arousal"')
    parser.add_argument('--length', '-l', type=int, default=5, help='number of consecutive 5s-signals in one segment')
    parser.add_argument('--which', '-w', type=str, default='last', help='which label to set for segments, must be either "last" or "majority"')
    parser.add_argument('--rolling', default=False, action='store_true', help='get segments with rolling: e.g., s1=[0:n], s2=[1:n+1], ...')
    parser.add_argument('--no_rolling', dest='rolling', action='store_false', help='get segments without rolling: e.g., s1=[0:n], s2=[n:2n], ...')
    parser.add_argument('--splits', '-k', type=int, default=5, help='number of fold in k-fold stratified classification')
    parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle data before splitting to folds')
    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false', help="don't shuffle data before splitting to folds")
    args = parser.parse_args()

    if args.target not in ['valence', 'arousal']:
        raise ValueError(f'target must be either "valence" or "arousal", but given {args.target}')

    if args.which not in ['last', 'majority']:
        raise ValueError(f'which must be either "last" or "majority", but given {args.which}')

    # initialize default logger
    logger = init_logger()

    # filter these RuntimeWarning messages
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in true_divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in subtract')

    # get features and labels
    segments_dir = os.path.expanduser(args.root)
    logger.info(f'Processing segments from {segments_dir}, with: seed={args.seed}, target={args.target}, length={args.length*5}s, which={args.which}, rolling={args.rolling}, splits={args.splits}, shuffle={args.shuffle}')
    features, labels = prepare_kemocon(segments_dir, args.length, args.which, args.rolling)
    logger.info('Processing complete.')

    # get classification results
    results = get_baseline_kfold(features, labels, args.seed, args.target, args.splits, args.shuffle)
    
    # print summary of classification results
    print(results.groupby(level='Metric').mean())

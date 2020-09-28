import os
import json
import logging
import argparse
import numpy as np
import pandas as pd

from collections import namedtuple
from datetime import datetime, timedelta
from utils.logging import init_logger


def aggregate_raw(paths, valid_pids):
    '''Aggregate raw data files by participant IDs and returns a dict.

    Args:
        paths (dict of str: paths to K-EmoCon dataset files): requires,
            e4_dir (str): path to a directory of raw Empatica E4 data files saved as CSV files
            h7_dir (str): path to a directory of raw Polar H7 data files saved as CSV files

        valid_pids (list of int): a list containing valid participant IDs

    Returns:
        pid_to_raw_df (dict of int: pandas DataFrame): maps participant IDs to DataFrames containing raw data.

    '''
    logger = logging.getLogger('default')
    e4_dir, h7_dir = paths['e4_dir'], paths['h7_dir']
    pid_to_raw_df = {}

    # store raw e4 data
    for pid in valid_pids:
        # get paths to e4 data files
        user_dir = os.path.join(e4_dir, str(pid))
        filepaths = [os.path.join(user_dir, f) for f in os.listdir(user_dir) if f != '.ipynb_checkpoints']

        # store e4 data files to dict as k = "{uid}/{filetype}" -> v = DataFrame
        for filepath in filepaths:
            try:
                filetype = filepath.split('/')[-1].split('_')[-1].split('.')[0].lower()
                filekey = f'{pid}/{filetype}'
                data = pd.read_csv(filepath)

                # take care of multi-entry issue
                if pid == 31 and filetype == 'ibi':
                    data = data.loc[data.device_serial == 'A01525']
                elif pid == 31:
                    data = data.loc[data.device_serial == 'A013E1']
                elif pid == 32:
                    data = data.loc[data.device_serial == 'A01A3A']

                pid_to_raw_df[filekey] = data

            except Exception as err:
                logger.warning(f'Following exception occurred while processing {filekey}: {err}')

    # store raw h7 data
    for pid in valid_pids:
        # get path to h7 data file
        filepath = os.path.join(h7_dir, str(pid), 'Polar_HR.csv')

        # store h7 data files to dict as k = "{uid}/ecg" -> v = DataFrame
        try:
            filekey = f'{pid}/ecg'
            pid_to_raw_df[filekey] = pd.read_csv(filepath)

        except Exception as err:
            logger.warning(f'Following exception occurred while processing {filekey}: {err}')

    return pid_to_raw_df


def get_baseline_and_debate(paths, valid_pids, filetypes, pid_to_raw_df):
    '''Split aggregated raw data files into baseline and debate dataframes.

    Args:
        paths (dict of str: paths to K-EmoCon dataset files): requires,
            e4_dir (str): path to a directory of raw Empatica E4 data files saved as CSV files
            h7_dir (str): path to a directory of raw Polar H7 data files saved as CSV files

        valid_pids (list of int)

        filetypes (list of str)

    Returns:
        pid_to_baseline_raw (dict of int: (dict of str: pandas Series))

        pid_to_debate_raw (dict of int: (dict of str: pandas Series))

    '''
    logger = logging.getLogger('default')
    subject_info_table = pd.read_csv(paths['subjects_info_path'], index_col='pid')
    pid_to_baseline_raw = {pid:dict() for pid in valid_pids}
    pid_to_debate_raw = {pid:dict() for pid in valid_pids}

    # for each participant
    for pid in valid_pids:
        print('-' * 80)
        # get session info and timestamps
        subject_info = subject_info_table.loc[subject_info_table.index == pid]
        init_time, start_time, end_time = tuple(subject_info[['initTime', 'startTime', 'endTime']].to_numpy()[0])

        # get baseline interval
        baseline_td = 2 * 60 * 1e3  # 2 minutes (120s) in milliseconds
        baseline_start, baseline_end = init_time, init_time + baseline_td
        
        # for each filetype
        for filetype in filetypes:
            filekey = f'{pid}/{filetype}'
            try:
                data = pid_to_raw_df[filekey]
            except KeyError as err:
                logger.warning(f'Following exception occurred: {err}')
                continue

            # get baseline and debate portions of data
            baseline = data.loc[lambda x: (x.timestamp >= baseline_start) & (x.timestamp < baseline_end)]
            debate = data.loc[lambda x: (x.timestamp >= start_time) & (x.timestamp < end_time)]

            # skip the process if debate data is missing
            if debate.empty:
                logger.warning(f'Debate data missing for {filekey}, skipped')
                continue
            else:
                # try storing data with corresponding filekeys and printing extra information
                debate_len = datetime.fromtimestamp(max(debate.timestamp) // 1e3) - datetime.fromtimestamp(min(debate.timestamp) // 1e3)
                pid_to_debate_raw[pid][filetype] = debate.set_index('timestamp').value  # is a series

                # however, baseline data might be missing, so take care of that
                try:
                    baseline_len = datetime.fromtimestamp(max(baseline.timestamp) // 1e3) - datetime.fromtimestamp(min(baseline.timestamp) // 1e3)
                    pid_to_baseline_raw[pid][filetype] = baseline.set_index('timestamp').value  # is a series
                    print(f'For {filekey}:\t baseline - {baseline_len}: {len(baseline):5} \t|\t debate - {debate_len}: {len(debate):5}')
                except ValueError:
                    print(f'WARNING - Baseline data missing for {filekey} \t|\t debate - {debate_len}: {len(debate):5}')

    print('-' * 80)
    return pid_to_baseline_raw, pid_to_debate_raw


def baseline_to_json(paths, pid_to_baseline_raw):
    save_dir = paths['baseline_dir']
    # create a new directory if there isn't one already
    os.makedirs(save_dir, exist_ok=True)

    # for each participant
    for pid, baseline in pid_to_baseline_raw.items():

        # resample and interpolate ECG signals as they have duplicate entries while the intended frequency of ECG is 1Hz
        if 'ecg' in baseline.keys():
            ecg = baseline['ecg']
            ecg.index = pd.DatetimeIndex(ecg.index * 1e6)
            ecg = ecg.resample('1S').mean().interpolate(method='time')
            ecg.index = ecg.index.astype(np.int64) // 1e6
            baseline['ecg'] = ecg

        # convert sig values to list
        baseline = {sigtype: sig.values.tolist() for sigtype, sig in baseline.items() if sigtype in ['bvp', 'eda', 'temp', 'ecg']}

        # save baseline as json file
        savepath = os.path.join(save_dir, f'p{pid:02d}.json')
        with open(savepath, 'w') as f:
            json.dump(baseline, f, sort_keys=True, indent=4)

    return


def debate_segments_to_json(paths, valid_pids, filetypes, pid_to_debate_raw):
    subject_info_table = pd.read_csv(paths['subjects_info_path'], index_col='pid')
    Ratings = namedtuple('Ratings', ['values', 'len'])
    save_dir = paths['segments_dir']

    # for each participant:
    print('-' * 100)
    print(f'{"pid"}\t{"debate_len":>10}\t{"sig_len":>10}\t{"s_len":>10}\t{"p_len":>10}\t{"e_len":>10}\t{"num_seg":>10}')
    for pid in valid_pids:
        os.makedirs(os.path.join(save_dir, str(pid)), exist_ok=True)
        signals = dict()

        # get session info and timestamps
        subject_info = subject_info_table.loc[subject_info_table.index == pid]
        debate_start, debate_end = tuple(subject_info[['startTime', 'endTime']].values[0])
        debate_len = debate_end - debate_start
        
        # load debate data and self/partner/external annotation files
        debate = pid_to_debate_raw[pid]
        ratings = {
            's': pd.read_csv(os.path.join(paths['self_ratings_dir'], f'P{pid}.self.csv')),
            'p': pd.read_csv(os.path.join(paths['partner_ratings_dir'], f'P{pid}.partner.csv')),
            'e': pd.read_csv(os.path.join(paths['external_ratings_dir'], f'P{pid}.external.csv'))
        }

        # save ratings information as (annotations, total duration of annotation in milliseconds)
        for tag in ['s', 'p', 'e']:
            ratings[tag] = Ratings(ratings[tag], int(ratings[tag].seconds.values[-1] * 1e3))

        # first, cut annotations longer than debate_len from their beginnings
        for tag in ['s', 'p', 'e']:
            if ratings[tag].len >= debate_len:
                ratings[tag] = ratings[tag]._replace(values=ratings[tag].values[-int(debate_len // 5e3):].reset_index(drop=True))
                ratings[tag] = ratings[tag]._replace(len=int((ratings[tag].values.index[-1] + 1) * 5e3))

        # find common timerange for signals
        sig_start, sig_end = 0, np.inf
        for sigtype in ['bvp', 'eda', 'temp', 'ecg']:
            sig_start = int(max(sig_start, debate[sigtype].index[0]))
            sig_end = int(min(sig_end, debate[sigtype].index[-1]))
        start_gap = sig_start - debate_start
        end_gap = debate_end - sig_end
        sig_len = sig_end - sig_start
        overlap = min(sig_len, ratings['s'].len, ratings['p'].len, ratings['e'].len)

        # second, cut singals w.r.t. common timerange
        for sigtype in ['bvp', 'eda', 'temp', 'ecg']:
            sig = debate[sigtype].loc[lambda x: (x.index >= sig_start) & (x.index < sig_end)]
            sig.index = sig.index.astype('float64')

            # also adjust start and end points of signals
            # this is necessary for consistency, as we want our signals within the overlapping duration
            diff = sig.index[-1] - sig.index[0] - overlap
            if diff > 0:
                start_diff = diff * start_gap / (start_gap + end_gap)
                end_diff = diff * end_gap / (start_gap + end_gap)
                sig = sig.loc[lambda x: (x.index >= sig.index[0] + start_diff) & (x.index < sig.index[-1] - end_diff)]

            # resample and interpolate only ECG signals as they have issues with some duplicate entries
            # while the frequency of recorded ECG is 1Hz, there are 1s-intervals where multiple values entered within the period
            if sigtype == 'ecg':
                sig.index = pd.DatetimeIndex(sig.index * 1e6)
                sig = sig.resample('1S').mean()
                sig.interpolate(method='time', inplace=True)
                sig.index = sig.index.astype(np.int64) // 1e6

            signals[sigtype] = sig
            sig_len = min(sig_len, sig.index[-1] - sig.index[0])

        # finally, cut annotations from their beginnings to match their lengths with each other
        min_len = min([ratings['s'].len, ratings['p'].len, ratings['e'].len])
        for tag in ['s', 'p', 'e']:
            if ratings[tag].len > min_len:
                ratings[tag] = ratings[tag]._replace(values=ratings[tag].values[int((ratings[tag].len - min_len) // 5e3):].reset_index(drop=True))
                ratings[tag] = ratings[tag]._replace(len=int((ratings[tag].values.index[-1] + 1) * 5e3))

            # and their sides to match their lengths with signals (approximately equal to overlap)
            diff = (ratings[tag].len - overlap) // 5e3
            if diff > 0:
                start_diff = round(diff * start_gap / (start_gap + end_gap))
                end_diff = round(diff * end_gap / (start_gap + end_gap))
                start_diff = start_diff + 1 if start_diff != 0 else start_diff

                ratings[tag] = ratings[tag]._replace(values=ratings[tag].values[int(start_diff):len(ratings[tag].values) - int(end_diff)].reset_index(drop=True))
                ratings[tag] = ratings[tag]._replace(len=int((ratings[tag].values.index[-1] + 1) * 5e3))

        # find the greatest possible number of 5-second segments we can extract
        # the greatest possible number of segments is the maximum overlap across the length of debate data, self annotations, partner annotations, and external annotations divided by 5
        num_seg = int(sig_len // 5e3)
        print(f"{pid}\t{debate_len:>10.0f}\t{sig_len:>10.0f}\t{ratings['s'].len:>10}\t{ratings['p'].len:>10}\t{ratings['e'].len:>10}\t{num_seg:>10}")

        # get segments and save them as json files
        for i in range(num_seg):
            s_values = ratings['s'].values.iloc[i]
            p_values = ratings['p'].values.iloc[i]
            e_values = ratings['e'].values.iloc[i]
            s_a, s_v = s_values.arousal, s_values.valence
            p_a, p_v = p_values.arousal, p_values.valence
            e_a, e_v = e_values.arousal, e_values.valence

            seg = dict()
            for sigtype in ['bvp', 'eda', 'temp', 'ecg']:
                sig = signals[sigtype]
                start = sig.index[0] + (i * 5e3)
                seg[sigtype] = sig.loc[lambda x: (x.index >= start) & (x.index < start + 5e3)].tolist()
            
            seg_savepath = os.path.join(save_dir, str(pid), f'p{pid:02d}-{i:03d}-{s_a}{s_v}{p_a}{p_v}{e_a}{e_v}.json')
            with open(seg_savepath, 'w') as f:
                json.dump(seg, f, sort_keys=True, indent=4)
                
    print('-' * 100)
    return


if __name__ == "__main__":
    # initialize parser
    parser = argparse.ArgumentParser(description='Preprocess K-EmoCon dataset and save BVP, EDA, HST, and ECG signals as JSON files.')
    parser.add_argument('--root', '-r', type=str, required=True, help='a path to a root directory for the dataset')
    args = parser.parse_args()

    # initialize default logger and constants
    logger = init_logger()
    logger.info(f'Read/writing files to {args.root}...')
    PATHS = {
        'e4_dir': os.path.expanduser(os.path.join(args.root, 'raw/e4_data')),
        'h7_dir': os.path.expanduser(os.path.join(args.root, 'raw/neurosky_polar_data')),
        'subjects_info_path': os.path.expanduser(os.path.join(args.root, 'metadata/subjects.csv')),
        'self_ratings_dir': os.path.expanduser(os.path.join(args.root, 'raw/emotion_annotations/self_annotations')),
        'partner_ratings_dir': os.path.expanduser(os.path.join(args.root, 'raw/emotion_annotations/partner_annotations')),
        'external_ratings_dir': os.path.expanduser(os.path.join(args.root, 'raw/emotion_annotations/aggregated_external_annotations')),
        'baseline_dir': os.path.expanduser(os.path.join(args.root, 'baseline')),
        'segments_dir': os.path.expanduser(os.path.join(args.root, 'segments')),
    }
    VALIDS = [1, 4, 5, 8, 9, 10, 11, 13, 14, 15, 16, 19, 22, 23, 24, 25, 26, 27, 28, 31, 32]
    FILETYPES = ['bvp', 'eda', 'hr', 'ibi', 'temp', 'ecg']

    # aggregate raw data
    logger.info('Preprocessing started, aggregating raw files...')
    pid_to_raw_df = aggregate_raw(PATHS, VALIDS)
    
    # get baseline and debate data
    logger.info('Getting baseline and debate data...')
    pid_to_baseline_raw, pid_to_debate_raw = get_baseline_and_debate(PATHS, VALIDS, FILETYPES, pid_to_raw_df)

    # save baseline data as json
    logger.info(f'Saving baseline as JSON files to {PATHS["baseline_dir"]}...')
    baseline_to_json(PATHS, pid_to_baseline_raw)

    # save 5s-segments
    logger.info(f'Saving debate segments as JSON files to {PATHS["segments_dir"]}...')
    debate_segments_to_json(PATHS, VALIDS, FILETYPES, pid_to_debate_raw)
    logger.info('Preprocessing complete.')

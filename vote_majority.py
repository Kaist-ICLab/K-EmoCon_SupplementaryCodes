import numpy as np
import pandas as pd


def get_rater_agreement(group):
    '''
    Implements majority voting.
    
    Parameters
    ----------
    group: a Pandas DataFrame.

        * A group is a Pandas dataframe of annotations from five external raters for emotions of a participant
        observed during a specific 5 second interval.

        E.g., 
                         arousal  valence  cheerful  happy  angry  nervous  sad boredom  \
            rid seconds                                                                   
            1   5              3        3         3      2      1        2    1     NaN   
            5   5              1        2         1      1      1        1    3     NaN   
            2   5              1        3         2      2      1        1    1     NaN   
            4   5              1        4         2      3      1        1    1     NaN   
            3   5              1        3         1      1      1        1    1     NaN   

                        confusion delight  ... surprise  none_1 confrustion contempt  \
            rid seconds                    ...                                         
            1   5             NaN     NaN  ...      NaN     NaN         NaN      NaN   
            5   5             NaN     NaN  ...      NaN     NaN         NaN      NaN   
            2   5             NaN     NaN  ...      NaN       x         NaN      NaN   
            4   5             NaN     NaN  ...      NaN     NaN         NaN      NaN   
            3   5             NaN     NaN  ...      NaN       x         NaN      NaN   

                        dejection  disgust  eureka  pride sorrow  none_2  
            rid seconds                                                   
            1   5             NaN      NaN     NaN    NaN    NaN       x  
            5   5             NaN      NaN     NaN    NaN      x     NaN  
            2   5             NaN      NaN     NaN    NaN    NaN       x  
            4   5             NaN      NaN     NaN    NaN    NaN       x  
            3   5             NaN      NaN     NaN    NaN    NaN       x  

        As shown, a group must contain five rows with the same values for 'seconds', but different values for 'rid'.

    Returns
    -------
    A pandas Series of aggregated annotations.
    
    '''

    # create lists of strings of emotion categories
    scaled_emotions = ['arousal', 'valence', 'cheerful', 'happy', 'angry', 'nervous', 'sad']
    binary_emotions_1 = ['boredom', 'confusion', 'delight', 'concentration', 'frustration', 'surprise', 'none_1']
    binary_emotions_2 = ['confrustion', 'contempt', 'dejection', 'disgust', 'eureka', 'pride', 'sorrow', 'none_2']

    # create an empty dictionary to save emotion annotations aggregated across raters
    aggregated_row = {}
    
    # for each emotion measured in ordinal scales,
    for emotion in scaled_emotions:
        # first find a mode for the column of annotations for the emotion
        majority = group[emotion].mode()

        # if there is more than 1 mode for the emotion,
        if len(majority) > 1:
            # randomly select only one value as the mode (majority)
            majority = majority.sample(n=1)

        # get the mode value and save that to the dictionary with an emotion as the key
        value = majority.to_list()[0]
        aggregated_row[emotion] = value

    # for each set of emotions measured in binary scales,
    for emotions in [binary_emotions_1, binary_emotions_2]:
        # create an empty list to store counts
        counts = []

        # for each emotion in the set of emotions (a column),
        for emotion in emotions:
            # count the number of non-nan cells in the column ('x's)
            x_counts = group[emotion].value_counts().to_numpy()

            # if there was more than zero non-nan cells,
            if len(x_counts) > 0:
                # append a tuple of (emotion, counts of non-nan cells) to the list
                counts.append((emotion, x_counts[0]))

        # after counting the number of non-nan cells for all emotions (columns), randomly shuffle the list,
        # this is to sample at random if there is a tie for the maximum
        np.random.shuffle(counts)
        # then choose the maximum from the list with the number of non-nan cells as the key
        max_emotion, _ = max(counts, key=lambda emotion_count: emotion_count[1])

        # if emotion is equal to the maximum emotion, then set the value equal to 'x', or nan
        for emotion in emotions:
            if emotion == max_emotion:
                value = 'x'
            else:
                value = np.nan

            # save the value to the dictionary with an emotion as they key
            aggregated_row[emotion] = value

    # after iterating over all emotions, return a pandas Series generated with the dict where (k, v) = (emotion, value)
    return pd.Series(aggregated_row)


def aggregate_by_majority_voting(annotations):
    '''
    Implements aggregation of external rater annotations by majority voting.

    Parameters
    ----------
    annotations: list of Pandas DataFrame

        * All dataframes in the list should contains annotations for one participant, each from 5 different raters.
        For example, annotations == [r1, r2, r3, r4, r5], where r<n> is a dataframe of annotations from an n_th rater.

        * A dataframe in the list must have a column named 'rid', which contains a unique number indentifying a rater.

    Returns
    -------
    aggregated: a Pandas Dataframe of annotations aggregated by majority voting.
    
    '''
    
    # first concatenate dataframes along index (vertically), ignoring indices without sorting
    concatenated = pd.concat(annotations, ignore_index=True, sort=False)

    # set multilevel index with columns 'rid' and 'seconds', then group the entire dataframe by 'seconds'
    grouped = concatenated.set_index(['rid', 'seconds']).groupby('seconds')

    # apply get_rater_agreement function to a grouped dataframe
    aggregated = grouped.apply(get_rater_agreement)

    # reset the 'seconds' column set as the index of aggregated dataframe
    aggregated.reset_index(inplace=True)

    return aggregated
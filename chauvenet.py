from scipy import special


def criterion(data):
    '''
    Implements Chauvenet's criterion for outlier detection.
    
    * See: https://en.wikipedia.org/wiki/Chauvenet%27s_criterion

    Parameters
    ----------
    data: a Pandas DataFrame or named Series.
    
    Returns
    -------
    A masked object of same shape and type as data, with Trues in where input 
    values were outliers, and Falses in where values were not outliers.
    
    '''

    mean, std, N = data.mean(), data.std(), len(data)  # mean, standard deviation, and length of input data
    criterion = 1.0 / (2 * N)                          # Chauvenet's criterion
    d = abs(data - mean) / std                         # distance of a value to mean in stdev.'s
    prob = special.erfc(d)                             # area of normal dist.
    
    return prob < criterion                            # if prob is below criterion, a value is an outlier
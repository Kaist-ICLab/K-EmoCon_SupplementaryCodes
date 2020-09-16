import os
import numpy as np
import pandas as pd
import krippendorff
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy import stats


def get_annotations(savepath, external=False):
    '''
    Load annoations saved as CSV.

    Parameters
    ----------
    savepath: string
        a path to a directory of annotations saved as CSV files.
    external: bool, default False
        should be set to True if the savepath contains unaggregated annotations from external raters
    
    Returns
    -------
    annotations: dict
        of the form {'pid': dataframe} if external = False,
        or of the form {'pid': list} where the list is of the form [df1, df2, ..., df5] if external = True.
    
    '''

    annotations = {}

    if not external:
        for filename in os.listdir(savepath):
            if 'csv' in filename:
                filepath = os.path.join(savepath, filename)
                pid, data = int(filename.split('.')[0][1:]), pd.read_csv(filepath)
                annotations[pid] = data
                
    else:
        for filename in os.listdir(savepath):
            if 'csv' in filename:
                filepath = os.path.join(savepath, filename)

                pid, rid, data = int(filename.split('.')[0][1:]), int(filename.split('.')[1][1:]), pd.read_csv(filepath)
                rater_annotations_list = annotations.setdefault(pid, [])
                
                data['rid'] = rid  # create a new column 'rid' containing a rater number
                rater_annotations_list.append(data)
    
    return annotations


def subtract_mode_from_values(annotations, emotion):
    '''
    Subtract mode from given annotations of an emotion.

    Parameters
    ----------
    annotations: DataFrame
        of annotations for a given emotion.
    emotion: string
        an emotion category name as string.
    
    Returns
    -------
    annotations: DataFrame
        of mode-subtracted annotations.
    
    '''

    if set(emotion).issubset(set(['arousal', 'valence'])):
        annotations = annotations - 3
    else:
        annotations = annotations - annotations.mode().iloc[0].values
    
    return annotations


def compute_krippendorff_alpha(emotion, all_annotations, subtract_mode=False):
    '''
    Compute Krippendorff's alphas given dict of annotations.

    Parameters
    ----------
    emotion: string
        an emotion category name as string.
    all_annotations: dict
        of the form {'perspective': dict}, where sub-dicts are of the form {'pid': dataframe}
    subtract_mode: bool, default False
        Apply mode-subtraction to all values for computing Krippendorff's alpha if set to True.
    
    Returns
    -------
    alphas: a Pandas DataFrame of Krippendorff's alphas.
    
    '''

    result = {'SP': [], 'SE': [], 'PE': [], 'All': []}
    self_annotations = all_annotations['self']
    partner_annotations = all_annotations['partner']
    external_annotations = all_annotations['external']

    for pid in range(1, 33):
        self_df = self_annotations[pid][[emotion]]
        partner_df = partner_annotations[pid][[emotion]]
        external_df = external_annotations[pid][[emotion]]
        
        # truncate longer dfs from beginning
        if len(self_df) >= len(external_df):
            self_df = self_df[len(self_df)-len(external_df):]
            partner_df = partner_df[len(partner_df)-len(external_df):]
        else:
            external_df = external_df[len(external_df)-len(self_df):]
            
        if subtract_mode:
            self_df = subtract_mode_from_values(self_df, emotion)
            partner_df = subtract_mode_from_values(partner_df, emotion)
            external_df = subtract_mode_from_values(external_df, emotion)

        values_self = self_df.to_numpy().flatten('F')
        values_partner = partner_df.to_numpy().flatten('F')
        values_ext_agg = external_df.to_numpy().flatten('F')

        # stack annotations into an array of shape (m, n) where m = number of raters & n = number of annotations
        reliability_data_sp = np.stack([values_self, values_partner], axis=0)
        reliability_data_se = np.stack([values_self, values_ext_agg], axis=0)
        reliability_data_pe = np.stack([values_partner, values_ext_agg], axis=0)
        reliability_data_all = np.stack([values_self, values_partner, values_ext_agg], axis=0)

        k_alpha_sp = krippendorff.alpha(reliability_data_sp, level_of_measurement='ordinal')
        k_alpha_se = krippendorff.alpha(reliability_data_se, level_of_measurement='ordinal')
        k_alpha_pe = krippendorff.alpha(reliability_data_pe, level_of_measurement='ordinal')
        k_alpha_all = krippendorff.alpha(reliability_data_all, level_of_measurement='ordinal')
        
        result['SP'].append(k_alpha_sp)
        result['SE'].append(k_alpha_se)
        result['PE'].append(k_alpha_pe)
        result['All'].append(k_alpha_all)
    
    return pd.DataFrame.from_dict(result, orient='index', columns=list(range(1, 33)))


def add_mean_and_diff(alphas):
    '''
    Adds mean across columns and the last row of difference between self-external annotations and self-partner annotations.

    Parameters
    ----------
    alphas: a Pandas Dataframe of Krippendorff's alphas.
    
    Returns
    -------
    alphas.T: a Pandas Dataframe of Krippendorff's alphas, with mean column and diff row added.
    
    '''

    alphas['Mean'] = alphas.mean(axis=1)
    alphas = alphas.T

    alphas['Diff. [SE - SP]'] = alphas['SE'] - alphas['SP']
    
    return alphas.T


def get_alphas(emotion, all_annotations, subtract_mode=False):
    '''
    Compute Krippendorff's alphas.

    Parameters
    ----------
    emotions: string
        an emotion category name to compute IRR for.
    all_annotations: dict
        of the form {'perspective': dict}, where sub-dicts are of the form {'pid': dataframe}
    subtract_mode: bool, default False
        apply mode-subtraction to all values for computing Krippendorff's alpha if set to True.
    
    Returns
    -------
    alphas: a Pandas DataFrame of Krippendorff's alphas. The first 4 rows of each dataframe contain alpha coefficients 
    across four different combinations of annotation perspectives: (1) SP = self vs. partner, (2) SE = self vs. external, 
    (3) PE = partner vs. external, and (4) All = self vs. partner vs. external, while the last row 'Diff [SE - SP]' shows 
    the difference between self vs. external agreement and self vs. partner agreement. The columns show those of each 
    participant.
    
    '''

    alphas = compute_krippendorff_alpha(emotion, all_annotations, subtract_mode)
    alphas = add_mean_and_diff(alphas)
    
    return alphas


# http://chris35wills.github.io/matplotlib_diverging_colorbar/
# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# modified from:
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", fontsize=16, title=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    
    H, W = data.shape
    
    if not ax:
        _, ax = plt.subplots(figsize=(W, H))

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=fontsize)
    ax.set_yticklabels(row_labels, fontsize=fontsize)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    
    if title is not None:
        ax.set_title(title, fontsize=20)
    
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


# modified from:
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm((data.min()*0.7, data.max()*0.7))
    
    threshold_l, threshold_r = threshold
    
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold_r or im.norm(data[i, j]) < threshold_l)])
            
            if abs(data[i, j]) >= 0.01:
                valfmt = "{x:.2f}"
            elif abs(data[i, j]) >= 0.001:
                valfmt = "{x:.3f}"
            elif abs(data[i, j]) > 0:
                valfmt = u"≈ 0"
            elif data[i, j] == 0.0:
                valfmt = "0"
            else:
                valfmt = "n/a"
                
            # Get the formatter in case a string is supplied
            if isinstance(valfmt, str):
                valfmt = mpl.ticker.StrMethodFormatter(valfmt)
            
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_heatmaps(all_annotations):
    '''
    Plots heatmaps for seven emotions of 'arousal', 'valence', 'cheerful', 'happy', 'angry', 'nervous', and 'sad'.
    
    Parameters
    ----------
    all_annotations: dict
        of the form {'perspective': dict}, where sub-dicts are of the form {'pid': dataframe}
        
    Returns
    -------
    None
    
    '''

    emotions = ['arousal', 'valence', 'cheerful', 'happy', 'angry', 'nervous', 'sad']
    fig, axes = plt.subplots(ncols=1, nrows=len(emotions), figsize=(33, 5*len(emotions)))
    
    for i, emotion in enumerate(emotions):
        alphas = get_alphas(emotion, all_annotations, subtract_mode=True)
        data = alphas.to_numpy()
        
        clim = -1, 1
        row_labels = alphas.index.to_list()
        col_labels = alphas.columns.to_list()
        normalize = MidpointNormalize(midpoint=0, vmin=np.nanmin(data), vmax=np.nanmax(data))
        
        im, _ = heatmap(
            data, row_labels, col_labels, ax=axes[i],
            cmap='coolwarm', cbarlabel="Krippendorff's alpha", 
            fontsize=18, title=emotion.capitalize(), clim=clim, norm=normalize
            )
        
        annotate_heatmap(im, valfmt="{x:.3f}", threshold=(-0.75, 0.75), fontsize=14)

    fig.tight_layout()
    
    return
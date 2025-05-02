import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter


def create_combined_rank_frequency_plot(datasets_dict,
                                        figsize=(10, 6),
                                        colors=None,
                                        markers=None,
                                        dpi=300,
                                        include_line=True,
                                        xlabel='Rank',
                                        ylabel='Number of Queries',
                                        legend_loc='upper right',
                                        yticks=None,
                                        log_y=False,
                                        y_min=None,
                                        y_max=None,
                                        linthresh=1):
    """
    Create a rank-frequency plot for multiple datasets combined in one chart.

    Parameters:
    -----------
    datasets_dict : dict
        Nested dictionary in the form {scenario: {label: num_queries, ...}, ...}
    figsize : tuple, optional
    colors : list, optional
    markers : list, optional
    dpi : int, optional
    include_line : bool, optional
    xlabel : str, optional
    ylabel : str, optional
    legend_loc : str, optional
    yticks : list, optional
        Explicit y-tick values to show
    log_y : bool or str, optional
        Set to 'log' for log scale (zero counts excluded),
        'symlog' for symmetric log scale (zero counts included),
        or False for linear scale.
    y_min : float, optional
    y_max : float, optional
    linthresh : float, optional
        Linear threshold for symlog scale (default 1)

    Returns:
    --------
    fig, ax : tuple
        Figure and axes objects for further customization if needed
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if colors is None:
        colors = plt.cm.tab10.colors
    if markers is None:
        markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h']

    for i, (dataset_name, label_counts) in enumerate(datasets_dict.items()):
        df = pd.DataFrame({
            'label': list(label_counts.keys()),
            'count': list(label_counts.values())
        })

        df = df.sort_values('count', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df['count'] = df['count'].astype(float)

        # Handle log/symlog transformations
        if log_y == 'log':
            df = df[df['count'] > 0]  # Cannot plot zero on log scale
        elif log_y == 'symlog':
            df['count'] = df['count'].fillna(0)  # Symlog allows 0

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.scatter(df['rank'], df['count'],
                   color=color,
                   marker=marker,
                   s=50,
                   alpha=0.7,
                   label=dataset_name,
                   zorder=3)

        if include_line:
            ax.plot(df['rank'], df['count'],
                    color=color,
                    alpha=0.4,
                    linewidth=1.5,
                    zorder=2)

    # Set axis scale
    if log_y == 'log':
        ax.set_yscale('log', base=2)
    elif log_y == 'symlog':
        ax.set_yscale('symlog', linthresh=linthresh, linscale=0.5, base=2)

    if log_y in ('log', 'symlog'):
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    # Custom y-ticks
    if yticks is not None:
        ax.set_yticks(yticks)

    fontsize=20

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(True, linestyle='--', alpha=0.7, zorder=1)
    ax.legend(loc=legend_loc, frameon=True, framealpha=0.9, fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.tight_layout()
    return fig, ax

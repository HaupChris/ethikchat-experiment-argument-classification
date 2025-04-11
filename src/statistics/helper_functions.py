import matplotlib.pyplot as plt
import pandas as pd

def create_combined_rank_frequency_plot(datasets_dict,
                                        figsize=(10, 6),
                                        colors=None,
                                        markers=None,
                                        dpi=300,
                                        include_line=True,
                                        xlabel='Rank',
                                        ylabel='Number of Queries',
                                        legend_loc='upper right'):
    """
    Create a rank-frequency plot for multiple datasets combined in one chart.

    Parameters:
    -----------
    datasets_dict : dict
        Nested dictionary in the form {scenario: {label: num_queries, ...}, ...}
    figsize : tuple, optional
        Figure size as (width, height) in inches
    colors : list, optional
        List of colors for each dataset. If None, default color cycle is used.
    markers : list, optional
        List of markers for each dataset. If None, default markers are used.
    output_file : str, optional
        Output filename (should end with .svg, .pdf, or other vector format)
    dpi : int, optional
        DPI for rasterized elements (not affecting vector elements)
    include_line : bool, optional
        Whether to include connecting lines between points
    title : str, optional
        Chart title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    legend_loc : str, optional
        Legend location

    Returns:
    --------
    fig, ax : tuple
        Figure and axes objects for further customization if needed
    """
    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Default colors and markers if not provided
    if colors is None:
        colors = plt.cm.tab10.colors  # Default matplotlib color cycle

    if markers is None:
        markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h']

    # Process each dataset
    for i, (dataset_name, label_counts) in enumerate(datasets_dict.items()):
        # Convert to dataframe for easier handling
        df = pd.DataFrame({
            'label': list(label_counts.keys()),
            'count': list(label_counts.values())
        })

        # Sort by count in descending order and add rank
        df = df.sort_values('count', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1

        # Remove zero counts if they exist (optional)
        df = df[df['count'] > 0]

        # Get color and marker for this dataset
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Plot scatter points
        ax.scatter(df['rank'], df['count'],
                   color=color,
                   marker=marker,
                   s=50,  # Marker size
                   alpha=0.7,  # Transparency
                   label=dataset_name,
                   zorder=3)  # Make sure points are on top of lines

        # Add connecting line if requested
        if include_line:
            ax.plot(df['rank'], df['count'],
                    color=color,
                    alpha=0.4,
                    linewidth=1.5,
                    zorder=2)

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    # ax.set_title(title, fontsize=14)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, zorder=1)

    # Add legend
    ax.legend(loc=legend_loc, frameon=True, framealpha=0.9)

    # Style tweaks for better readability
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Tight layout
    plt.tight_layout()

    return fig, ax




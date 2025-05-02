import os
from collections import defaultdict
from typing import Sequence, List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


def plot_confidence_threshold_performance(top_run_dirs, top_models_df, threshold_range=None):
    """
    Plot accuracy@1 vs confidence threshold for top models on normal and noisy queries

    Parameters:
    -----------
    top_run_dirs : list
        List of directories for the top models
    top_models_df : pandas DataFrame
        DataFrame containing information about the top models
    threshold_range : list, optional
        List of confidence thresholds to plot, defaults to reasonable range if None

    Returns:
    --------
    None (displays the plot)
    """
    # Set up plot aesthetics
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")

    # Define a color palette for the models
    colors = sns.color_palette("husl", len(top_run_dirs))

    # Process each model
    for i, run_dir in enumerate(top_run_dirs):
        model_name = top_models_df.iloc[i]['model_name']
        model_name_short = model_name.split('/')[-1]  # Get just the last part of the model name

        # Load normal queries confidence threshold data
        normal_path = os.path.join(run_dir, "confidence_threshold_metrics",
                                   "cosine_single_argument_classification_normal_queries.csv")

        # Load noisy queries confidence threshold data
        noisy_path = os.path.join(run_dir, "confidence_threshold_metrics",
                                  "cosine_single_argument_classification_noisy_queries.csv")

        if os.path.exists(normal_path) and os.path.exists(noisy_path):
            normal_df = pd.read_csv(normal_path)
            noisy_df = pd.read_csv(noisy_path)

            # If no specific thresholds provided, use those in the data
            if threshold_range is None:
                threshold_range = sorted(normal_df['Confidence'].unique())

            # Filter data to the specified thresholds
            normal_df = normal_df[normal_df['Confidence'].isin(threshold_range)]
            noisy_df = noisy_df[noisy_df['Confidence'].isin(threshold_range)]

            # Plot for normal queries
            plt.plot(normal_df['Confidence'], normal_df['Accuracy'],
                     marker='o', linestyle='-', color=colors[i],
                     label=f"{model_name_short} (Normal)")

            # Plot for noisy queries - use dashed line for distinction
            plt.plot(noisy_df['Confidence'], noisy_df['Accuracy'],
                     marker='x', linestyle='--', color=colors[i],
                     label=f"{model_name_short} (Noisy)")

    # Set plot labels and title
    plt.xlabel('Confidence Threshold', fontsize=14)
    plt.ylabel('Accuracy@1', fontsize=14)
    plt.title('Accuracy@1 vs Confidence Threshold for Normal and Noisy Queries', fontsize=16)

    # Improve legend readability
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    # Set appropriate y-axis limits
    plt.ylim(0, 1.05)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# Calculate false positive rates as well
def calculate_fp_rates(top_run_dirs, top_models_df, threshold_range=None):
    """
    Calculate False Positive Rate on noisy queries at different confidence thresholds

    False Positive Rate = Fraction of noisy queries incorrectly identified as containing an argument
    """
    results = []

    for i, run_dir in enumerate(top_run_dirs):
        model_name = top_models_df.iloc[i]['model_name']
        model_name_short = model_name.split('/')[-1]

        # Load noisy queries confidence threshold data
        noisy_path = os.path.join(run_dir, "confidence_threshold_metrics",
                                  "cosine_single_argument_classification_noisy_queries.csv")

        if os.path.exists(noisy_path):
            noisy_df = pd.read_csv(noisy_path)

            # If no specific thresholds provided, use those in the data
            if threshold_range is None:
                threshold_range = sorted(noisy_df['Confidence'].unique())

            # Filter data to the specified thresholds
            noisy_df = noisy_df[noisy_df['Confidence'].isin(threshold_range)]

            # Calculate FP rate as 1 - accuracy (since all predictions on noisy queries are false positives)
            for _, row in noisy_df.iterrows():
                fp_rate = 1.0 - row['Accuracy']
                results.append({
                    'model': model_name_short,
                    'threshold': row['Confidence'],
                    'fp_rate': fp_rate,
                    'queries_above_threshold': row['Queries_Above_Threshold']
                })

    return pd.DataFrame(results)


def plot_threshold_tradeoff(top_run_dirs, top_models_df, thresholds=None):
    """
    Plot accuracy vs. false positive rate tradeoffs, with markers
    at 10%, 20% FPR and at each model’s first max-accuracy point,
    and a legend showing line + marker style.
    """
    plt.figure(figsize=(12, 8), dpi=300)
    markers    = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    colors     = sns.color_palette("colorblind", len(top_run_dirs))

    # Prepare custom legend handles
    legend_handles = []
    legend_labels  = []

    model_names_indices = defaultdict(lambda: 1)
    # Process each model
    for i, run_dir in enumerate(top_run_dirs):

        model_name = top_models_df.iloc[i]['model_name']
        model_name_index = model_names_indices[model_name]
        model_names_indices[model_name] += 1
        model_name += f" {model_name_index}"
        short_name = model_name.split('/')[-1]

        # --- load your CSVs as before ---
        normal_df = pd.read_csv(os.path.join(
            run_dir, "confidence_threshold_metrics",
            "cosine_single_argument_classification_normal_queries.csv"))
        noisy_df  = pd.read_csv(os.path.join(
            run_dir, "confidence_threshold_metrics",
            "cosine_single_argument_classification_noisy_queries.csv"))

        # build combined DataFrame
        combined = []
        for thr in normal_df['Confidence'].unique():
            if thr in noisy_df['Confidence'].values:
                acc = normal_df.loc[normal_df['Confidence']==thr, 'Accuracy'].iloc[0]
                fp  = 1.0 - noisy_df.loc[noisy_df['Confidence']==thr, 'Accuracy'].iloc[0]
                combined.append((thr, fp, acc))
        model_df = pd.DataFrame(combined, columns=['threshold','fp_rate','accuracy'])

        if thresholds is not None:
            model_df = model_df[ model_df['threshold'].isin(thresholds) ]
        model_df = model_df.sort_values('fp_rate').reset_index(drop=True)

        fp  = model_df['fp_rate']
        acc = model_df['accuracy']

        # 1) plot the line only
        plt.plot(fp, acc,
                 color=colors[i],
                 linestyle='solid',
                 linewidth=2.0)

        # helper: index closest to a target fp
        def idx_closest(target):
            return (model_df['fp_rate'] - target).abs().idxmin()

        # 2) marker at ~10% FPR
        if model_df['fp_rate'].min() <= 0.1 <= model_df['fp_rate'].max():
            j = idx_closest(0.1)
            plt.scatter(fp[j], acc[j],
                        marker=markers[i % len(markers)],
                        s=100, color=colors[i], zorder=3)

        # 3) marker at ~20% FPR
        if model_df['fp_rate'].min() <= 0.2 <= model_df['fp_rate'].max():
            j = idx_closest(0.2)
            plt.scatter(fp[j], acc[j],
                        marker=markers[i % len(markers)],
                        s=100, color=colors[i], zorder=3)

        # 4) marker at first max-accuracy
        max_acc       = acc.max()
        first_max_idx = model_df[model_df['accuracy']==max_acc].index[0]
        plt.scatter(fp[first_max_idx], acc[first_max_idx],
                    marker=markers[i % len(markers)],
                    edgecolors='k', s=140, linewidths=1.5,
                    facecolor=colors[i], zorder=4)

        # build a proxy artist for legend: line + marker
        legend_handles.append(
            Line2D([0], [0],
                   color=colors[i],
                   linestyle='solid',
                   linewidth=2.0,
                   marker=markers[i % len(markers)],
                   markersize=8,
                   markerfacecolor=colors[i],
                   markeredgecolor='k')
        )
        legend_labels.append(short_name)

    plt.rcParams.update({'font.size': 15})
    # draw the legend with our custom handles
    plt.legend(legend_handles, legend_labels,
               loc='center right', fontsize=15)

    plt.xlabel('False Positive Rate on Noisy Queries', fontsize=15)
    plt.ylabel('Accuracy@1 on Normal Queries', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.title('Accuracy vs FPR Tradeoff', fontsize=14)
    plt.xlim(0, 1.05)
    plt.ylim(0.4, 0.95)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("acc_vs_fp.eps", format="eps", dpi=300)
    plt.show()

def create_threshold_metrics_table(top_run_dirs, top_models_df, selected_thresholds=None):
    """
    Create a table with metrics at selected confidence thresholds for normal and noisy queries

    Parameters:
    -----------
    top_run_dirs : list
        List of directories for the top models
    top_models_df : pandas DataFrame
        DataFrame containing information about the top models
    selected_thresholds : list, optional
        List of confidence thresholds to include, defaults to a strategic selection if None

    Returns:
    --------
    pandas DataFrame
        Table with metrics for each model at each threshold
    """
    results = []

    # Process each model
    for i, run_dir in enumerate(top_run_dirs):
        model_name = top_models_df.iloc[i]['model_name']
        model_name_short = model_name.split('/')[-1]

        # Load normal queries confidence threshold data
        normal_path = os.path.join(run_dir, "confidence_threshold_metrics",
                                   "cosine_single_argument_classification_normal_queries.csv")

        # Load noisy queries confidence threshold data
        noisy_path = os.path.join(run_dir, "confidence_threshold_metrics",
                                  "cosine_single_argument_classification_noisy_queries.csv")

        if os.path.exists(normal_path) and os.path.exists(noisy_path):
            normal_df = pd.read_csv(normal_path)
            noisy_df = pd.read_csv(noisy_path)

            # Get all available thresholds
            all_thresholds = sorted(set(normal_df['Confidence'].unique()).intersection(
                set(noisy_df['Confidence'].unique())))

            # If no thresholds provided, select a strategic set
            if selected_thresholds is None:
                # Include the lowest and highest threshold
                selected_thresholds = [min(all_thresholds), max(all_thresholds)]

                # Add thresholds at 0.5, 0.7, and 0.9 if they exist
                for standard_thresh in [0.5, 0.7, 0.9]:
                    if standard_thresh in all_thresholds:
                        selected_thresholds.append(standard_thresh)
                    else:
                        # Find closest available threshold
                        closest = min(all_thresholds, key=lambda x: abs(x - standard_thresh))
                        selected_thresholds.append(closest)

                # De-duplicate and sort
                selected_thresholds = sorted(set(selected_thresholds))

            # Filter to selected thresholds
            normal_filtered = normal_df[normal_df['Confidence'].isin(selected_thresholds)]
            noisy_filtered = noisy_df[noisy_df['Confidence'].isin(selected_thresholds)]

            # Create metrics for each threshold
            for threshold in selected_thresholds:
                normal_row = normal_filtered[normal_filtered['Confidence'] == threshold].iloc[0]
                noisy_row = noisy_filtered[noisy_filtered['Confidence'] == threshold].iloc[0]

                # Calculate metrics
                normal_accuracy = normal_row['Accuracy']
                noisy_accuracy = noisy_row['Accuracy']
                fp_rate = 1.0 - noisy_accuracy

                # Calculate F1 score based on normal accuracy and false positive rate
                # Treating normal accuracy as recall and (1-fp_rate) as precision
                precision = 1.0 - fp_rate
                recall = normal_accuracy
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                # Number of queries above threshold
                normal_queries = normal_row['Queries_Above_Threshold']
                noisy_queries = noisy_row['Queries_Above_Threshold']

                results.append({
                    'model': model_name_short,
                    'threshold': threshold,
                    'normal_accuracy': normal_accuracy,
                    'normal_queries': normal_queries,
                    'false_positive_rate': fp_rate,
                    'noisy_queries': noisy_queries,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Pivot to have models as rows and metrics as columns, grouped by threshold
    pivot_df = results_df.pivot_table(
        index=['model', 'threshold'],
        values=['normal_accuracy', 'false_positive_rate', 'precision', 'recall', 'f1_score',
                'normal_queries', 'noisy_queries'],
        aggfunc='first'
    ).reset_index()

    # Format numeric columns
    for col in ['normal_accuracy', 'false_positive_rate', 'precision', 'recall', 'f1_score']:
        pivot_df[col] = pivot_df[col].map(lambda x: f"{x:.4f}")

    # Sort by model and threshold
    pivot_df = pivot_df.sort_values(['model', 'threshold'])

    return pivot_df


def find_optimal_thresholds(top_run_dirs, top_models_df):
    """
    Find optimal thresholds for each model based on different criteria

    Parameters:
    -----------
    top_run_dirs : list
        List of directories for the top models
    top_models_df : pandas DataFrame
        DataFrame containing information about the top models

    Returns:
    --------
    pandas DataFrame
        Table with optimal thresholds for each model based on different criteria
    """
    results = []

    # Process each model
    for i, run_dir in enumerate(top_run_dirs):
        model_name = top_models_df.iloc[i]['model_name']
        model_name_short = model_name.split('/')[-1]

        # Load data
        normal_path = os.path.join(run_dir, "confidence_threshold_metrics",
                                   "cosine_single_argument_classification_normal_queries.csv")
        noisy_path = os.path.join(run_dir, "confidence_threshold_metrics",
                                  "cosine_single_argument_classification_noisy_queries.csv")

        if os.path.exists(normal_path) and os.path.exists(noisy_path):
            normal_df = pd.read_csv(normal_path)
            noisy_df = pd.read_csv(noisy_path)

            # Merge datasets on Confidence threshold
            merged_df = pd.merge(normal_df, noisy_df, on='Confidence',
                                 suffixes=('_normal', '_noisy'))

            # Calculate additional metrics
            merged_df['false_positive_rate'] = 1.0 - merged_df['Accuracy_noisy']
            merged_df['precision'] = 1.0 - merged_df['false_positive_rate']
            merged_df['recall'] = merged_df['Accuracy_normal']

            # Calculate F1 score
            merged_df['f1_score'] = 2 * (merged_df['precision'] * merged_df['recall']) / \
                                    (merged_df['precision'] + merged_df['recall'])
            merged_df['f1_score'] = merged_df['f1_score'].fillna(0)

            # Calculate balanced accuracy
            merged_df['balanced_accuracy'] = (merged_df['Accuracy_normal'] +
                                              merged_df['Accuracy_noisy']) / 2

            # Find threshold that maximizes different metrics
            max_f1_row = merged_df.loc[merged_df['f1_score'].idxmax()]
            max_balanced_acc_row = merged_df.loc[merged_df['balanced_accuracy'].idxmax()]

            # Find threshold with FP rate <= 0.05 that maximizes normal accuracy
            low_fp_df = merged_df[merged_df['false_positive_rate'] <= 0.05]
            max_acc_low_fp_row = low_fp_df.loc[low_fp_df['Accuracy_normal'].idxmax()] \
                if not low_fp_df.empty else None

            # Add results
            model_results = {
                'model': model_name_short,
                'max_f1_threshold': max_f1_row['Confidence'],
                'max_f1_value': max_f1_row['f1_score'],
                'max_f1_normal_acc': max_f1_row['Accuracy_normal'],
                'max_f1_fp_rate': max_f1_row['false_positive_rate'],

                'max_balanced_acc_threshold': max_balanced_acc_row['Confidence'],
                'max_balanced_acc_value': max_balanced_acc_row['balanced_accuracy'],
                'max_balanced_acc_normal_acc': max_balanced_acc_row['Accuracy_normal'],
                'max_balanced_acc_fp_rate': max_balanced_acc_row['false_positive_rate'],
            }

            # Add low false positive rate results if available
            if max_acc_low_fp_row is not None:
                model_results.update({
                    'max_acc_low_fp_threshold': max_acc_low_fp_row['Confidence'],
                    'max_acc_low_fp_value': max_acc_low_fp_row['Accuracy_normal'],
                    'max_acc_low_fp_fp_rate': max_acc_low_fp_row['false_positive_rate']
                })
            else:
                model_results.update({
                    'max_acc_low_fp_threshold': None,
                    'max_acc_low_fp_value': None,
                    'max_acc_low_fp_fp_rate': None
                })

            results.append(model_results)

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Format numeric columns
    for col in results_df.columns:
        if 'threshold' not in col and col != 'model' and results_df[col].dtype != object:
            results_df[col] = results_df[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

    return results_df


def summary_table(top_run_dirs: Sequence[str],
                  model_name_col: pd.Series,
                  fp_targets: Sequence[float] = (0.10, 0.20)
                 ) -> pd.DataFrame:
    """
    One-row-per-model summary:
      ┌ model ┬ acc@10% ┬ thr@10% ┬ acc@20% ┬ thr@20% ┬ max_acc ┬ fp@max ┬ thr@max ┐
    """
    # ---------------- handle duplicate names -----------------
    counter = defaultdict(int)
    disambig_names = []
    for name in model_name_col:
        base = name.split('/')[-1]          # strip path
        counter[base] += 1
        disambig_names.append(f"{base} {counter[base]}")

    rows: List[Dict] = []

    for run_dir, model in zip(top_run_dirs, disambig_names):
        # ---- load csvs ----
        nrm = pd.read_csv(os.path.join(
            run_dir, "confidence_threshold_metrics",
            "cosine_single_argument_classification_normal_queries.csv"))
        nzy = pd.read_csv(os.path.join(
            run_dir, "confidence_threshold_metrics",
            "cosine_single_argument_classification_noisy_queries.csv"))

        merged = pd.merge(nrm, nzy, on="Confidence",
                          suffixes=("_normal", "_noisy"))
        merged["fp_rate"] = 1.0 - merged["Accuracy_noisy"]
        merged.sort_values("fp_rate", inplace=True)   # ascending fp

        # row skeleton
        rec = {"model": model}

        # -------- specific FPR targets -----------
        for fp_t in fp_targets:
            idx = (merged["fp_rate"] - fp_t).abs().idxmin()
            sel = merged.loc[idx]
            col_prefix = f"{int(fp_t*100)}"  # "10", "20", ...
            rec[f"acc@{col_prefix}%"] = sel["Accuracy_normal"]
            rec[f"thr@{col_prefix}%"] = sel["Confidence"]

        # -------- earliest max-accuracy ----------
        max_acc = merged["Accuracy_normal"].max()
        first_idx = merged[merged["Accuracy_normal"] == max_acc].index[0]
        sel = merged.loc[first_idx]
        rec.update({
            "max_acc" : max_acc,
            "fp@max"  : sel["fp_rate"],
            "thr@max" : sel["Confidence"]
        })

        rows.append(rec)

    # nice column order
    fp_cols = []
    for fp_t in fp_targets:
        p = f"{int(fp_t*100)}%"
        fp_cols += [f"acc@{p}", f"thr@{p}"]
    ordered_cols = ["model"] + fp_cols + ["max_acc", "fp@max", "thr@max"]

    return pd.DataFrame(rows)[ordered_cols]

import json
import os
from typing import Dict

import pandas as pd


def get_folder_and_table_information(sweep_dir):
    def print_indented_head(df, indent=4):
        spaces = ' ' * indent
        head_str = df.head().to_string()
        print("\n".join(spaces + line for line in head_str.splitlines()))

    print("fragrant-sweep-16")
    for dir in os.listdir(sweep_dir):
        eval_table_dir = os.path.join(sweep_dir, dir)
        print(f" ├── {dir}")
        if os.path.isdir(eval_table_dir):
            for file in os.listdir(eval_table_dir):
                print(f"  ├── {file}")
                df = pd.read_csv(os.path.join(eval_table_dir, file))
                print_indented_head(df.head(1))
                print()


def get_run_config(run_dir):
    """Extract configuration details from the run directory"""
    # This is a placeholder - you'll need to adapt it to how your configs are stored
    # Potentially from wandb API, config files, or directory names
    with open(f"{run_dir}/run_config.json") as run_config_json:
        config = json.load(run_config_json)
    return config


def calculate_recall_at_k(error_analysis_df, k=10):
    """
    Calculate recall@k from the error analysis table.

    Recall@k is defined as the number of relevant labels in the top k predictions
    divided by the total number of labels for the query.

    Parameters:
    -----------
    error_analysis_df : pandas DataFrame
        The error analysis dataframe containing relevant columns
    k : int, default=10
        The number of top predictions to consider

    Returns:
    --------
    float
        The average recall@k across all queries
    """
    # Initialize a list to store recall values for each query
    recall_values = []

    for _, row in error_analysis_df.iterrows():
        # Get the ground truth labels for this query
        anchor_labels = row['anchor_labels']
        if isinstance(anchor_labels, str):
            # If it's stored as a string representation of a list, convert it
            anchor_labels = eval(anchor_labels)

        # Number of ground truth labels
        num_labels = len(anchor_labels)

        # Parse the top10 predictions string to extract labels
        top_predictions = row['top10']
        if isinstance(top_predictions, str):
            # If it's stored as a string representation of a list, convert it
            top_predictions = eval(top_predictions)

        # Extract just the labels from the predictions (format varies)
        pred_labels = []
        for pred in top_predictions[:k]:  # Consider only top-k predictions
            if '//' in pred:
                # Format appears to be 'rank//label//text//score'
                pred_labels.append(pred.split('//')[1])
            else:
                # If different format, you might need to adjust this parsing
                pred_labels.append(pred)

        # Count how many ground truth labels are in the top-k predictions
        relevant_in_topk = sum(1 for label in anchor_labels if label in pred_labels)

        # Calculate recall for this query
        if num_labels > 0:  # Avoid division by zero
            recall = relevant_in_topk / num_labels
        else:
            recall = float('nan')  # or some other appropriate value

        recall_values.append(recall)

    # Calculate average recall across all queries
    avg_recall = sum(r for r in recall_values if not pd.isna(r)) / sum(1 for r in recall_values if not pd.isna(r))

    return avg_recall


def get_overall_sweep_results(run_dirs, model_names: Dict[str, str] = None, include_recall=True):
    # Empty list to store results
    results = []
    # Process each run directory
    for run_dir in run_dirs:
        try:
            # Load run configuration
            with open(os.path.join(run_dir, "run_config.json"), "r") as f:
                config = json.load(f)

            # Get model name
            if model_names:
                model_name=model_names[config.get("model_name")]
            else:
                model_name = config.get("model_name")

            # Load overall accuracy metrics
            acc_path = os.path.join(run_dir, "overall_metrics", "cosine_overall_accuracy.csv")
            if os.path.exists(acc_path):
                acc_df = pd.read_csv(acc_path)
                # Extract accuracy values for different K
                accuracy_dict = {f"Acc@{int(row['Metric'])}": row['Value']
                                 for _, row in acc_df.iterrows()}
            else:
                accuracy_dict = {}

            # Load overall precision metrics
            prec_path = os.path.join(run_dir, "overall_metrics", "cosine_overall_precision.csv")
            if os.path.exists(prec_path):
                prec_df = pd.read_csv(prec_path)
                # Extract precision values for different K
                precision_dict = {f"Prec@{int(row['Metric'])}": row['Value']
                                  for _, row in prec_df.iterrows()}
            else:
                precision_dict = {}

            # Calculate recall metrics if requested
            recall_dict = {}
            if include_recall:
                error_analysis_path = os.path.join(run_dir, "error_analysis", "cosine_error_analysis.csv")
                if os.path.exists(error_analysis_path):
                    error_df = pd.read_csv(error_analysis_path)
                    for k in [1, 3, 5, 7]:
                        if k <= 10:  # We only have top 10 predictions
                            recall_dict[f'Rec@{k}'] = calculate_recall_at_k(error_df, k)

            # Get additional configuration parameters
            add_discussion_info = config.get("add_discussion_scenario_info", False)
            context_length = config.get("context_length", "NaN")
            learning_rate = config.get("learning_rate", "NaN")
            batch_size = config.get("batch_size", "NaN")
            num_epochs = config.get("num_epochs", "NaN")
            warmup_ratio = config.get("warmup_ratio", "NaN")

            # Combine all information
            run_info = {
                "model_name": model_name,
                "add_discussion_info": add_discussion_info,
                "context_length": context_length,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "warmup_ratio": warmup_ratio,
                "run_dir": os.path.basename(run_dir),
                **accuracy_dict,
                **precision_dict,
                **recall_dict
            }

            results.append(run_info)
        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    return results_df
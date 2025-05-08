import os
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Dict

from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection
from ethikchat_argtoolkit.ArgumentGraph.stance import Stance

from src.data.classes import Passage, Query
from src.data.dataset_splitting.utils import load_splits_from_disk
from src.models.train_model_sweep import load_argument_graphs


def convert_hf_dataset_to_objects(dataset):
    """Convert HuggingFace dataset to Query and Passage objects"""
    # Convert queries
    queries = Query.get_queries_from_hf_dataset(dataset["queries"])
    passages = Passage.get_passages_from_hf_dataset(dataset["passages"])


    # Get the query-passage mappings
    query_passage_mapping = {}
    for item in dataset["queries_relevant_passages_mapping"]:
        query_passage_mapping[item["query_id"]] = item["passages_ids"]

    return queries, passages, query_passage_mapping

# === Filtering functions ===

def scenario_filter(query: Query, passages: List[Passage]) -> List[Passage]:
    """Filter passages to only include those from the same discussion scenario as the query."""
    return [p for p in passages if p.discussion_scenario == query.discussion_scenario]


def stance_filter(query: Query, passages: List[Passage],
                  response_template_collections: Dict[str, ResponseTemplateCollection]) -> List[Passage]:
    """Filter passages to only include those matching the user's stance."""
    # Convert user_stance int to Stance enum
    user_stance = Stance(query.user_stance)

    filtered_passages = []
    for passage in passages:
        template = response_template_collections[passage.discussion_scenario].get_template_for_label(passage.label)
        if template and template.stance == user_stance:
            filtered_passages.append(passage)

    return filtered_passages


def history_filter(query: Query, passages: List[Passage]) -> List[Passage]:
    """
    Filter out arguments that have already been uttered in the dialogue history.
    """
    # Flatten all labels from context_labels history
    history_labels = []
    for labels_turn in query.context_labels:
        history_labels.extend(labels_turn)

    # Remove passages with labels already in the dialogue history
    return [p for p in passages if p.label not in history_labels]


def counter_argument_filter(query: Query, passages: List[Passage],
                            response_template_collections: Dict[str, ResponseTemplateCollection]) -> List[Passage]:
    """
    Filter based on relationship to last bot utterance:
    1. Keep arguments that counter bot's last arguments (or counter-counters, etc.)
    2. Keep arguments that counter similar arguments to bot's last arguments
    3. Keep first-level arguments that can be uttered independently
    """
    # If no context or empty context_labels, keep all passages

    # Get the labels from the bot's last utterance
    bot_last_labels = query.context_labels[-1] if query.context_labels else []

    if not bot_last_labels:
        return passages  # No previous bot labels to filter by

    rtc = response_template_collections[query.discussion_scenario]

    # Find all valid counter argument labels (including counter-counters, etc.)
    valid_labels = set()

    # Get first level argument labels
    first_level_args = set(label for label in rtc.user_arguments_labels
                           if not rtc.get_template_for_label(label).has_parent_labels)

    # For each bot's last argument
    for label in bot_last_labels:
        template = rtc.get_template_for_label(label)
        if not template:
            continue

        # Add all counter arguments (traversing down the argument tree)
        counter_queue = list(template.child_templates)
        while counter_queue:
            counter = counter_queue.pop(0)
            valid_labels.add(counter.label)
            counter_queue.extend(counter.child_templates)

        # Add counters to similar arguments
        for similar in template.similar_templates:
            for counter in similar.child_templates:
                valid_labels.add(counter.label)

    # Combine with first-level arguments
    valid_labels.update(first_level_args)

    # Return passages with labels in our valid set
    filtered_passages = [p for p in passages if p.label in valid_labels]

    # If filtering is too restrictive, return original passages
    if len(filtered_passages) < 5:
        return passages

    return filtered_passages


def group_filter(query: Query, passages: List[Passage],
                 response_template_collections: Dict[str, ResponseTemplateCollection]) -> List[Passage]:
    """
    Filter to include only arguments that are in the same groups as the bot's last arguments.
    """
    # If no context or empty context_labels, keep all passages

    # Get the labels from the bot's last utterance
    bot_last_labels = query.context_labels[-1] if query.context_labels else []

    if not bot_last_labels:
        return passages  # No previous bot labels to filter by

    rtc = response_template_collections[query.discussion_scenario]

    # Get all groups from bot's last arguments
    bot_groups = set()
    for label in bot_last_labels:
        template = rtc.get_template_for_label(label)
        if template and template.groups:
            bot_groups.update(template.groups)

    if not bot_groups:
        return passages  # No groups found

    filtered_passages = []
    for passage in passages:
        template = rtc.get_template_for_label(passage.label)
        if template and any(group in template.groups for group in bot_groups):
            filtered_passages.append(passage)

    # If filtering is too restrictive, return original passages
    if len(filtered_passages) < 5:
        return passages

    return filtered_passages


def combined_filter(query: Query, passages: List[Passage],
                    response_template_collections: Dict[str, ResponseTemplateCollection],
                    use_scenario=True, use_stance=True, use_group=False,
                    use_counter=False, use_history=False) -> List[Passage]:
    """Combine multiple filtering approaches."""
    filtered = passages

    if use_scenario:
        filtered = scenario_filter(query, filtered)

    if use_stance:
        stance_filtered = stance_filter(query, filtered, response_template_collections)
        # Only use stance filtering if it doesn't eliminate too many options
        if len(stance_filtered) >= 5:
            filtered = stance_filtered

    if use_history:
        history_filtered = history_filter(query, filtered)
        # Only use history filtering if it doesn't eliminate too many options
        if len(history_filtered) >= 5:
            filtered = history_filtered

    if use_group:
        group_filtered = group_filter(query, filtered, response_template_collections)
        # Only use group filtering if it doesn't eliminate too many options
        if len(group_filtered) >= 5:
            filtered = group_filtered

    if use_counter:
        counter_filtered = counter_argument_filter(query, filtered, response_template_collections)
        # Only use counter argument filtering if it doesn't eliminate too many options
        if len(counter_filtered) >= 5:
            filtered = counter_filtered

    return filtered


def evaluate_filter(filter_function, queries, passages, query_passage_mapping,
                    response_template_collections, **filter_kwargs):
    """
    Evaluate a filtering function to see if it retains gold standard passages.

    Args:
        filter_function: Function that filters passages
        queries: List of Query objects
        passages: List of Passage objects
        query_passage_mapping: Mapping from query IDs to gold passage IDs
        response_template_collections: Collection of argument templates
        filter_kwargs: Additional arguments for the filter function

    Returns:
        Dict with metrics on filter performance
    """
    results = {
        'total_queries': 0,
        'queries_with_retained_gold': 0,
        'avg_retained_passages': 0,
        'avg_retained_gold_passages': 0,
        'avg_reduction_percentage': 0,
        'precision_improvement': 0
    }

    total_retained = 0
    total_gold_retained = 0
    total_reduction = 0
    total_precision_improvement = 0

    for query in tqdm(queries, desc="Evaluating filter"):
        if query.id not in query_passage_mapping:
            continue

        gold_passage_ids = query_passage_mapping[query.id]
        if not gold_passage_ids:
            continue

        # Apply filter
        filtered_passages = filter_function(query, passages, response_template_collections, **filter_kwargs)

        # Check how many gold passages are retained
        filtered_passage_ids = [p.id for p in filtered_passages]
        retained_gold = [pid for pid in gold_passage_ids if pid in filtered_passage_ids]

        # Calculate precision improvement
        original_precision = len(gold_passage_ids) / len(passages) if passages else 0
        filtered_precision = len(retained_gold) / len(filtered_passages) if filtered_passages else 0
        precision_improvement = filtered_precision - original_precision

        # Update metrics
        results['total_queries'] += 1
        if retained_gold:
            results['queries_with_retained_gold'] += 1

        total_retained += len(filtered_passages)
        total_gold_retained += len(retained_gold)
        total_reduction += (1 - (len(filtered_passages) / len(passages))) * 100 if passages else 0
        total_precision_improvement += precision_improvement

    # Calculate averages
    if results['total_queries'] > 0:
        results['avg_retained_passages'] = total_retained / results['total_queries']
        results['avg_retained_gold_passages'] = total_gold_retained / results['total_queries']
        results['avg_reduction_percentage'] = total_reduction / results['total_queries']
        results['recall'] = results['queries_with_retained_gold'] / results['total_queries']
        results['precision_improvement'] = total_precision_improvement / results['total_queries']

    return results


def main():
    # Set paths directly instead of using command line arguments
    project_root = "/home/christian/PycharmProjects/ethikchat-experiment-argument-classification"
    dataset_path = os.path.join(project_root, "data/processed/with_context/dataset_split_in_distribution_from_v3")  # Adjust this path
    print(f"Loading dataset from {dataset_path}")
    dataset = load_splits_from_disk(dataset_path)
    print("Dataset loaded")
    arg_graphs = load_argument_graphs(project_root, is_test_run=True)
    output_dir = "filter_evaluation_results"  # Adjust this path if needed

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert dataset to objects
    queries, passages, query_passage_mapping = convert_hf_dataset_to_objects(dataset['train'])
    print(f"Converted dataset to {len(queries)} queries and {len(passages)} passages")

    # Define filters to evaluate
    # Define filters to evaluate
    filters = {
        "no_filter": lambda q, p, rtc, **kwargs: p,  # Baseline: no filtering
        "scenario_filter": lambda q, p, rtc, **kwargs: scenario_filter(q, p),
        "stance_filter": lambda q, p, rtc, **kwargs: stance_filter(q, p, rtc),
        "history_filter": lambda q, p, rtc, **kwargs: history_filter(q, p),
        "group_filter": lambda q, p, rtc, **kwargs: group_filter(q, p, rtc),
        "counter_argument_filter": lambda q, p, rtc, **kwargs: counter_argument_filter(q, p, rtc),
        "combined_scenario_stance": lambda q, p, rtc, **kwargs: combined_filter(
            q, p, rtc, use_scenario=True, use_stance=True, use_group=False, use_counter=False, use_history=False),
        "combined_scenario_stance_history": lambda q, p, rtc, **kwargs: combined_filter(
            q, p, rtc, use_scenario=True, use_stance=True, use_group=False, use_counter=False, use_history=True),
        "combined_scenario_counter": lambda q, p, rtc, **kwargs: combined_filter(
            q, p, rtc, use_scenario=True, use_stance=False, use_group=False, use_counter=True, use_history=False),
        "combined_scenario_group": lambda q, p, rtc, **kwargs: combined_filter(
            q, p, rtc, use_scenario=True, use_stance=False, use_group=True, use_counter=False, use_history=False),
        "combined_all": lambda q, p, rtc, **kwargs: combined_filter(
            q, p, rtc, use_scenario=True, use_stance=True, use_group=True, use_counter=True, use_history=True),
    }

    # Evaluate each filter
    results = {}
    for filter_name, filter_func in filters.items():
        print(f"Evaluating {filter_name}...")
        results[filter_name] = evaluate_filter(
            filter_func, queries, passages, query_passage_mapping, arg_graphs)
        print(f"Results for {filter_name}:")
        for metric, value in results[filter_name].items():
            print(f"  {metric}: {value}")

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, "filter_evaluation_results.csv"))

    # Create visualizations
    plt.figure(figsize=(12, 8))

    # Plot recall vs reduction as a scatter plot
    plt.scatter(
        results_df['avg_reduction_percentage'],
        results_df['recall'],
        s=100,
        alpha=0.7
    )

    # Add labels for each point
    for idx, row in results_df.iterrows():
        plt.annotate(
            idx,
            (row['avg_reduction_percentage'], row['recall']),
            xytext=(5, 5),
            textcoords='offset points'
        )

    plt.xlabel('Average Search Space Reduction (%)')
    plt.ylabel('Recall (% of queries with retained gold passages)')
    plt.title('Filter Performance: Recall vs Search Space Reduction')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "filter_performance.png"))

    # Create separate bar charts for key metrics
    metrics = ['recall', 'avg_reduction_percentage', 'precision_improvement']

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))

    for i, metric in enumerate(metrics):
        axes[i].bar(results_df.index, results_df[metric])
        axes[i].set_title(f'{metric}')
        axes[i].set_xticklabels(results_df.index, rotation=45, ha='right')
        axes[i].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "filter_metrics.png"))

    print(f"Results saved to {output_dir}")

    return results_df  # Return results DataFrame for further analysis in IDE


if __name__ == "__main__":
    main()

    # TODO: Dialogue history filter ausprobieren
    #   Counter Argument Filter so verändern, dass main argumente auch möglich sind, aber vom richtigen stance
    #   untersuchen, warum stance filter nicht 100% recall hat
    #   ACHTUNG: Aktuell beziehen sich die filter nicht auf das vorangehende Argument, sondern auf das Argument selbst.
    #   Auf das Argument selbst kann sich aber nur der stance und der scenario filter beziehen.
    #   Group, counterargument filter müssen sich auf die vorhergehende query in der history beziehen.
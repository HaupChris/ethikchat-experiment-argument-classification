import random
from collections import defaultdict
from typing import List, Dict

from datasets import DatasetDict
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario
from matplotlib import pyplot as plt


from src.data.create_corpus_dataset import Query


def approximate_n_cover(queries: List[Query], n: int) -> List[Query]:
    """
    Given a list of Query objects and an integer n, return a subset of queries
    that approximates an n-cover. For each label:

      - If there are fewer than n queries containing that label,
        include all of them.
      - If there are at least n queries containing that label,
        include at least n of those queries (this naive approach will simply
        pick any n of them).

    This ensures that no label is under-represented (fewer than n) if
    enough queries are available, but it may over-represent some labels
    if the same query carries multiple labels.
    """

    # Map each label -> all queries that contain it
    label_to_queries = defaultdict(list)
    for query in queries:
        for label in query.labels:
            label_to_queries[label].append(query)

    # Build the result set (using a set to avoid duplicates)
    selected_queries = set()

    for label, qlist in label_to_queries.items():
        if len(qlist) <= n:
            # If fewer than n queries exist for this label, pick them all
            selected_queries.update(qlist)
        else:
            # Otherwise, pick exactly n queries for this label
            # (Could be random or sorted by id; here we simply take the first n)
            selected_queries.update(qlist[:n])

    # Convert the set back to a list for the final result
    return list(selected_queries)



def analyze_cover(
    all_queries: List[Query],
    cover_queries: List[Query],
    n: int
) -> bool:
    """
    Plots a histogram showing how many queries in `cover_queries` contain each label
    (considering all labels that appear in `all_queries`). Also returns a boolean
    indicating whether `cover_queries` is an exact n-cover for all labels.
    """
    # Collect all unique labels
    unique_labels = set()
    for q in all_queries:
        unique_labels.update(q.labels)
    all_labels = sorted(unique_labels)

    # Count coverage for each label in the chosen cover
    label_counts = {lbl: 0 for lbl in all_labels}
    for q in cover_queries:
        for lbl in q.labels:
            label_counts[lbl] += 1

    # Check if we have an exact n-cover
    is_exact_cover = all(count == n for count in label_counts.values())

    # Plot the histogram (bar chart) of label coverage in the chosen subset
    labels_list = list(label_counts.keys())
    counts_list = list(label_counts.values())

    plt.bar(labels_list, counts_list)
    plt.xlabel("Labels")
    plt.ylabel("Number of Queries in Cover")
    plt.title("Label Coverage in Cover Subset")
    plt.xticks(rotation=90)  # rotate label names if needed
    plt.tight_layout()
    plt.show()

    return is_exact_cover


def count_label_freq(queries: List[Query]) -> Dict[str, int]:
    """
    Return a dictionary mapping each label to the number of queries containing it.
    """
    freq = defaultdict(int)
    for query in queries:
        for label in query.labels:
            freq[label] += 1
    return freq

def visualize_n_cover_distribution(full_query_set: List[Query],
                                   n_cover_queries: List[List[Query]],
                                   ns: List[int],
                                   accuracies: List[float]) -> None:
    """
    Plot a single graph showing:
    - The label frequency distribution in the full query set (descending).
    - The label frequency distribution for each approximate n-cover.
    - The accuracy score for each cover using a secondary y-axis.

    :param full_query_set: All Query objects.
    :param n_cover_queries: A list of coverage subsets, each from approximate_n_cover for some n.
    :param ns: The list of n-values in the same order as n_cover_queries.
    :param accuracies: Accuracy scores corresponding to each n-cover.
    """
    # 1. Count frequencies in the full set
    label_freq_full = count_label_freq(full_query_set)

    # 2. Sort labels by their frequency in descending order
    sorted_labels = sorted(label_freq_full.keys(),
                           key=lambda lbl: label_freq_full[lbl],
                           reverse=True)

    # 3. Create figure with larger size for better readability
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 4. Plot frequency for the full set
    y_full = [label_freq_full[lbl] for lbl in sorted_labels]
    ax1.plot(y_full, label="Full Set")

    # 5. For each n-cover, plot frequencies and collect lines for legend reuse
    lines = []
    labels = []
    for coverage, n in zip(n_cover_queries, ns):
        label_freq_cov = count_label_freq(coverage)
        y_cov = [label_freq_cov.get(lbl, 0) for lbl in sorted_labels]
        line, = ax1.plot(y_cov, label=f"n={n}-cover")
        lines.append(line)
        labels.append(f"n={n}-cover")

    # 6. Labeling for left y-axis
    ax1.set_xlabel("Label")
    ax1.set_ylabel("Number of Queries Containing Label")
    ax1.set_title("Label Frequency Distribution and Accuracy Scores")
    ax1.set_xticks(ticks=range(len(sorted_labels)))
    ax1.set_xticklabels(labels=sorted_labels, rotation=90)

    # 7. Add secondary y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)

    # 8. Plot accuracy values using the same color as the corresponding cover
    for line, acc, n in zip(lines, accuracies, ns):
        ax2.plot([], [], color=line.get_color(), label=f"Accuracy n={n}: {acc:.2f}")

    # 9. Combined legend
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    # 10. Show the plot
    plt.tight_layout()
    plt.show()







if __name__ == "__main__":
    from datasets import load_from_disk, DatasetDict
    from src.data.dataset_splits import create_splits_from_corpus_dataset, DatasetSplitType

    dataset_folder = "../../data/processed/with_context"
    corpus_ds = load_from_disk(f"{dataset_folder}/corpus_dataset_v2")
    in_distribution_split = create_splits_from_corpus_dataset(corpus_dataset=corpus_ds,
                                                              dataset_split_type=DatasetSplitType.InDistribution,
                                                              save_folder=dataset_folder,
                                                              dataset_save_name="dataset_split_in_distribution")
    ids_train = in_distribution_split["train"]
    filter_queries_for_few_shot_setting(ids_train,   5)

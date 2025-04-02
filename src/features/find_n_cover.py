import random
from collections import defaultdict
from typing import List, Dict

from datasets import DatasetDict
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario
from matplotlib import pyplot as plt
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    LpBinary,
    lpSum,
    LpStatus,
    PULP_CBC_CMD, LpMaximize, LpInteger,
)

from src.data.create_corpus_dataset import Query


def _build_exact_cover_model(queries: List[Query], n: int, labels: List[str]):
    """
    Build a PuLP model for an 'exact cover' approach:
      - For each label, exactly n of the chosen queries must contain that label.
    Returns the model and the list of binary decision variables.
    """
    model = LpProblem("ExactCover", LpMinimize)

    decision_vars = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(len(queries))]
    model += lpSum(decision_vars), "MinimizeNumberOfChosenQueries"

    for lbl in labels:
        model += lpSum(decision_vars[i] for i, q in enumerate(queries) if lbl in q.labels) == n

    return model, decision_vars


def _build_balanced_fallback_model(queries: List[Query], n: int, labels: List[str], alpha: float = 1.0):
    """
    Fallback model to:
      1) Maximize the number of labels with coverage >= n (z_l = 1 if coverage >= n).
      2) Among solutions with the same # of z_l, maximize total coverage (sum of cov_l).
      3) Subtract alpha * sum(x_i) to avoid selecting more queries than needed.

    You can tune 'alpha' to control how strongly you penalize selecting more queries.
    """
    model = LpProblem("BalancedFallback", LpMaximize)

    # Binary x_i: 1 if we pick query i
    x_vars = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(len(queries))]

    # cov_l: integer coverage for label l
    cov_vars = {lbl: LpVariable(f"cov_{lbl}", lowBound=0, cat=LpInteger) for lbl in labels}

    # z_l: 1 if coverage >= n, else 0
    z_vars = {lbl: LpVariable(f"z_{lbl}", cat=LpBinary) for lbl in labels}

    # coverage definition: cov_l = sum of x_i for queries containing lbl
    for lbl in labels:
        coverage_expr = lpSum(x_vars[i] for i, q in enumerate(queries) if lbl in q.labels)
        model += cov_vars[lbl] - coverage_expr == 0

    # if z_l = 1, coverage >= n
    #   cov_l >= n * z_l
    for lbl in labels:
        model += cov_vars[lbl] >= n * z_vars[lbl]

    # Weighted objective:
    #   M * sum(z_l) + sum(cov_l) - alpha * sum(x_i)
    M = len(queries) * 10  # "big" to strongly favor more labels reaching coverage >= n
    model += (
        M * lpSum(z_vars[lbl] for lbl in labels)
        + lpSum(cov_vars[lbl] for lbl in labels)
        - alpha * lpSum(x_vars)
    )

    return model, x_vars, cov_vars, z_vars


def _extract_chosen_queries(queries: List[Query], decision_vars: List[LpVariable], threshold: float = 0.5) -> List[Query]:
    chosen_indices = [
        i for i, var in enumerate(decision_vars)
        if var.varValue is not None and var.varValue > threshold
    ]
    return [queries[i] for i in chosen_indices]


def find_n_cover(queries: List[Query], n: int) -> List[Query]:
    """
    1) Attempt exact cover (each label exactly n).
    2) If infeasible, fallback to a model that:
       - Maximizes # labels with coverage >= n
       - Then total coverage
       - Minimizes # of queries selected (controlled by alpha).
    """
    unique_labels = sorted(set(lbl for q in queries for lbl in q.labels))

    # 1) Exact cover attempt
    exact_model, exact_decision_vars = _build_exact_cover_model(queries, n, unique_labels)
    exact_status = exact_model.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[exact_status] == "Optimal":
        return _extract_chosen_queries(queries, exact_decision_vars)

    # 2) Fallback
    #    - Adjust alpha to tune how strongly you penalize picking extra queries.
    #      If alpha is too small, it may still pick almost everything.
    #      If alpha is too large, it may sacrifice coverage for fewer queries.
    fallback_model, fallback_x, _, _ = _build_balanced_fallback_model(
        queries, n, unique_labels, alpha=1.0
    )
    fallback_status = fallback_model.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[fallback_status] in ("Optimal", "Feasible"):
        return _extract_chosen_queries(queries, fallback_x)

    return []


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




def filter_queries_for_few_shot_setting(split_dataset: DatasetDict, num_shots: int) -> DatasetDict:
    """

    Args:
        split_dataset ():
        num_shots ():

    Returns:

    """
    queries = [Query(
        id=entry["id"],
        text=entry["text"],
        labels=entry["labels"],
        discussion_scenario=entry["discussion_scenario"],
        context=entry["context"],
        scenario_description=entry["scenario_description"],
        scenario_question=entry["scenario_question"]
    ) for entry in split_dataset["queries"]]

    queries = list(filter(lambda query: query.discussion_scenario == DiscussionSzenario.MEDAI, queries))

    shots = list(range(num_shots))
    shots = [shot*3 for shot in shots]
    accuracies = [random.random() for shot in shots]
    n_covers = []
    for i in shots:
        n_covers.append(approximate_n_cover(queries, i))

    visualize_n_cover_distribution(queries, n_covers, shots, accuracies)


    num_shot_queries = approximate_n_cover(queries, num_shots)
    # analyze_cover(queries, num_shot_queries, num_shots)

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
    filter_queries_for_few_shot_setting(ids_train,   200)

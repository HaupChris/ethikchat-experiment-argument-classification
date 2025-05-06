"""dialogue_level_splitting.py  (v1.3, 2025‑05‑06)
-------------------------------------------------
*Scenario‑aware* in‑distribution splitter.

Guarantee (formal)
~~~~~~~~~~~~~~~~~~
For every tuple **(scenario, label)** that occurs in *validation* or *test*,
there exists **at least one *different* query** with the *same* (scenario, label)
inside the *training* split.

What changed in v1.3
~~~~~~~~~~~~~~~~~~~~
* All counting & promotion logic now keys on `(scenario, label)` instead of
  label alone.  Fixes the bug where a label in MEDAI could be satisfied by the
  same label in REFAI (Christian’s remaining issue).
* Added assertion that checks the guarantee explicitly.
"""

from __future__ import annotations

import random
from collections import defaultdict, Counter
from typing import Dict, List, Set, Iterable, Tuple

from datasets import DatasetDict
from src.data.classes import Query
from src.data.dataset_splitting.utils import create_datasetdict_for_query_ids


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _group_by_dialogue(queries_ds) -> Dict[str, List[Query]]:
    grouped = defaultdict(list)
    for ex in queries_ds:
        grouped[ex["original_dialogue_id"]].append(Query(**ex))
    return grouped


def _scenelab_counts(dialogue: List[Query]) -> Counter[Tuple[str, str]]:
    c: Counter[Tuple[str, str]] = Counter()
    for q in dialogue:
        scen = q.discussion_scenario
        c.update((scen, lab) for lab in q.labels)
    return c


def _aggregate_counts(dialogues: Dict[str, List[Query]], dlg_ids: Set[str]) -> Counter[Tuple[str, str]]:
    total = Counter()
    for d in dlg_ids:
        total.update(_scenelab_counts(dialogues[d]))
    return total


def _ids_from(dialogues: Dict[str, List[Query]], dlg_ids: Iterable[str]) -> List[str]:
    return [q.id for d in dlg_ids for q in dialogues[d]]


# ---------------------------------------------------------------------------
# Public splitter
# ---------------------------------------------------------------------------

def dialogue_level_in_distribution_split(
    corpus_dataset: DatasetDict,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetDict:
    """Return HF DatasetDict with *scenario‑aware* in‑distribution guarantee."""

    rng = random.Random(seed)

    dialogues = _group_by_dialogue(corpus_dataset["queries"])
    dlg_ids = list(dialogues.keys())
    rng.shuffle(dlg_ids)

    # --- greedy cover per (scenario, label) ------------------------------ #
    key2dlg = defaultdict(list)
    for d in dlg_ids:
        for key in _scenelab_counts(dialogues[d]):
            key2dlg[key].append(d)

    forced_train: Set[str] = set()
    covered: Set[Tuple[str, str]] = set()
    for key, cands in key2dlg.items():
        if key in covered:
            continue
        chosen = min(cands, key=lambda x: len(dialogues[x]))
        forced_train.add(chosen)
        covered.update(_scenelab_counts(dialogues[chosen]).keys())

    # --- size balancing --------------------------------------------------- #
    remaining = [d for d in dlg_ids if d not in forced_train]
    rng.shuffle(remaining)
    n_total = len(dlg_ids)
    n_train_target = int(n_total * train_ratio)
    n_val_target = int(n_total * val_ratio)

    extra = remaining[: max(0, n_train_target - len(forced_train))]
    train_dlg: Set[str] = forced_train.union(extra)

    rest = remaining[len(extra):]
    val_dlg = set(rest[:n_val_target])
    test_dlg = set(rest[n_val_target:])

    # --- promotion loop (scenario‑aware) --------------------------------- #
    while True:
        train_counts = _aggregate_counts(dialogues, train_dlg)
        valtest_counts = _aggregate_counts(dialogues, val_dlg | test_dlg)
        missing = [key for key, cnt in valtest_counts.items() if train_counts[key] == 0]
        if not missing:
            break
        promote: Set[str] = set()
        for key in missing:
            cands = [d for d in val_dlg | test_dlg if key in _scenelab_counts(dialogues[d])]
            promote.add(min(cands, key=lambda x: len(dialogues[x])))
        for d in promote:
            (val_dlg if d in val_dlg else test_dlg).remove(d)
            train_dlg.add(d)

    # --- final guarantee check ------------------------------------------- #
    train_counts = _aggregate_counts(dialogues, train_dlg)
    for split_name, dlg_set in (("validation", val_dlg), ("test", test_dlg)):
        split_counts = _aggregate_counts(dialogues, dlg_set)
        for key, cnt in split_counts.items():
            assert train_counts[key] >= 1, (
                f"(scenario, label) {key} occurs in {split_name} but not in train")

    # --- build HF DatasetDict -------------------------------------------- #
    train_ids = _ids_from(dialogues, train_dlg)
    val_ids   = _ids_from(dialogues, val_dlg)
    test_ids  = _ids_from(dialogues, test_dlg)

    ds_train = create_datasetdict_for_query_ids(corpus_dataset, train_ids)
    ds_val   = create_datasetdict_for_query_ids(corpus_dataset, val_ids)
    ds_test  = create_datasetdict_for_query_ids(corpus_dataset, test_ids)

    # prune duplicated user‑utterance passages
    forbidden = set(val_ids) | set(test_ids)
    ds_train["passages"] = ds_train["passages"].filter(
        lambda ex: ex["passage_source"] != "user_utterance" or ex["retrieved_query_id"] not in forbidden
    )

    return DatasetDict(train=ds_train, validation=ds_val, test=ds_test)

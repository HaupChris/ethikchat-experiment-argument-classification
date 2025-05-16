from __future__ import annotations

"""

A memory‑efficient contrastive loss for Sentence‑Transformers that **ignores
in‑batch items whose label overlaps with the anchor’s label** (so‑called *false
negatives*). It extends the GradCache‑powered
:class:`sentence_transformers.losses.CachedMultipleNegativesRankingLoss`.

### Label requirements

* When ``exclude_same_label_negatives=False`` nothing special is required.
* With ``exclude_same_label_negatives=True`` **each tower** returned by the
  dataloader **must** contain a 1‑D *label field* – any key that ends with
  ``_label`` (e.g. ``label``, ``query_label`` …).  The values can be:

    * **Scalar label** → ``"cat"``
    * **Multi‑label** encoded as a *pipe‑separated* string → ``"cat|animal"``

  These are treated as hashable strings, so no tensor conversion is needed in
  the collator.
"""

from typing import Any, Iterable, List, Sequence, Tuple, Dict, Callable
from functools import partial

import tqdm
import torch
from torch import Tensor

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import _backward_hook


# --------------------------------------------------------------------------- helpers
def first_key_present(sample: dict[str, Any], *candidate_keys: str) -> str | None:
    """Return the first key present in *sample* or *None*."""
    for key in candidate_keys:
        if key in sample:
            return key
    return None


def split_scenario_label(raw: str) -> Tuple[str, str]:
    """Split `SCENARIO_label` → (`SCENARIO`, `label`)."""
    if "_" not in raw:
        return "GLOBAL", raw
    return tuple(raw.split("_", 1))  # type: ignore[return-value]


# =============================================================== main loss class

class MaskedCachedMultipleNegativesRankingLoss(CachedMultipleNegativesRankingLoss):
    """Drop‑in replacement with optional *label‑aware masking*.
    CachedMultipleNegativesRankingLoss uses for each anchor the positive_passages of the other anchors in the same
    batch as in-batch-negatives. Based on their labels, this loss masks out in-batch-negatives witht the same
    labels as the respective anchor, preventing false-in-batch-negatives.

    Parameters
    ----------
    exclude_same_label_negatives
        When *True* items that share at least one label with the anchor are
        masked out before the softmax.
    """

    # --------------------------------------------------------------------- init

    def __init__(
            self,
            model: SentenceTransformer,
            scale: float = 20.0,
            similarity_fct: Callable[[Tensor, Tensor], Tensor] = util.cos_sim,
            mini_batch_size: int = 32,
            show_progress_bar: bool = False,
            exclude_same_label_negatives: bool = False,
    ) -> None:
        super().__init__(
            model,
            scale=scale,
            similarity_fct=similarity_fct,
            mini_batch_size=mini_batch_size,
            show_progress_bar=show_progress_bar,
        )
        self.exclude_same_label_negatives = exclude_same_label_negatives

        # containers for metrics -------------------------------------------------------
        self._latest_batch_metrics: Dict[str, float] | None = None
        self._epoch_overlap_counts: Dict[str, Dict[Tuple[str, str], int]] = {}

    def pop_batch_metrics(self) -> Dict[str, float] | None:
        """Return & reset mini-batch statistics (thread-safe)."""
        metrics, self._latest_batch_metrics = self._latest_batch_metrics, None
        return metrics

    def pop_epoch_heatmaps(self) -> Dict[str, Dict[Tuple[str, str], int]]:
        """Return & reset accumulated overlap counts (one heatmap per scenario).
        Dict is of format Scenario -> (anchor_label, passage_label) -> count

        """


        heatmaps = self._epoch_overlap_counts
        self._epoch_overlap_counts = {}
        return heatmaps

    @staticmethod
    def _gather_label_lists(feature_dicts: Sequence[dict[str, Any]]) -> List[List[str]]:
        """Extract labels as `List[List[str]]` – one inner list per tower."""
        all_label_lists: List[List[str]] = []

        for tower_index, features in enumerate(feature_dicts):
            key = first_key_present(features, "label", "query_label", "positive_label")
            if key is None:  # fallback: any *_label key
                candidate_keys = [k for k in features if k.endswith("_label")]
                if not candidate_keys:
                    raise ValueError(
                        f"Tower {tower_index} has no *_label field while masking is enabled."
                    )
                key = candidate_keys[0]

            raw_values = features[key]
            if torch.is_tensor(raw_values):
                raw_values = raw_values.tolist()
            all_label_lists.append(list(raw_values) if isinstance(raw_values, (list, tuple)) else [raw_values])

        return all_label_lists

    # ---------------------------------------------------------- overlap heat-map utils
    def _update_overlap_heatmap(
            self,
            anchor_labels: List[str],
            candidate_labels: List[str],
            mask_matrix: Tensor,
            positive_column_offset: int,
    ) -> None:
        """Update per-scenario heatmap with newly masked pairs."""
        num_anchor_rows, _ = mask_matrix.shape
        for row in range(num_anchor_rows):
            positive_col = positive_column_offset + row
            for col in mask_matrix[row].nonzero(as_tuple=False).flatten().tolist():
                if col == positive_col:  # skip the true positive
                    continue
                anchor_raw = anchor_labels[row]
                negative_raw = candidate_labels[col]
                anchor_scenario, anchor_label = split_scenario_label(anchor_raw)
                negative_scenario, negative_label = split_scenario_label(negative_raw)
                if anchor_scenario != negative_scenario:
                    continue  # cross-scenario; ignore
                scenario_map = self._epoch_overlap_counts.setdefault(anchor_scenario, {})
                scenario_map[(anchor_label, negative_label)] = scenario_map.get(
                    (anchor_label, negative_label), 0
                ) + 1

    # ---------------------------------------------------------------- masking

    @staticmethod
    def _create_label_mask(
            anchor_labels: list[str],
            passage_labels: list[str],
            positive_column_offset: int,
            *,
            device: torch.device,
    ) -> Tensor:
        """Return a boolean mask with shape ``(len(query_labels), len(passage_labels))``.

        *True* means *ignore*.  The diagonal (true positives) is always kept
        unmasked.
        """
        num_anchors = len(anchor_labels)
        num_candidates = len(passage_labels)

        # ---------------------------------------------------------------- scalar fast path
        all_scalar = all("|" not in lbl for lbl in anchor_labels + passage_labels)
        if all_scalar:
            # Encode strings → ints on the fly for GPU broadcasting
            label_to_int: Dict[str, int] = {}
            encode = lambda s: label_to_int.setdefault(s, len(label_to_int))
            anchor_ids = torch.tensor([encode(l) for l in anchor_labels], device=device)
            passage_ids = torch.tensor([encode(l) for l in passage_labels], device=device)
            mask = anchor_ids.unsqueeze(1).eq(passage_ids)  # get matrix of size (anchors, passages)
            diagonal_rows = torch.arange(num_anchors, device=device)
            diagonal_cols = torch.arange(
                positive_column_offset, positive_column_offset + num_anchors, device=device
            )
            valid_diag = diagonal_cols < num_candidates
            mask[diagonal_rows[valid_diag], diagonal_cols[valid_diag]] = False
            return mask

        # multi-label fallback -------------------------------------------------
        mask = torch.zeros((num_anchors, num_candidates), dtype=torch.bool, device=device)
        anchor_sets = [set(l.split("|")) for l in anchor_labels]
        candidate_sets = [set(l.split("|")) for l in passage_labels]
        for row, anchor_set in enumerate(anchor_sets):
            positive_col = positive_column_offset + row
            for col, cand_set in enumerate(candidate_sets):
                if col == positive_col:
                    continue
                mask[row, col] = bool(anchor_set & cand_set)
        return mask

    # ----------------------------------------------------------- loss helpers

    # --------------------------------------------------- core loss (with statistics)
    def calculate_loss(  # type: ignore[override]
        self,
        batched_embeddings_per_tower: List[List[Tensor]],
        batched_labels_per_tower: List[List[str]] | None = None,
        *,
        with_backward: bool = False,
    ) -> Tensor:
        anchor_embeddings = torch.cat(batched_embeddings_per_tower[0])
        candidate_embeddings = torch.cat(
            [torch.cat(batch) for batch in batched_embeddings_per_tower[1:]]
        )

        batch_size = anchor_embeddings.size(0)
        device = anchor_embeddings.device
        correct_targets = torch.arange(batch_size, device=device, dtype=torch.long)

        # flatten candidate labels once --------------------------------------
        candidate_labels_flat: List[str] | None = None
        if self.exclude_same_label_negatives and batched_labels_per_tower is not None:
            candidate_labels_flat = [
                label for tower_labels in batched_labels_per_tower[1:] for label in tower_labels
            ]

        cumulative_loss: List[Tensor] = []
        total_masked = total_negatives = 0

        for anchor_start in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Prepare caches",
            disable=not self.show_progress_bar,
        ):
            anchor_end = anchor_start + self.mini_batch_size
            similarity = (
                self.similarity_fct(anchor_embeddings[anchor_start:anchor_end], candidate_embeddings)
                * self.scale
            )

            if candidate_labels_flat is not None:
                mask = self._create_label_mask(
                    anchor_labels=batched_labels_per_tower[0][anchor_start:anchor_end],
                    passage_labels=candidate_labels_flat,
                    positive_column_offset=anchor_start,
                    device=device,
                )
                self._update_overlap_heatmap(
                    anchor_labels=batched_labels_per_tower[0][anchor_start:anchor_end],
                    candidate_labels=candidate_labels_flat,
                    mask_matrix=mask,
                    positive_column_offset=anchor_start,
                )
                masked_count = mask.sum().item()
                similarity = similarity.masked_fill(mask, float("-inf"))
            else:
                masked_count = 0

            negatives_in_mb = similarity.numel() - similarity.size(0)
            total_masked += masked_count
            total_negatives += negatives_in_mb

            mb_loss = (
                self.cross_entropy_loss(similarity, correct_targets[anchor_start:anchor_end])
                * similarity.size(0)
                / batch_size
            )
            if with_backward:
                mb_loss.backward()
                mb_loss = mb_loss.detach()
            cumulative_loss.append(mb_loss)

        # store mini-batch metrics -------------------------------------------
        if total_negatives:
            self._latest_batch_metrics = {
                "masked_negatives_per_batch": total_masked,
                "masked_ratio": total_masked / total_negatives,
                "effective_negatives": total_negatives - total_masked,
            }

        return torch.stack(cumulative_loss).sum()


    # --------------------------------------------------------------- forward pass
    def forward(  # noqa: D401
        self,
        sentence_features: Iterable[Dict[str, Any]],
        labels: Tensor | None = None,  # kept for ST trainer compatibility
    ) -> Tensor:
        label_lists: List[List[str]] | None = None
        if self.exclude_same_label_negatives:
            label_lists = self._gather_label_lists(sentence_features)

        # 1) no-grad embed pass + rand-state capture -------------------------
        embeddings_per_tower: List[List[Tensor]] = []
        self.random_states = []
        for feature_dict in sentence_features:
            tower_emb_batches, tower_random_states = [], []
            for emb_mb, rng_state in self.embed_minibatch_iter(
                feature_dict, with_grad=False, copy_random_state=True
            ):
                tower_emb_batches.append(emb_mb.detach().requires_grad_())
                tower_random_states.append(rng_state)
            embeddings_per_tower.append(tower_emb_batches)
            self.random_states.append(tower_random_states)

        # 2) loss & gradient caching ----------------------------------------
        if torch.is_grad_enabled():
            loss_val = self.calculate_loss_and_cache_gradients(
                embeddings_per_tower, label_lists
            )
            loss_val.register_hook(
                partial(_backward_hook, sentence_features=sentence_features, loss_obj=self)
            )
        else:
            loss_val = self.calculate_loss(embeddings_per_tower, label_lists)
        return loss_val

    def calculate_loss_and_cache_gradients(
        self,
        embeddings_per_tower: List[List[Tensor]],
        label_lists: List[List[str]] | None = None,
    ) -> Tensor:
        loss_val = self.calculate_loss(
            embeddings_per_tower,
            label_lists,
            with_backward=True,
        )
        loss_val = loss_val.detach().requires_grad_()
        self.cache = [[mb_grad.grad for mb_grad in tower] for tower in embeddings_per_tower]
        return loss_val


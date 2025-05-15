from __future__ import annotations

import wandb
from wandb.wandb_run import Run
from transformers import TrainerCallback

"""masked_cached_mnrl.py

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

from typing import Any, Iterable, List, Sequence, Tuple, Dict
from functools import partial

import tqdm
import torch
from torch import Tensor

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import _backward_hook


# --------------------------------------------------------------------------- util

def _first_matching_key(d: dict[str, Any], *candidates: str) -> str | None:
    """Return the first key from *candidates* present in *d* or *None*."""
    for key in candidates:
        if key in d:
            return key
    return None


def _split_label(raw: str) -> Tuple[str, str]:
    """Return ``(scenario, label)`` from a raw label string ``SCENARIO_label``."""
    if "_" not in raw:
        return "GLOBAL", raw  # fallback
    scenario, label = raw.split("_", 1)
    return scenario, label


# =============================================================== main loss class

class MaskedCachedMultipleNegativesRankingLoss(CachedMultipleNegativesRankingLoss):
    """Drop‑in replacement with optional *label‑aware masking*.

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
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.cos_sim,
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

        # ---- runtime containers ------------------------------------------------------
        self._batch_metrics: Dict[str, float] | None = None
        # scenario → (anchor_label, neg_label) → count
        self._overlap_counts: Dict[str, Dict[Tuple[str, str], int]] = {}



    # ----------------------------------------------------------- public helpers

    def pop_batch_metrics(self) -> Dict[str, float] | None:
        """Return and *reset* last mini‑batch stats (thread‑safe for trainer)."""
        out, self._batch_metrics = self._batch_metrics, None
        return out

    def pop_overlap_counts(self) -> Dict[str, Dict[Tuple[str, str], int]]:
        """Return and reset accumulated overlap counts for current epoch."""
        out = self._overlap_counts
        self._overlap_counts = {}
        return out

    # ----------------------------------------------------------- label helpers

    @staticmethod
    def _extract_labels(sentence_features: Sequence[dict[str, Any]]) -> list[list[str]]:
        label_lists: list[list[str]] = []
        for idx, sf in enumerate(sentence_features):
            key = _first_matching_key(sf, "label", "query_label", "positive_label")
            if key is None:
                matches = [k for k in sf if k.endswith("_label")]
                if not matches:
                    raise ValueError("No *_label key found in sentence_features[{idx}].")
                key = matches[0]
            values = sf[key]
            if torch.is_tensor(values):
                values = values.tolist()
            label_lists.append(list(values) if isinstance(values, (list, tuple)) else [values])
        return label_lists


    # ------------------------------------------------- mask + stats calculation

    def _update_overlap_counts(
        self,
        q_labels: list[str],
        p_labels: list[str],
        mask: Tensor,
        row_offset: int,
    ) -> None:
        """Accumulate masked pair frequencies for heatmaps."""
        q_size, _ = mask.shape
        for i in range(q_size):
            pos_col = row_offset + i
            # indices where mask is True for this row
            cols = mask[i].nonzero(as_tuple=False).flatten().tolist()
            for j in cols:
                if j == pos_col:
                    continue  # skip positive
                q_lbl_raw = q_labels[i]
                p_lbl_raw = p_labels[j]
                q_scen, q_lbl = _split_label(q_lbl_raw)
                p_scen, p_lbl = _split_label(p_lbl_raw)
                if q_scen != p_scen:  # heatmap per scenario – ignore cross‑scenario pairs
                    continue
                scen_dict = self._overlap_counts.setdefault(q_scen, {})
                scen_dict[(q_lbl, p_lbl)] = scen_dict.get((q_lbl, p_lbl), 0) + 1

    # ---------------------------------------------------------------- masking

    @staticmethod
    def _create_label_mask(
        query_labels: list[str],
        passage_labels: list[str],
        row_offset: int,
        device: torch.device,
    ) -> Tensor:
        """Return a boolean mask with shape ``(len(query_labels), len(passage_labels))``.

        *True* means *ignore*.  The diagonal (true positives) is always kept
        unmasked.
        """
        q_size = len(query_labels)
        p_size = len(passage_labels)

        # ---------------------------------------------------------------- scalar fast path
        all_scalar = all("|" not in lbl for lbl in query_labels + passage_labels)
        if all_scalar:
            # Encode strings → ints on the fly for GPU broadcasting
            id_map: dict[str, int] = {}

            def _enc(label: str) -> int:  # closure capturing id_map
                if label not in id_map:
                    id_map[label] = len(id_map)
                return id_map[label]

            q_ids = torch.tensor([_enc(l) for l in query_labels], device=device)
            p_ids = torch.tensor([_enc(l) for l in passage_labels], device=device)
            mask = q_ids.unsqueeze(1).eq(p_ids)  # (q, p)

            # remove diagonal (offset by row_offset)
            row_idx = torch.arange(q_size, device=device)
            col_idx = torch.arange(row_offset, row_offset + q_size, device=device)
            valid = col_idx < p_size  # guard when >2 towers
            mask[row_idx[valid], col_idx[valid]] = False
            return mask

        # ---------------------------------------------------------------- multi‑label path
        mask = torch.zeros((q_size, p_size), dtype=torch.bool, device=device)

        q_sets = [set(lbl.split("|")) for lbl in query_labels]
        p_sets = [set(lbl.split("|")) for lbl in passage_labels]

        for i in range(q_size):
            pos_col = row_offset + i  # column of the positive
            for j in range(p_size):
                if j == pos_col:
                    continue  # keep the genuine positive
                mask[i, j] = bool(q_sets[i] & p_sets[j])
        return mask

    # ----------------------------------------------------------- loss helpers

    def calculate_loss(
        self,
        reps: list[list[Tensor]],
        labels_info: list[list[str]] | None = None,
        with_backward: bool = False,
    ) -> Tensor:  # type: ignore[override]
        """Exact same semantics as the parent class, plus optional masking."""
        emb_a = torch.cat(reps[0])  # (bsz, dim)
        emb_b = torch.cat([torch.cat(r) for r in reps[1:]])  # ((1+nneg)*bsz, dim)

        bsz = emb_a.size(0)
        device = emb_a.device
        target = torch.arange(bsz, device=device, dtype=torch.long)

        # flatten passage labels once (list‑of‑lists → flat list)
        passage_labels: list[str] | None = None
        if self.exclude_same_label_negatives and labels_info is not None:
            passage_labels = [lbl for tower in labels_info[1:] for lbl in tower]

        losses: List[Tensor] = []
        total_masked = 0
        total_negatives = 0

        for b in tqdm.trange(
            0,
            bsz,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores = self.similarity_fct(emb_a[b:e], emb_b) * self.scale

            if passage_labels is not None:
                mask = self._create_label_mask(
                    query_labels=labels_info[0][b:e],
                    passage_labels=passage_labels,
                    row_offset=b,
                    device=device,
                )
                self._update_overlap_counts(labels_info[0][b:e], passage_labels, mask, row_offset=b)
                masked_cnt = mask.sum().item()
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                masked_cnt = 0
                mask = None  # type: ignore

            # stats ---------------------------------------------------------
            all_neg = scores.numel() - scores.size(0)  # remove positives (diag)
            total_masked += masked_cnt
            total_negatives += all_neg

            loss_mb = self.cross_entropy_loss(scores, target[b:e]) * scores.size(0) / bsz
            if with_backward:
                loss_mb.backward()
                loss_mb = loss_mb.detach()
            losses.append(loss_mb)

            # save batch‑level metrics for logging ------------------------------
            if total_negatives > 0:
                self._batch_metrics = {
                    "masked_negatives_per_batch": total_masked,
                    "masked_ratio": total_masked / total_negatives,
                    "effective_negatives": total_negatives - total_masked,
                }
            else:
                self._batch_metrics = None

        return torch.stack(losses).sum()

    # --------------------------------------------------------------- forward pass

    def forward(
        self,
        sentence_features: Iterable[dict[str, Any]],
        labels: Tensor | None = None,  # kept for ST compatibility
    ) -> Tensor:  # noqa: D401  – keeps parent signature
        label_lists: list[list[str]] | None = None
        if self.exclude_same_label_negatives:
            label_lists = self._extract_labels(sentence_features)

        # -------------------------- first forward pass (no-grad, cache rand‑states)
        reps: list[list[Tensor]] = []
        self.random_states = []
        for sf in sentence_features:
            reps_mb, states_mb = [], []
            for reps_i, rs_i in self.embed_minibatch_iter(sf, with_grad=False, copy_random_state=True):
                reps_mb.append(reps_i.detach().requires_grad_())
                states_mb.append(rs_i)
            reps.append(reps_mb)
            self.random_states.append(states_mb)

        # ---------------------------- loss + cached backward hook when training
        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps, label_lists)
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            loss = self.calculate_loss(reps, label_lists)
        return loss

    # -------------------------------- override to propagate labels downstream

    def calculate_loss_and_cache_gradients(
        self,
        reps: list[list[Tensor]],
        labels_info: list[list[str]] | None = None,
    ) -> Tensor:  # noqa: D401 – signature kept for ST trainer
        loss = self.calculate_loss(reps, labels_info, with_backward=True)
        loss = loss.detach().requires_grad_()
        self.cache = [[x.grad for x in row] for row in reps]
        return loss


# ===================================================== callback for easy logging

class MaskLoggingCallback(TrainerCallback):
    """Logs masking statistics and per‑scenario heatmaps to W&B."""

    def __init__(self, loss_obj: MaskedCachedMultipleNegativesRankingLoss, run: Run):
        self.loss_obj = loss_obj
        self.run = run

    # per step ------------------------------------------------------------------------
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control, **kwargs):  # type: ignore
        stats = self.loss_obj.pop_batch_metrics()
        if stats is not None:
            self.run.log(stats)

    # epoch‑level heatmaps -------------------------------------------------------------
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control, **kwargs):  # type: ignore
        overlap = self.loss_obj.pop_overlap_counts()
        for scenario, pair_dict in overlap.items():
            if not pair_dict:
                continue
            # build square label set ---------------------------------------------------
            labels = sorted({lbl for pair in pair_dict for lbl in pair})
            size = len(labels)
            mat = [[0] * size for _ in range(size)]
            idx = {lbl: i for i, lbl in enumerate(labels)}
            for (anchor_lbl, neg_lbl), cnt in pair_dict.items():
                mat[idx[anchor_lbl]][idx[neg_lbl]] = cnt
            # log heatmap -------------------------------------------------------------
            table = wandb.Table(data=mat, columns=labels, rows=labels)
            self.run.log({f"label_overlap_heatmap/{scenario}": self._wandb.plot.heatmap(table, "_idx", "_col", "value", title=f"Overlap {scenario}")}, step=state.global_step)


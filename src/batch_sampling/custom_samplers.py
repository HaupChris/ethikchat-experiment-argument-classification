import random
from collections import defaultdict, deque
from torch.utils.data import Sampler

class LabelUniqueSampler(Sampler):
    """1 anchor per label; iterates until all data exhausted."""
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.by_label = defaultdict(list)
        for idx, lab in enumerate(labels):
            self.by_label[lab].append(idx)

    def __iter__(self):
        pools = {l: deque(idxs) for l, idxs in self.by_label.items()}
        labels = list(pools)
        while pools:
            random.shuffle(labels)
            batch = []
            for lab in labels:
                if lab in pools:
                    batch.append(pools[lab].popleft())
                    if not pools[lab]:
                        pools.pop(lab)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
            if batch:
                yield batch  # final short batch

    def __len__(self):
        return sum(len(v) for v in self.by_label.values()) // self.batch_size

class StratifiedSampler(LabelUniqueSampler):
    """Like LabelUnique but oversamples rare labels."""
    def __init__(self, labels, batch_size, tail_boost=2.0):
        super().__init__(labels, batch_size)
        freqs = {l: len(v) for l, v in self.by_label.items()}
        self.weights = {l: (freqs[l] ** -0.5) ** tail_boost for l in freqs}

    def __iter__(self):
        pools = {l: deque(idxs) for l, idxs in self.by_label.items()}
        labels = list(pools)
        while pools:
            weights = [self.weights[l] for l in labels]
            for _ in range(self.batch_size):
                lab = random.choices(labels, weights)[0]
                batch = getattr(self, 'batch', [])
                batch.append(pools[lab].popleft())
                if not pools[lab]:
                    idx = labels.index(lab)
                    labels.pop(idx); weights.pop(idx); pools.pop(lab)
                if len(batch) == self.batch_size:
                    yield batch
                    self.batch = []
            self.batch = batch

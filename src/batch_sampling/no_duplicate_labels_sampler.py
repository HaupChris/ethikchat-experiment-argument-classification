import random
from torch.utils.data import Sampler, BatchSampler

class NoDuplicateLabelsBatchSampler(BatchSampler):
    """
    A BatchSampler that ensures no two examples in the same batch share any label.
    Each example appears exactly once per epoch.
    Expects that each dataset row is something like (anchor, text, labels),
    where `labels` is a list/set of string labels.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        """
        :param dataset: A dataset or list-like object where each item[i] = (anchor, text, labels).
                        Alternatively, it could be a dict-like structure with keys ["anchor", "text", "labels"].
        :param batch_size: Maximum number of examples in a batch (if possible).
        :param shuffle: Whether to shuffle the dataset before creating batches.
        :param drop_last: If True, discard the last batch if it's smaller than batch_size.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Store list of all indices
        self.indices = list(range(len(dataset)))

        # Shuffle entire dataset order if requested
        if self.shuffle:
            random.shuffle(self.indices)

        # This will hold all the final batches, each batch is a list of indices
        self.batches = []
        self._create_batches()

    def _create_batches(self):
        print(f"[NoDuplicateLabelsBatchSampler] Creating batches for {len(self.dataset)} items...")
        used = [False] * len(self.dataset)

        batch_count = 0
        while True:
            used_labels = set()
            current_batch = []

            for idx in self.indices:
                if used[idx]:
                    continue
                example_labels = set(self.dataset[idx]["labels"])

                if used_labels.intersection(example_labels):
                    continue

                current_batch.append(idx)
                used[idx] = True
                used_labels.update(example_labels)

                if len(current_batch) >= self.batch_size:
                    # Batch is 'full'
                    break

            if not current_batch:
                print(f"[NoDuplicateLabelsBatchSampler] No more items could be batched. Exiting loop.")
                break

            self.batches.append(current_batch)
            batch_count += 1
            print(f"[NoDuplicateLabelsBatchSampler] Created batch {batch_count} with {len(current_batch)} items.")

        # If drop_last=True, remove the last batch if it's smaller than batch_size
        if self.drop_last and len(self.batches):
            if len(self.batches[-1]) < self.batch_size:
                print("[NoDuplicateLabelsBatchSampler] Dropping last underfilled batch.")
                self.batches.pop()

        print(f"[NoDuplicateLabelsBatchSampler] Finished creating {len(self.batches)} batches total.")

    def __iter__(self):
        """
        Yield one batch (list of example indices) at a time.
        """
        return iter(self.batches)

    def __len__(self):
        """
        Number of batches total.
        """
        return len(self.batches)

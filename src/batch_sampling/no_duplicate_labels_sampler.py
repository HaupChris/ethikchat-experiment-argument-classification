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
        """ Partition self.indices into label-disjoint batches. """
        used = [False] * len(self.dataset)
        idx_ptr = 0  # We'll iterate over self.indices in a loop

        # We keep looping until we cannot form any more batches
        while True:
            used_labels = set()
            current_batch = []
            # Try to fill up to batch_size
            for idx in self.indices:
                if used[idx]:
                    continue
                # Extract labels from the dataset. Adjust as needed if your dataset has a different structure.
                # Assuming each row is (anchor, text, labels).
                example_labels = set(self.dataset[idx]["labels"])

                # Check if there's an intersection with already-used labels
                if used_labels.intersection(example_labels):
                    continue  # can't add this example, it conflicts with the current batch

                # Otherwise, add it
                current_batch.append(idx)
                used[idx] = True
                used_labels.update(example_labels)

                if len(current_batch) >= self.batch_size:
                    # batch is 'full'
                    break

            if not current_batch:
                # no new example could be placed => we are done
                break

            self.batches.append(current_batch)

        # If drop_last=True, remove the last batch if it's not exactly batch_size
        if self.drop_last and len(self.batches):
            if len(self.batches[-1]) < self.batch_size:
                self.batches.pop()

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

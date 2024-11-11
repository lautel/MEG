import random
import numpy as np
from torch.utils.data import Sampler


class UniqueSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.codes = dataset.codes
    
    def __iter__(self):
        batches = []
        remaining_indices = list(self.codes.keys())
        np.random.shuffle(remaining_indices)
        while remaining_indices:
            batch = []
            used_codes = set()
            
            for idx in list(remaining_indices):
                code = self.codes[idx]
                if code not in used_codes:
                    batch.append(idx)
                    used_codes.add(code)
                    remaining_indices.remove(idx)
                    
                if len(batch) == self.batch_size:
                    break
            batches.extend(batch)
        return iter(batches)
    
    def __len__(self):
        # Total number of batches
        num_samples = len(self.codes)
        return (num_samples + self.batch_size - 1) // self.batch_size


class MinTwoLabelBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.labels = dataset.codes
        self.batch_size = batch_size
        self.label_to_indices = self._group_by_labels()
        self.all_indices = list(range(len(self.labels)))
        
    def _group_by_labels(self):
        # Create a dictionary mapping each label to a list of indices where it occurs
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices
    
    def _has_at_least_two_labels(self, batch_indices):
        # Check if the current batch has at least two distinct labels
        batch_labels = [self.labels[idx] for idx in batch_indices]
        return len(set(batch_labels)) > 1

    def __iter__(self):
        # Shuffle all indices for randomness
        random.shuffle(self.all_indices)
        batch = []

        for idx in self.all_indices:
            batch.append(idx)
            
            if len(batch) == self.batch_size:
                # Check if the batch has at least two different labels
                if self._has_at_least_two_labels(batch):
                    yield batch
                else:
                    # If batch does not meet the condition, keep trying to fix it
                    for i in range(len(batch)):
                        for label, indices in self.label_to_indices.items():
                            if self.labels[batch[i]] != label:
                                # Replace the sample with one from a different label
                                replacement = random.choice(indices)
                                batch[i] = replacement
                                if self._has_at_least_two_labels(batch):
                                    yield batch
                                    break
                batch = []

        # Handle remaining items if any
        if len(batch) > 1 and self._has_at_least_two_labels(batch):
            yield batch

    def __len__(self):
        # Calculate the number of batches
        return len(self.labels) // self.batch_size

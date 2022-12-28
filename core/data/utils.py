from typing import Callable, List

class CompareAttributeSubset:
    def __init__(self, subset_ids: List[int]):
        self.subset_ids = subset_ids

    def __call__(self, attrs, a):
        return (attrs[:, self.subset_ids] == a[self.subset_ids]).sum(-1) == len(self.subset_ids)
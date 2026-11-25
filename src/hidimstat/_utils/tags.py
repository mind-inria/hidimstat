from dataclasses import dataclass

from sklearn.utils import Tags

__all__ = ["HidimstatTags"]


@dataclass
class HidimstatTags(Tags):
    needs_importance_data: bool = True

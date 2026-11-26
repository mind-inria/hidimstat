from dataclasses import dataclass

from sklearn.utils import Tags


@dataclass
class HidimstatTags(Tags):
    needs_importance_data: bool = True

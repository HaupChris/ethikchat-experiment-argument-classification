from dataclasses import dataclass
from enum import Enum


class UtteranceType(Enum):
    User = "user"
    Bot = "bot"
    UserAndBot = "user_and_bot"
    Sample = "sample"


@dataclass
class DatasetConfig:
    """
    Configuration class for dataset creation.

    Attributes:
    ----------
    model_name_or_path : str
        The name or path of the pre-trained model (e.g., Huggingface model) to be used for tokenization.
    dataset_path : str
        The path to the directory where the created dataset will be saved.
    project_dir : str
        The directory head directory of the project from which will be infered where the dialogue files for the dataset creation are stored.
    with_context : bool, optional
        A flag indicating whether to include the previous utterance in the dialogue as context.
    utterance_type : UtteranceType, optional
        The type of utterance that should be used for segmentation or classification. It can be either user or bot
        utterances, or both (default is UtteranceType.UserAndBot).
    downsample_ratio: float = 1.0
        The ratio by which the instances in the dataset with number_of_true_labels == 1 should be downsampled.
    """

    dataset_path: str
    project_dir: str
    with_context: bool
    utterance_type: UtteranceType
    downsample_ratio: float

# was lade ich aus den Dialogen? Texte mit Label. bzw. Utterances aus denen ich Texte mit Label holen kann
# Es gibt verschiedene Formate von Huggingface für ein Dataset, um Sentence Transformer zu benutzen und auch verschiedene Loss functions.
# Für Datensätze gibt es die Formate pair, pair-class, pair-score, triplet
# pair: [text_a, text_b]
# pair-class: [text_a, text_b, label]
# pair-score: [text_a, text_b, score]
# triplet: [anchor, positive, negative]
# https://www.sbert.net/docs/sentence_transformer/training_overview.html#dataset
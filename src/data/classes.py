from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional

from datasets import Dataset
from ethikchat_argtoolkit.ArgumentGraph.stance import Stance
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario


class UtteranceType(Enum):
    User = "user"
    Bot = "bot"
    UserAndBot = "user_and_bot"


@dataclass
class ProcessedUtterance:
    """
    Dataclass for a processed utterance in the dataset.
    """
    id: int
    text: str
    labels: List[str]
    bounds: List[Tuple[int, int]]
    context: List[Tuple[str, str]]
    discussion_scenario: DiscussionSzenario
    scenario_description: str
    scenario_question: str
    user_stance: Stance
    context_labels: List[List[str]]
    original_dialogue_id: str


@dataclass
class NoisyProcessedUtterance(ProcessedUtterance):
    reason: str = ""


class PassageSource(Enum):
    UserUtterance = "user_utterance"
    ArgumentgraphSummary = "argumentgraph_summary"
    ArgumentgraphFullText = "argumentgraph_full_text"
    ArgumentgraphSample = "argumentgraph_sample"


@dataclass
class Passage:
    """
    Dataclass for a passage in the dataset. It contains an id, the text, the label, and the discussion scenario.
    The retrieved_query_id is optional and is used to link the passage to the query from which`s text it was retrieved.
    That means that the passage is relevant to the query but in a trivial way. Because it is part of the query itself.

    """
    id: Optional[int]
    text: str
    label: str
    discussion_scenario: str
    passage_source: str
    retrieved_query_id: Optional[int] = None

    def __eq__(self, other: 'Passage'):
        return ((self.text == other.text) and
                (self.label == other.label) and
                (self.discussion_scenario == other.discussion_scenario))

    @staticmethod
    def get_passages_from_hf_dataset(passages: Dataset) -> List['Passage']:
        return [Passage(
            id=passage["id"],
            text=passage["text"],
            label=passage["label"],
            discussion_scenario=passage["discussion_scenario"],
            passage_source=passage["passage_source"],
            retrieved_query_id=passage["retrieved_query_id"]
        ) for passage in passages]


@dataclass
class Query:
    """
    Attributes:
        id (int)
        text (str)
        labels (List[str])
        discussion_scenario (str)
        context (List[Tuple[str, str]] = field(default_factory=list))
        scenario_description (str = "")
        scenario_question (str = "")
        user_stance (str = "")
        context_labels (List[List[str]] = field(default_factory=list))
        original_dialogue_id ( str = "")

    """
    id: int
    text: str
    labels: List[str]
    discussion_scenario: str
    context: List[Tuple[str, str]] = field(default_factory=list)
    scenario_description: str = ""
    scenario_question: str = ""
    user_stance: str = ""
    context_labels: List[List[str]] = field(default_factory=list)
    original_dialogue_id: str = ""

    def __hash__(self):
        return hash((self.text, tuple(self.labels)))

    def __eq__(self, other):
        return ((self.text == other.text) and
                (self.labels == other.labels))

    @staticmethod
    def get_queries_from_hf_dataset(queries: Dataset) -> List['Query']:
        return [Query(
            id=query["id"],
            text=query["text"],
            labels=query["labels"],
            discussion_scenario=query["discussion_scenario"],
            context=query["context"],
            scenario_description=query["scenario_description"],
            scenario_question=query["scenario_question"],
            user_stance=query["user_stance"],
            context_labels=query["context_labels"]
        ) for query in queries]


@dataclass
class NoisyQuery(Query):
    reason: str = ""

    @staticmethod
    def get_queries_from_hf_dataset(noisy_queries: Dataset) -> List['NoisyQuery']:
        return [NoisyQuery(
            id=query["id"],
            text=query["text"],
            labels=query["labels"],
            discussion_scenario=query["discussion_scenario"],
            context=query["context"],
            scenario_description=query["scenario_description"],
            scenario_question=query["scenario_question"],
            user_stance=query["user_stance"],
            reason=query["reason"]
        ) for query in noisy_queries]


@dataclass
class DatasetConfig:
    """
    Configuration class for dataset creation.

    Attributes:
    ----------
    dataset_path : str
        The path to the directory where the created dataset will be saved.
    project_dir : str
        The directory head directory of the project from which will be infered where the dialogue files for the dataset creation are stored.
    utterance_type : UtteranceType, optional
        The type of utterance that should be used for segmentation or classification. It can be either user or bot
        utterances, or both (default is UtteranceType.UserAndBot).
    eval_size: float, optional
        The size of the validation + test split compared to the train split (default is 0.2).
    validation_test_ratio: float, optional
        The ratio of test to validation split (default is 0.5 for validation and test splits of the same size).
    """
    dataset_path: str
    project_dir: str
    utterance_type: UtteranceType = UtteranceType.UserAndBot
    eval_size: float = 0.2
    validation_test_ratio: float = 0.5


class DatasetSplitType(Enum):
    """
    Enum class for different dataset splits.
    InDistribution makes sure that each label in the valid or test set is also present in the training set but not with the same anchors.
    OutOfDistributionSimple splits the dataset so that the valid and test set contain labels that are not present in the training set but other labels from the same discussion scenario are in the training set.
    OutOfDistributionHard leaves out all labels from a selected discussion scenario in the training set.

    """
    InDistribution = "in_distribution"
    OutOfDistributionSimple = "out_of_distribution_easy"
    OutOfDistributionHard = "out_of_distribution_hard"
    kFold = "k_fold"

    @classmethod
    def from_str(cls, value: str):
        if value == "in_distribution":
            return cls.InDistribution
        if value == "out_of_distribution_easy":
            return cls.OutOfDistributionSimple
        if value == "out_of_distribution_hard":
            return cls.OutOfDistributionHard
        if value == "k_fold":
            return cls.kFold
        raise ValueError(f"Unknown DatasetSplitType: {value}")

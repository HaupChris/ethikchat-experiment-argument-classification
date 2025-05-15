from typing import Union

from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario
from pydantic import BaseModel

from src.data.classes import DatasetSplitType


class ExperimentConfig(BaseModel):
    """
    Configuration for an experiment run.

    This is used to pass the configuration to the training script and log the configuration.
    Attributes:
        project_root: str (absolute path to the project root)
        experiment_dir: str (relative path to the experiments directory)
        experiment_run: str (name of the experiment run)
        dataset_dir: str (relative path to the dataset directory)
        dataset_name: str (name of the dataset)
        dataset_split_type: DatasetSplitType, e.g. DatasetSplitType.InDistribution
        split_dataset_name: str (name of the split dataset), e.g. "dataset_split_in_distribution"
        model_name: str (name of the model to train)
        model_name_escaped: str (model name with "/" replaced by "-")
        model_run_dir: str (absolute path to the model run directory, e.g. /home/user/my_experiments/this_experiment/experiments_outputs/v0/airnicco8-xlm-roberta-de_lr2e-05_bs4)
        learning_rate: float
        batch_size: int or str ("auto")
        num_epochs: int
        loss_function: str (name of the loss function to use, does not change anything but only for logging)
        run_time: str (timestamp of the start of the experiment run)
        warmup_ratio: float (ratio of the total number of training steps to warmup steps)
        context_length: int (-1 for all available context, 0 for no context, > 0 for the specified number of utterances before the given user utterance)
        add_discussion_scenario_info: bool (if true, at the beginning of a query and passage text the discussion_scenario is added, e.g. [MEDAI])
        test_scenario: DiscussionSzenario (the discussion scenario that is to be left out from the train and kept for the test set)
        num_shots_passages: int (The number of passages for each label for each scenario in the dataset. If for a certain lable not enough passages are available, all available are used. -1 selects all)
        num_shots_queries: int (The number of queries for each label for each scenario in the dataset. If for a certain lable not enough queries are available, all available are used. -1 selects all)

    """
    project_root: str
    experiment_dir: str
    experiment_run: str
    dataset_dir: str
    dataset_name: str
    dataset_split_type: DatasetSplitType
    split_dataset_name: str
    model_name: str
    model_name_escaped: str
    model_run_dir: str
    learning_rate: float
    batch_size: Union[int, str]
    num_epochs: int
    loss_function: str
    run_time: str
    warmup_ratio: float
    context_length: int
    add_discussion_scenario_info: bool
    test_scenario: DiscussionSzenario
    num_shots_passages: int
    num_shots_queries: int
    exclude_same_label_negatives: bool


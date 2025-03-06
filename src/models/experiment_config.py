from typing import Union

from pydantic import BaseModel

from src.data.create_corpus_dataset import DatasetSplitType


class ExperimentConfig(BaseModel):
    """
    Configuration for an experiment run.

    This is used to pass the configuration to the training script and log the configuration.
    Args:
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


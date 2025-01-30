import ast
import json
import os
import pandas as pd
import re
import warnings
import random
from torch.utils.data import DataLoader

from dataclasses import dataclass
from datasets import DatasetDict, Dataset, load_from_disk
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple, Literal

from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection
from huggingface_hub import repo_exists
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerFast, AutoTokenizer, RobertaTokenizerFast
from transformers.utils import PaddingStrategy
from ethikchat_argtoolkit.Dialogue.utterance import Utterance
from ethikchat_argtoolkit.ArgumentGraph.stance import Stance
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario
from ethikchat_argtoolkit.Loading.dialogue_loader import DialogueLoader
from ethikchat_argtoolkit.Preprocessing.gender_language_tools import GenderLanguageTools
from ethikchat_argtoolkit.Dialogue.dialogue import Dialogue, UserUtterance, BotUtterance
from ethikchat_argtoolkit.Dialogue.dialogue_szenario import DialogueSzenario

# Define the label mapping
label_list = ['O', 'B-ARG', 'I-ARG', 'X']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

class UtteranceType(Enum):
    User = "user"
    Bot = "bot"
    UserAndBot = "user_and_bot"

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
    num_previous_turns: int
        The amount of previous utterance turns to be included in the conversation context. A single turn is considered as one utterance,
        e.g., for num_previous_turns = 1 the previous bot message would be included as context for every user message and vice versa.
    include_role: bool
        Specifies whether the speaker's role (e.g., [Bot] or [User]) should be included for each utterance in the context.
    sep_token: str, optional
        The token used to seperate between different utterances in the conversation context field (default is "\n").
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
    num_previous_turns: int
    include_role: bool
    sep_token: str = "\n"
    utterance_type: UtteranceType = UtteranceType.UserAndBot
    eval_size: float = 0.2
    validation_test_ratio: float = 0.5


def load_response_template_collection(topic: str) -> ResponseTemplateCollection:
    return ResponseTemplateCollection.from_csv_files(
        templates_directory_path=f"../../data/external/argument_graphs/szenario_{topic}"
    )

def replace_german_chars(text: str) -> str:
    """
    Replaces German characters with their ASCII equivalents.
    """
    replacements = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'Ä': 'Ae',
        'Ö': 'Oe',
        'Ü': 'Ue',
        'ß': 'ss'
    }
    pattern = re.compile('|'.join(re.escape(key) for key in replacements.keys()))
    return pattern.sub(lambda x: replacements[x.group()], text)


def normalize_whitespace(text: str) -> str:
    """
    Normalizes whitespaces by replacing multiple spaces and line breaks with a single space.
    """
    return re.sub(r'\s+', ' ', text)


def preprocess_text(text: str, gender_language_tools: GenderLanguageTools) -> str:
    """
    Applies preprocessing steps to the text.
    """
    # Remove gender language
    text = gender_language_tools.remove(text)
    # Replace German characters
    text = replace_german_chars(text)
    # Normalize whitespaces
    text = normalize_whitespace(text)
    return text


def check_bounds_correctness(utterance: Utterance, dialogue_id) -> None:
    """
    Checks for a given utterance if:
        - All text of the utterance is included within the bounds
        - The punctuations belong to the correct bounds
    """
    # Ensure that bounds are continuous
    for idx in range(len(utterance.true_bounds) - 1):
        current_bounds = utterance.true_bounds[idx]
        next_bounds = utterance.true_bounds[idx + 1]
        if next_bounds[0] != current_bounds[1]:
            raise ValueError(f"Bounds are not continuous in dialogue {dialogue_id}: {current_bounds}, {next_bounds}")

    # Ensure that bounds cover the entire text
    if len(utterance.text) > 0 and ((len(utterance.text)) != int(utterance.true_bounds[-1][1])):
        raise ValueError(
            f"Bounds do not cover whole utterance text! Text length in dialogue {dialogue_id}: {len(utterance.text)}, last bound: {utterance.true_bounds[-1]}"
        )


def utterance_contains_noisy_data(utterance: Utterance) -> bool:
    """
    Function that checks if certain utterances would introduce noisy data into the dataset.
    """
    if "aus der folgenden Liste" in utterance.text:
        return True

    if "Ich bin mir nicht sicher, was du genau meinst." in utterance.text:
        return True


def load_dataset_from_excel_file(file_path: str, discussion_scenario: DiscussionSzenario) -> List[Dialogue]:
    """
    Loads dialogues from an Excel file and constructs Dialogue objects.
    """
    df = pd.read_excel(file_path)
    dialogues = []

    for dialogue_id, dialogue_df in df.groupby('dialogue'):
        utterances = []

        for _, row in dialogue_df.iterrows():
            if row['utterance_type'] == 'user':
                utterance = UserUtterance(
                    user="user",
                    text=row['utterance_text'],
                    timestamp=datetime.now(),
                    id_in_dialogue=int(row['utterance_id']),
                    true_labels=ast.literal_eval(row['true_labels']),
                    true_bounds=ast.literal_eval(row['true_bounds'])
                )
            else:
                utterance = BotUtterance(
                    user="bot",
                    text=row['utterance_text'],
                    timestamp=datetime.now(),
                    id_in_dialogue=int(row['utterance_id']),
                    true_labels=ast.literal_eval(row['true_labels']),
                    true_bounds=ast.literal_eval(row['true_bounds'])
                )
            check_bounds_correctness(utterance, dialogue_id)
            utterances.append(utterance)

        dialogue = Dialogue(
            start=datetime.now(),
            szenario=DialogueSzenario.STD_BOT,
            discussion_szenario=discussion_scenario,
        )
        dialogue.utterances = utterances
        dialogues.append(dialogue)
    return dialogues


def preprocess_utterance(
        utterance: Utterance,
        gender_language_tools: GenderLanguageTools):
    """
    Preprocesses a single utterance by applying text normalization and updating bounds.
    """
    processed_bounds_text = []
    new_bounds = []
    cumulative_length = 0
    for idx, (start, end) in enumerate(utterance.true_bounds):
        bound_text = utterance.text[start:end]
        bound_label = utterance.true_labels[idx]

        processed_text = preprocess_text(bound_text, gender_language_tools)

        processed_bounds_text.append({'text': processed_text, 'label': bound_label})

        new_start = cumulative_length
        new_end = new_start + len(processed_text)
        new_bounds.append((new_start, new_end))
        cumulative_length = new_end

    processed_text = ''.join([item['text'] for item in processed_bounds_text])

    utterance.text = processed_text
    utterance.true_bounds = new_bounds
    utterance.true_labels = [item['label'] for item in processed_bounds_text]

    if len(utterance.true_bounds) != len(utterance.true_labels):
        raise ValueError(f"bounds and values do not have the same length after preprocessing for utterance {utterance}.\n"
                         f"bounds: {utterance.true_bounds}, labels:{utterance.true_labels}")

    return utterance.text, utterance.true_labels, utterance.true_bounds


def build_context(dialogue_turns, num_previous_turns, sep_token, include_role):
    selected_turns = dialogue_turns[-(num_previous_turns + 1):-1]

    if include_role:
        result = sep_token.join(f"[{role}] {text}" for role, text in selected_turns)
    else:
        result = sep_token.join(text for _, text in selected_turns)

    return result


def preprocess_dataset(dialogues: List[Dialogue],
                       num_previous_turns: int,
                       utterance_type: UtteranceType,
                       sep_token: str,
                       include_role: bool):
    labels_list = []
    text_list = []
    bounds_list = []
    previous_context_list = []
    topics_list = []

    gender_language_tools = GenderLanguageTools()

    for dialogue in dialogues:
        dialogue_turns = []
        for utterance in dialogue.utterances:
            check_bounds_correctness(utterance, dialogue.name)

            dialogue_turns.append((
                "User" if utterance.is_from_user() else "Bot",
                preprocess_text(utterance.text, gender_language_tools)
            ))

            if utterance_type == UtteranceType.User and not utterance.is_from_user():
                continue

            if utterance_type == UtteranceType.Bot and utterance.is_from_user():
                continue

            if utterance_contains_noisy_data(utterance):
                continue

            text, labels, bounds = preprocess_utterance(utterance, gender_language_tools)
            previous_context = build_context(dialogue_turns, num_previous_turns, sep_token, include_role)

            labels_list.append(labels)
            text_list.append(text)
            bounds_list.append(bounds)
            previous_context_list.append(previous_context)
            topics_list.append(dialogue.discussion_szenario)

    return labels_list, text_list, bounds_list, previous_context_list, topics_list


def extract_positive_passages(labels: List[str], rtc: ResponseTemplateCollection) -> List[str]:
    allowed_labels = rtc.z_arguments_labels.union(rtc.nz_arguments_labels)

    positive_passages = []
    for label in labels:
        if label in allowed_labels:
            template = rtc.get_template_for_label(label)
            positive_passages.extend([template.summary, template.full_text])

    return positive_passages


def extract_negative_passages(labels: List[str], rtc: ResponseTemplateCollection) -> List[str]:
    label_pool = rtc.z_arguments_labels.union(rtc.nz_arguments_labels)

    label_pool.difference_update(labels)

    negative_passages = []
    for label in label_pool:
        template = rtc.get_template_for_label(label)
        negative_passages.extend([template.summary, template.full_text])

    return negative_passages


def extract_hard_negative_passages(labels: List[str], rtc: ResponseTemplateCollection) -> List[str]:
    allowed_labels = rtc.z_arguments_labels.union(rtc.nz_arguments_labels)

    hard_negative_passages = []
    for label in labels:
        if label in allowed_labels:
            template = rtc.get_template_for_label(label)

            for counter_template in template.child_templates:
                hard_negative_passages.extend([counter_template.summary, counter_template.full_text])

    return hard_negative_passages


def create_dataset(config: DatasetConfig) -> None:
    save_path = config.dataset_path
    project_dir = config.project_dir
    num_previous_turns = config.num_previous_turns
    include_role = config.include_role
    sep_token = config.sep_token
    utterance_type = config.utterance_type
    eval_size = config.eval_size
    validation_test_ratio = config.validation_test_ratio

    path_mensateria_survey_1 = os.path.join(project_dir, "data", "external", "ethikchat_data-main", "mensateria_survey", "processed", "curated")
    path_mensateria_survey_2 = os.path.join(project_dir, "data", "external", "ethikchat_data-main", "mensateria_survey_2", "processed", "curated")

    m1_dialogues_medai = load_dataset_from_excel_file(
        os.path.join(path_mensateria_survey_1, "mensateria_survey_medai_curated_dialogues.xlsx"),
        DiscussionSzenario.MEDAI
    )
    m1_dialogues_jurai = load_dataset_from_excel_file(
        os.path.join(path_mensateria_survey_1, "mensateria_survey_jurai_curated_dialogues.xlsx"),
        DiscussionSzenario.JURAI
    )
    m1_dialogues_autoai = load_dataset_from_excel_file(
        os.path.join(path_mensateria_survey_1, "mensateria_survey_autoai_curated_dialogues.xlsx"),
        DiscussionSzenario.AUTOAI
    )
    m2_dialogues_medai = load_dataset_from_excel_file(
        os.path.join(path_mensateria_survey_2, "mensateria_survey_2_medai_curated_dialogues.xlsx"),
        DiscussionSzenario.MEDAI
    )
    m2_dialogues_jurai = load_dataset_from_excel_file(
        os.path.join(path_mensateria_survey_2, "mensateria_survey_2_jurai_curated_dialogues.xlsx"),
        DiscussionSzenario.JURAI
    )
    m2_dialogues_autoai = load_dataset_from_excel_file(
        os.path.join(path_mensateria_survey_2, "mensateria_survey_2_autoai_curated_dialogues.xlsx"),
        DiscussionSzenario.AUTOAI
    )
    m2_dialogues_refai = load_dataset_from_excel_file(
        os.path.join(path_mensateria_survey_2, "mensateria_survey_2_refai_curated_dialogues.xlsx"),
        DiscussionSzenario.REFAI
    )
    m3_dialogues_medai = DialogueLoader.from_directory(
        dialogues_directory_path=os.path.join(project_dir, "data", "external", "ethikchat_data-main", "mensateria_survey_3", "processed", "medai"),
        version="webathen"
    )

    for m3_dialogue in m3_dialogues_medai:
        m3_dialogue.discussion_szenario = DiscussionSzenario.MEDAI

    dialogues = (m1_dialogues_medai + m1_dialogues_jurai + m1_dialogues_autoai
                 + m2_dialogues_medai + m2_dialogues_jurai + m2_dialogues_autoai + m2_dialogues_refai
                 + m3_dialogues_medai)

    labels, texts, bounds, contexts, topics = preprocess_dataset(dialogues,
                                                                 num_previous_turns,
                                                                 utterance_type,
                                                                 sep_token,
                                                                 include_role)

    rtc_med = load_response_template_collection("s1")
    rtc_jur = load_response_template_collection("s2")
    rtc_auto = load_response_template_collection("s3")
    rtc_ref = load_response_template_collection("s4")

    positive_passages = []
    negative_passages = []
    for label, topic in zip(labels, topics):
        if topic == DiscussionSzenario.MEDAI:
            rtc = rtc_med
        elif topic == DiscussionSzenario.JURAI:
            rtc = rtc_jur
        elif topic == DiscussionSzenario.AUTOAI:
            rtc = rtc_auto
        elif topic == DiscussionSzenario.REFAI:
            rtc = rtc_ref
        else:
            raise Exception(f"Invalid discussion topic: {topic}")

        positive_passages.append(extract_positive_passages(label, rtc))
        negative_passages.append(extract_negative_passages(label, rtc))

    data = []
    for utterance, context, positive_passages, negative_passages in zip(texts, contexts, positive_passages, negative_passages):
        data.append({
            "utterance": utterance,
            "context": context,
            "positive_passages": positive_passages,
            "negative_passages": negative_passages
        })

    train_data, evaluation = train_test_split(data, test_size=eval_size)
    validation_data, test_data = train_test_split(evaluation, test_size=validation_test_ratio)

    splits_save_path = os.path.join(save_path, "splits")

    if not os.path.exists(splits_save_path):
        os.makedirs(splits_save_path)

    with open(os.path.join(splits_save_path, "train.json"), "w", encoding="utf-8") as train_file:
        json.dump(train_data, train_file, ensure_ascii=False, indent=4)

    with open(os.path.join(splits_save_path, "validation.json"), "w", encoding="utf-8") as validation_file:
        json.dump(validation_data, validation_file, ensure_ascii=False, indent=4)

    with open(os.path.join(splits_save_path, "test.json"), "w", encoding="utf-8") as test_file:
        json.dump(test_data, test_file, ensure_ascii=False, indent=4)

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(validation_data)
    test_dataset = Dataset.from_list(test_data)

    hf_dataset = DatasetDict({"train": train_dataset, "validation": eval_dataset, "test": test_dataset})
    hf_dataset.save_to_disk(save_path)


if __name__ == "__main__":
    pass
    # Beispiel zum erstellen eines Datensatzes. Mögliche Optionen von DatasetConfig sind im DocString beschrieben.
    # create_dataset(
    #     DatasetConfig(
    #         dataset_path="dummy_dataset",
    #         project_dir="../../",
    #         num_previous_turns=3,
    #         include_role=True,
    #         sep_token="[SEP]",
    #         utterance_type=UtteranceType.User,
    #         eval_size=0.5,
    #         validation_test_ratio=0.5
    #     )
    # )

    # Beispiel zum Laden des Datensatzes + collate_function des DataLoaders um dynamisch ein Subset der negative passages zu laden.
    hf_dataset = load_from_disk("dummy_dataset")

    def collate_fn(batch, num_negative_samples: int):
        utterances = []
        contexts = []
        positive_passages = []
        negative_passages = []

        for item in batch:
            utterances.append(item["utterance"])
            contexts.append(item["context"])
            positive_passages.append(item["positive_passages"])

            negative_passage_pool = item["negative_passages"]
            sampled_negatives = random.sample(negative_passage_pool, min(num_negative_samples, len(negative_passage_pool)))
            negative_passages.append(sampled_negatives)

        return {
            "utterances": utterances,
            "contexts": contexts,
            "positive_passages": positive_passages,
            "negative_passages": negative_passages
        }

    train_loader = DataLoader(
        hf_dataset["train"],
        batch_size = 2,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, num_negative_samples=2)
    )

    for batch in train_loader:
        print(f"Utterances: {batch['utterances']}")
        print(f"Contexts: {batch['contexts']}")
        print(f"Positive Passages: {batch['positive_passages']}")
        print(f"Negative Passages: {batch['negative_passages']}\n")
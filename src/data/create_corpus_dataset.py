import ast
import os
import pandas as pd
import re

from dataclasses import dataclass
from datasets import DatasetDict, Dataset, load_from_disk
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Dict

from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection

from ethikchat_argtoolkit.Dialogue.utterance import Utterance
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario
from ethikchat_argtoolkit.Loading.dialogue_loader import DialogueLoader
from ethikchat_argtoolkit.Preprocessing.gender_language_tools import GenderLanguageTools
from ethikchat_argtoolkit.Dialogue.dialogue import Dialogue, UserUtterance, BotUtterance
from ethikchat_argtoolkit.Dialogue.dialogue_szenario import DialogueSzenario


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


class DatasetSplitType(Enum):
    """
    Enum class for different dataset splits.
    Simple is a simple train, validation, test split in the ratio 80:10:10.
    ByDiscussionSzenario leaves out one discussion szenario for Test and uses the other ones for Train and Validation.
    kFold splits the dataset into k folds for cross-validation.

    """
    Simple = "simple"
    ByDiscussionSzenario = "by_discussion_szenario"
    kFold = "k_fold"


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


def check_bounds_correctness(utterance: Utterance, dialogue_id: int) -> None:
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
        gender_language_tools: GenderLanguageTools) -> Tuple[str, List[str], List[Tuple[int, int]]]:
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
        raise ValueError(
            f"bounds and values do not have the same length after preprocessing for utterance {utterance}.\n"
            f"bounds: {utterance.true_bounds}, labels:{utterance.true_labels}")

    return utterance.text, utterance.true_labels, utterance.true_bounds


def build_context(dialogue_turns: List[Tuple[str, str]], num_previous_turns: int, sep_token: str,
                  include_role: bool) -> str:
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
                       include_role: bool) -> Tuple[
    List[List[str]], List[str], List[List[Tuple[int, int]]], List[str], List[DiscussionSzenario]]:
    labels_list = []
    utterance_texts_list = []
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

            processed_utterance_text, processed_labels, processed_bounds = preprocess_utterance(utterance,
                                                                                                gender_language_tools)
            previous_context = build_context(dialogue_turns, num_previous_turns, sep_token, include_role)

            labels_list.append(processed_labels)
            utterance_texts_list.append(processed_utterance_text)
            bounds_list.append(processed_bounds)
            previous_context_list.append(previous_context)
            topics_list.append(dialogue.discussion_szenario)

    return labels_list, utterance_texts_list, bounds_list, previous_context_list, topics_list


def extract_positive_passages(labels: List[str], rtc: ResponseTemplateCollection) -> List[str]:
    allowed_labels = rtc.arguments_labels
    labels = [label for label in labels if label in allowed_labels]

    positive_passages = []
    for label in labels:
        template = rtc.get_template_for_label(label)
        positives = [template.summary, template.full_text]
        positives.extend(template.samples)
        positive_passages.extend(positives)

    return positive_passages


def extract_negative_passages(labels: List[str], rtc: ResponseTemplateCollection) -> List[str]:
    label_pool = rtc.arguments_labels.difference(labels)

    negative_passages = []
    for label in label_pool:
        template = rtc.get_template_for_label(label)
        negatives = [template.summary, template.full_text]
        negatives.extend(template.samples)
        negative_passages.extend(negatives)

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


def create_queries_split(utterances: List[str], discussion_szenario: DiscussionSzenario) -> List[Tuple[int, str]]:
    """
    Creates a split that contains utterances and an id. This is the "queries" split.
    """
    return [(idx, utterance, discussion_szenario.value) for idx, utterance in enumerate(utterances)]


def create_queries_relevant_passages_mapping_split(queries_labels: List[List[str]],
                                                   queries_discussion_scenarios: List[DiscussionSzenario],
                                                   passages_split: List[Tuple[int, str, str, str]]) -> Dict[
    int, List[int]]:
    """
    Creates a mapping from query ids to relevant passage ids. This is the "queries_relevant_passages_mapping" split.
    """
    if len(queries_labels) != len(queries_discussion_scenarios):
        raise ValueError("Queries labels and discussion scenarios should have the same length.")


    queries_relevant_passages_mapping = {}
    for query_idx, (query_labels, query_discussion_scenario) in enumerate(zip(queries_labels, queries_discussion_scenarios)):
        relevant_passages = []
        for passage_idx, passage_text, passage_label, passage_discussion_scenario in passages_split:
            if passage_label in query_labels and query_discussion_scenario == passage_discussion_scenario:
                relevant_passages.append(passage_idx)
        queries_relevant_passages_mapping[query_idx] = relevant_passages

    return queries_relevant_passages_mapping


def create_passages_from_utterances(utterances: List[str],
                                    bounds: List[List[Tuple[int, int]]],
                                    labels: List[List[str]],
                                    utterances_discussion_scenarios: List[DiscussionSzenario]) -> List[Tuple[str, str]]:
    """
    Creates passages from utterances.
    """
    passages = []
    for utterance, utterance_bounds, utterance_labels, utterance_discussion_scenario in zip(utterances, bounds, labels, utterances_discussion_scenarios):
        for idx, (start, end) in enumerate(utterance_bounds):
            passage = utterance[start:end]
            passages.append((passage, utterance_labels[idx], utterance_discussion_scenario.value))
    return passages


def create_passages_from_argument_graph(argument_graph: ResponseTemplateCollection, discussion_scenario: DiscussionSzenario) -> List[Tuple[str, str]]:
    """
    Creates passages from the argument graph.
    """
    passages = []

    for template in argument_graph.arguments_templates:
        passages.append((template.summary, template.label, discussion_scenario.value))
        passages.append((template.full_text, template.label, discussion_scenario.value))
        passages.extend([(sample, template.label, discussion_scenario.value) for sample in template.samples])
    return passages


def create_dataset_splits(dialogues: List[Dialogue],
                          include_role: bool,
                          num_previous_turns: int,
                          sep_token: str,
                          utterance_type: UtteranceType,
                          argument_graphs: Dict[DiscussionSzenario, ResponseTemplateCollection]) \
        -> Tuple[List[Tuple[int, str, str]],
        List[Tuple[int, str, str, str]],
        Dict[int, List[int]]]:
    """
    Creates the dataset splits for the information retrieval task.
    """
    utterance_labels, utterance_texts, utterance_bounds, contexts, utterances_discussion_scenarios = preprocess_dataset(
        dialogues,
        num_previous_turns,
        utterance_type,
        sep_token,
        include_role)
    # discussion_scenario is known for every utterance

    # create passages from utterances
    utterances_passages = create_passages_from_utterances(utterance_texts, utterance_bounds, utterance_labels, utterances_discussion_scenarios)

    argument_graphs_passages = []
    for discussion_scenario, argument_graph in argument_graphs.items():
        argument_graphs_passages.extend(create_passages_from_argument_graph(argument_graph, discussion_scenario))


    # TODO: currently, queries are only an utterance. This should be extended to include the context as well.
    queries_split = [(idx, utterance, utterance_scenario.value) for idx, (utterance, utterance_scenario) in
                     enumerate(zip(utterance_texts, utterances_discussion_scenarios))]

    passages_split = [(idx, passage, label, discussion_scenario) for idx, (passage, label, discussion_scenario) in
                      enumerate(utterances_passages + argument_graphs_passages)]

    queries_relevant_passages_mapping_split = create_queries_relevant_passages_mapping_split(
        utterance_labels, utterances_discussion_scenarios, passages_split
    )

    return queries_split, passages_split, queries_relevant_passages_mapping_split


def create_dataset(config: DatasetConfig) -> None:
    save_path = config.dataset_path
    project_dir = config.project_dir
    num_previous_turns = config.num_previous_turns
    include_role = config.include_role
    sep_token = config.sep_token
    utterance_type = config.utterance_type

    path_mensateria_survey_1 = os.path.join(project_dir, "data", "external", "ethikchat_data-main", "mensateria_survey",
                                            "processed", "curated")
    path_mensateria_survey_2 = os.path.join(project_dir, "data", "external", "ethikchat_data-main",
                                            "mensateria_survey_2", "processed", "curated")

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
        dialogues_directory_path=os.path.join(project_dir, "data", "external", "ethikchat_data-main",
                                              "mensateria_survey_3", "processed", "medai"),
        version="webathen"
    )
    # TODO: load here dialogues from older studies.

    for m3_dialogue in m3_dialogues_medai:
        m3_dialogue.discussion_szenario = DiscussionSzenario.MEDAI

    dialogues_medai = m1_dialogues_medai + m2_dialogues_medai + m3_dialogues_medai
    dialogues_jurai = m1_dialogues_jurai + m2_dialogues_jurai
    dialogues_autoai = m1_dialogues_autoai + m2_dialogues_autoai
    dialogues_refai = m2_dialogues_refai

    all_dialogues = dialogues_medai + dialogues_jurai + dialogues_autoai + dialogues_refai

    argument_graph_med = load_response_template_collection("s1")
    argument_graph_jur = load_response_template_collection("s2")
    argument_graph_auto = load_response_template_collection("s3")
    argument_graph_ref = load_response_template_collection("s4")

    argument_graphs = {
        DiscussionSzenario.MEDAI: argument_graph_med,
        DiscussionSzenario.JURAI: argument_graph_jur,
        DiscussionSzenario.AUTOAI: argument_graph_auto,
        DiscussionSzenario.REFAI: argument_graph_ref
    }


    queries, passages, queries_relevant_passages_mapping = create_dataset_splits(
        all_dialogues, include_role, num_previous_turns, sep_token, utterance_type, argument_graphs)


    # medai_queries, medai_passages, medai_queries_relevant_passages_mapping = create_dataset_splits(
    #     dialogues_medai, include_role, num_previous_turns, sep_token, utterance_type, argument_graph_med,
    #     DiscussionSzenario.MEDAI
    # )
    # jurai_queries, jurai_passages, jurai_queries_relevant_passages_mapping = create_dataset_splits(
    #     dialogues_jurai, include_role, num_previous_turns, sep_token, utterance_type, argument_graph_jur,
    #     DiscussionSzenario.JURAI
    # )
    # autoai_queries, autoai_passages, autoai_queries_relevant_passages_mapping = create_dataset_splits(
    #     dialogues_autoai, include_role, num_previous_turns, sep_token, utterance_type, argument_graph_auto,
    #     DiscussionSzenario.AUTOAI
    # )
    # refai_queries, refai_passages, refai_queries_relevant_passages_mapping = create_dataset_splits(
    #     dialogues_refai, include_role, num_previous_turns, sep_token, utterance_type, argument_graph_ref,
    #     DiscussionSzenario.REFAI
    # )

    # queries = medai_queries + jurai_queries + autoai_queries + refai_queries
    # passages = medai_passages + jurai_passages + autoai_passages + refai_passages
    # queries_relevant_passages_mapping = {**medai_queries_relevant_passages_mapping,
    #                                      **jurai_queries_relevant_passages_mapping,
    #                                      **autoai_queries_relevant_passages_mapping,
    #                                      **refai_queries_relevant_passages_mapping}

    # create hf dataset
    hf_dataset = DatasetDict({
        "queries": Dataset.from_dict({"id": [query_id for query_id, _, _ in queries],
                                      "text": [query_text for _, query_text, _ in queries],
                                      "discussion_scenario": [discussion_scenario for _, _, discussion_scenario in
                                                              queries]}),
        "passages": Dataset.from_dict({"id": [passage_id for passage_id, _, _, _ in passages],
                                       "text": [passage_text for _, passage_text, _, _ in passages],
                                       "label": [passage_label for _, _, passage_label, _ in passages],
                                       "discussion_scenario": [discussion_scenario for _, _, _, discussion_scenario in
                                                               passages]}),
        "queries_relevant_passages_mapping": Dataset.from_dict({
            "query_id": [id for id, _ in queries_relevant_passages_mapping.items()],
            "passages_ids": [ids for _, ids in queries_relevant_passages_mapping.items()]
        })
    })

    hf_dataset.save_to_disk(save_path)


if __name__ == "__main__":

    if not os.path.exists("dummy_dataset"):
        # Beispiel zum Erstellen eines Datensatzes. Mögliche Optionen von DatasetConfig sind im DocString beschrieben.
        create_dataset(
            DatasetConfig(
                dataset_path="dummy_dataset",
                project_dir="../../",
                num_previous_turns=3,
                include_role=True,
                sep_token="[SEP]",
                utterance_type=UtteranceType.User,
                eval_size=0.5,
                validation_test_ratio=0.5
            )
        )

    # Beispiel zum Laden des Datensatzes + collate_function des DataLoaders um dynamisch ein Subset der negative passages zu laden.
    hf_dataset = load_from_disk("dummy_dataset")

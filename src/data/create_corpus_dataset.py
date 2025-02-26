import ast
import os
import pandas as pd
import re

from dataclasses import dataclass
from datasets import DatasetDict, Dataset, load_from_disk
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Dict, Optional

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


@dataclass
class ProcessedUtterance:
    """
    Dataclass for a processed utterance in the dataset.
    """
    id: int
    text: str
    labels: List[str]
    bounds: List[Tuple[int, int]]
    context: str
    discussion_scenario: DiscussionSzenario


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
    retrieved_query_id: Optional[int] = None


@dataclass
class Query:
    """
    Dataclass for a query in the dataset.
    """
    id: int
    text: str
    labels: List[str]
    discussion_scenario: str


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

    @classmethod
    def from_str(cls, value: str):
        if value == "simple":
            return cls.Simple
        if value == "by_discussion_szenario":
            return cls.ByDiscussionSzenario
        if value == "k_fold":
            return cls.kFold
        raise ValueError(f"Unknown DatasetSplitType: {value}")


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
        print(f"Flagged utterance with 'aus der folgenden Liste' in text: {utterance.text} as noisy data.")
        return True

    if "Ich bin mir nicht sicher, was du genau meinst." in utterance.text:
        print(f"Flagged utterance with 'bot uttering not understanding': {utterance.text} as noisy data.")
        return True

    if utterance.true_labels == ['OTHER']:
        print(f"Flagged utterance with OTHER label: {utterance.text}, {utterance.true_labels} as noisy data.")
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
            check_bounds_correctness(utterance, int(dialogue_id))
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
                       include_role: bool) -> List[ProcessedUtterance]:
    """
    Preprocesses a dataset by applying text normalization and updating bounds.
    Returns a list of ProcessedUtterance objects.
    """

    processed_utterances = []

    gender_language_tools = GenderLanguageTools()


    id_counter = 0
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

            processed_utterances.append(ProcessedUtterance(
                id=id_counter,
                text=processed_utterance_text,
                labels=processed_labels,
                bounds=processed_bounds,
                context=previous_context,
                discussion_scenario=dialogue.discussion_szenario
            ))
            id_counter += 1


    return processed_utterances


def create_queries_split(processed_utterances: List[ProcessedUtterance]) -> List[Query]:
    """
    Creates a split that contains utterances and an id. This is the "queries" split.
    """

    return [Query(utt.id, utt.text, utt.labels, utt.discussion_scenario) for utt in processed_utterances]


def create_queries_relevant_passages_mapping_split(queries: List[Query], passages: List[Passage]) -> Dict[int, List[int]]:
    """
    Creates a mapping from query ids to relevant passage ids. This is the "queries_relevant_passages_mapping" split.
    """

    queries_relevant_passages_mapping = {}
    for query in queries:
        relevant_passages = []
        for passage in passages:
            if passage.label in query.labels and query.discussion_scenario == passage.discussion_scenario:
                relevant_passages.append(passage.id)
        queries_relevant_passages_mapping[query.id] = relevant_passages

    return queries_relevant_passages_mapping


def create_passages_from_utterances(processed_utterances: List[ProcessedUtterance], excluded_labels: List[str]) -> List[Passage]:
    """
    Creates passages from utterances.
    """
    passages = []

    for pu in processed_utterances:
        for idx, (start, end) in enumerate(pu.bounds):
            passage_text = pu.text[start:end]

            if pu.labels[idx] not in excluded_labels:
                passages.append(Passage(id=None,
                                        text=passage_text,
                                        label=pu.labels[idx],
                                        discussion_scenario=pu.discussion_scenario.value,
                                        retrieved_query_id=pu.id))
    return passages


def create_passages_from_argument_graph(argument_graph: ResponseTemplateCollection,
                                        discussion_scenario: DiscussionSzenario) -> List[Passage]:
    """
    Creates passages from the argument graph.
    """
    passages = []

    for template in argument_graph.arguments_templates:
        passages.append(Passage(None, template.summary, template.label, discussion_scenario.value, None))
        passages.append(Passage(None, template.full_text, template.label, discussion_scenario.value, None))
        passages.extend([Passage(None, sample, template.label, discussion_scenario.value, None) for sample in template.samples])
    return passages


def create_queries_trivial_passages_mapping_split(queries: List[Query], passages: List[Passage]) -> Dict[int, List[int]]:
    """
    Creates a mapping from query ids to trivial passage ids. A trivial passage is a passage that was extracted from the
    query itself. And is therefore partly or fully identical to the query.
    """
    queries_trivial_passages_mapping = {}

    for query in queries:
        trivial_passages = []
        for passage in passages:
            if passage.retrieved_query_id == query.id:
                trivial_passages.append(passage.id)
        queries_trivial_passages_mapping[query.id] = trivial_passages

    return queries_trivial_passages_mapping


def create_dataset_splits(dialogues: List[Dialogue],
                          include_role: bool,
                          num_previous_turns: int,
                          sep_token: str,
                          utterance_type: UtteranceType,
                          argument_graphs: Dict[DiscussionSzenario, ResponseTemplateCollection]) \
        -> Tuple[List[Query], List[Passage], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Creates the dataset splits for the information retrieval task. This consists of the following splits:
    - queries: A split containing the queries.
    - passages: A split containing the passages.
    - queries_relevant_passages_mapping: A mapping from query ids to relevant passage ids.
    - queries_trivial_passages_mapping: A mapping from query ids to trivial passage ids. (Passages that are part of the query itself)
    """
    processed_utterances = preprocess_dataset(
        dialogues,
        num_previous_turns,
        utterance_type,
        sep_token,
        include_role)

    # create passages from utterances
    utterances_passages = create_passages_from_utterances(processed_utterances)

    argument_graphs_passages = []
    for discussion_scenario, argument_graph in argument_graphs.items():
        argument_graphs_passages.extend(create_passages_from_argument_graph(argument_graph, discussion_scenario))

    # TODO: currently, queries are only an utterance. This should be extended to include the context as well.
    # im queries split hat jede query_id die riehenfolge der processed_utterances.
    queries = create_queries_split(processed_utterances)


    # im passages_split entsprechen die retrieved_query_ids auch
    passages = [Passage(idx, passage.text, passage.label, passage.discussion_scenario, passage.retrieved_query_id) for idx, passage in
                      enumerate(utterances_passages + argument_graphs_passages)]

    queries_relevant_passages_mapping = create_queries_relevant_passages_mapping_split(queries, passages)
    queries_trivial_passages_mapping = create_queries_trivial_passages_mapping_split(queries, passages)

    return queries, passages, queries_relevant_passages_mapping, queries_trivial_passages_mapping


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

    queries, passages, queries_relevant_passages_mapping, queries_trivial_passages_mapping = create_dataset_splits(
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
        "queries": Dataset.from_dict({"id": [query.id for query in queries],
                                      "text": [query.text for query in queries],
                                      "labels": [query.labels for query in queries],
                                      "discussion_scenario": [query.discussion_scenario for query in queries]}),
        "passages": Dataset.from_dict({"id": [passage.id for passage in passages],
                                       "text": [passage.text for passage in passages],
                                       "label": [passage.label for passage in passages],
                                       "discussion_scenario": [passage.discussion_scenario for passage in passages]}),
        "queries_relevant_passages_mapping": Dataset.from_dict({
            "query_id": [id for id, _ in queries_relevant_passages_mapping.items()],
            "passages_ids": [ids for _, ids in queries_relevant_passages_mapping.items()]
        }),
        "queries_trivial_passages_mapping": Dataset.from_dict({
            "query_id": [id for id, _ in queries_trivial_passages_mapping.items()],
            "passages_ids": [ids for _, ids in queries_trivial_passages_mapping.items()]
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

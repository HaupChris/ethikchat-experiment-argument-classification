import ast
import os
import pandas as pd
import re

from dataclasses import dataclass, field
from datasets import DatasetDict, Dataset, load_from_disk
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Dict, Optional

from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection
from ethikchat_argtoolkit.ArgumentGraph.stance import Stance

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
    context: List[Tuple[str, str]]
    discussion_scenario: DiscussionSzenario
    scenario_description: str
    scenario_question: str

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


@dataclass
class Query:
    """
    Person represents a user record.

    Attributes:
        id (int)
        text (str)
        labels (List[str])
        discussion_scenario (str)
        context (List[Tuple[str, str]] = field(default_factory=list))
        scenario_description (str = "")
        scenario_question (str = "")

    """
    id: int
    text: str
    labels: List[str]
    discussion_scenario: str
    context: List[Tuple[str, str]] = field(default_factory=list)
    scenario_description: str = ""
    scenario_question: str = ""

    def __hash__(self):
        return hash(("text", self.text, "labels", self.labels))

    def __eq__(self, other):
        return ((self.text == other.text) and
                (self.labels == other.labels))


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
    noisy_labels = ['OTHER', "INTRO",
                    "Z_ARG", "PRO_ZARG", "CON_ZARG",
                    "NZ_ARG", "PRO_NZARG", "CON_NZARG",
                    "FAQ.G1",
                    "Z.GK1", "Z.GK2", "Z.GK3", "Z.GK4", "Z.GK5", "Z.GK6", "Z.GK7", "Z.GK8", "Z.GK9", "Z.GK10",
                    "Z.GP1", "Z.GP2", "Z.GP3", "Z.GP4", "Z.GP5", "Z.GP6", "Z.GP7", "Z.GP8", "Z.GP9", "Z.GP10",
                    "NZ.G1", "NZ.G2", "NZ.G3", "NZ.G4",
                    "NZ.P1-1", "NZ.P2-1", "NZ.P3-1", "NZ.P4-1", "NZ.P5-1", "NZ.P6-1", "NZ.P7-1", "NZ.P8-1", "NZ.P9-1",
                    "NZ.P10-1",
                    "NZ.K1-1", "NZ.K2-1", "NZ.K3-1", "NZ.K4-1", "NZ.K5-1", "NZ.K6-1", "NZ.K7-1", "NZ.K8-1", "NZ.K9-1",
                    "NZ.K10-1",
                    "CONSENT", "DISSENT"]


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


def get_scenario_question(discussion_scenario: DiscussionSzenario) -> str:
    descriptions = {
        DiscussionSzenario.MEDAI: "Darf ein ausreichend gutes medizinisches Programm aus ethischer Sicht selbstständig Diagnose- und Therapie-Entscheidungen treffen?",
        DiscussionSzenario.JURAI: "Soll im Zivilrecht (kein Strafrecht und kein Familienrecht) ein juristisch intelligentes Programm (jurKI) Streitfälle in erster Instanz entscheiden, wenn seine Urteile besser sind, d.h. seltener in der Berufung korrigiert werden als bei Richter:innen?",
        DiscussionSzenario.AUTOAI: "Wir gehen von einem zukünftigen Szenario aus, indem eine KI autonom ein Fahrzeug steuert (AutoKI) und vollkommen ohne mögliches Eingreifen eines Menschen im Straßenverkehr agiert.",
        DiscussionSzenario.REFAI: "Würden Sie eine solche Schiri-KI begrüßen?",
    }

    if discussion_scenario not in descriptions:
        raise ValueError(f"No question text available for scenario {discussion_scenario}!")
    else:
        return descriptions[discussion_scenario]


def get_scenario_description(discussion_scenario: DiscussionSzenario) -> str:
    descriptions = {
        DiscussionSzenario.MEDAI: "Wir gehen von einem zukünftigen Szenario aus, indem eine medizinische KI (medKI) in Untersuchungszentren eigenständig Diagnosen stellt und therapeutische Entscheidungen trifft.\n            In den Untersuchungszentren arbeitet medizinisches Personal, das Daten erhebt und den Patient:innen die Entscheidungen der medKI erklären kann, aber keine Entscheidungen trifft.",
        DiscussionSzenario.JURAI: "Wir gehen von einem zukünftigen Szenario aus, in dem die KI als Richter:in grundsätzlich erlaubt und ein vom Staat bereitgestelltes, intelligentes (regel- und lernbasiert) Programm ist. Die Aufgabe der beteiligten Richter:innen beschränkt sich auf die Feststellung der Fakten, die KI würde den Prozess leiten und das Urteil fällen.",
        DiscussionSzenario.AUTOAI: "Wir gehen von einem zukünftigen Szenario aus, indem eine KI autonom ein Fahrzeug steuert (AutoKI) und vollkommen ohne mögliches Eingreifen eines Menschen im Straßenverkehr agiert.",
        DiscussionSzenario.REFAI: "Bei Profi-Fußballspielen werden alle Schiedsrichter-Entscheidungen von einer sehr guten KI (Künstlichen Intelligenz) auf Basis der Videos verschiedener Kameras im Stadion in Echtzeit getroffen, die umfangreich mit Videos früherer Spiele trainiert wurde.\n Die KI-Entscheidungen werden vom einem menschlichen Schiedsrichter auf dem Platz nur noch kommuniziert.",
    }

    if discussion_scenario not in descriptions:
        raise ValueError(f"No desription text available for scenario {discussion_scenario}!")
    else:
        return descriptions[discussion_scenario]


def get_welcome_message(user_name: str, user_stance: str, discussion_scenario: DiscussionSzenario) -> str:
    welcome_messages = {
        DiscussionSzenario.MEDAI: f"Deine Meinung ist, dass in unserem Szenario KI die Zulassung als Ärzt:innen {'' if user_stance == Stance.PRO else 'nicht'} erhalten sollte. Bitte nenne zunächst das wichtigste Argument für deine Meinung.",
        DiscussionSzenario.JURAI: f"Deine Meinung ist, dass in unserem Szenario KI Fälle in Zivilrechtsprozessen {'' if user_stance == Stance.PRO else 'nicht'} entscheiden sollte. Bitte nenne zunächst das wichtigste Argument für deine Meinung.",
        DiscussionSzenario.AUTOAI: f"Deine Meinung ist, dass in unserem Szenario die AutoKI für den Straßenverkehr {'' if user_stance == Stance.PRO else 'nicht'} zugelassen werden sollten. Bitte nenne zunächst das wichtigste Argument für deine Meinung.",
        DiscussionSzenario.REFAI: f"Deine Meinung ist, dass in unserem Szenario KI die Entscheidungen {'' if user_stance == Stance.PRO else 'nicht'} allein treffen sollte. Bitte nenne zunächst das wichtigste Argument für deine Meinung."
    }

    if discussion_scenario not in welcome_messages:
        raise ValueError(f"No welcome message text available for scenario {discussion_scenario}!")

    if user_stance == Stance.OTHER:
        raise ValueError(f"Cannot create a valid welcome messaage with Stance.OTHER!")

    return f"Hallo {user_name}, willkommen im Chat. {welcome_messages[discussion_scenario]}"


def load_response_template_collection(topic: str,
                                      project_root: str = "/home/christian/PycharmProjects/ethikchat-experiment-argument-classification",
                                      argument_graphs_dir: str = "data/external/argument_graphs/") -> ResponseTemplateCollection:
    argument_graph_directory = os.path.join(project_root, argument_graphs_dir, f"szenario_{topic}")

    return ResponseTemplateCollection.from_csv_files(
        templates_directory_path=argument_graph_directory,
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


def utterance_contains_noisy_data(utterance: Utterance, noisy_labels) -> Tuple[bool, str]:
    """
    Function that checks if certain utterances would introduce noisy data into the dataset.
    """
    if "aus der folgenden Liste" in utterance.text:
        print(f"Flagged utterance with 'aus der folgenden Liste' in text: {utterance.text} as noisy data.")
        return True, "aus der folgenden Liste"

    if "Ich bin mir nicht sicher, was du genau meinst." in utterance.text:
        print(f"Flagged utterance with 'bot uttering not understanding': {utterance.text} as noisy data.")
        return True, "Ich bin mir nicht sicher, was du genau meinst."

    ## remove duplicates from labels
    true_labels = list(set(utterance.true_labels))

    # check if utterance exclusively contains noisy labels
    if set(true_labels).issubset(set(noisy_labels)):
        print(f"Flagged utterance with noisy labels: {utterance.text}, {utterance.true_labels} as noisy data.")
        return True, "contains only noisy labels"

    return False, ""


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
        dialogue.name = dialogue_id
        dialogues.append(dialogue)
    return dialogues


def copy_start_survey_result_from_unprocessed_to_processed_dialogues(unprocessed_dialogues: List[Dialogue],
                                                                     processed_dialogues: List[Dialogue]) -> List[
    Dialogue]:
    """
    Processed dialogues that are stored in excel files do not contain a start survey result which holds the users initial
    stance. The stance is necessary for further processing of the dataset. The unprocessed data holds this information.
    This function loads the information from unprocessed into processed dialogues.
    """
    unprocessed_lookup = {dialogue.name: dialogue for dialogue in unprocessed_dialogues}

    for dialogue in processed_dialogues:
        if dialogue.name not in unprocessed_lookup:
            raise ValueError(f"Unprocessed dialogue {dialogue.name} not found.")

        dialogue.start_survey = unprocessed_lookup[dialogue.name].start_survey

    return processed_dialogues


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
                       utterance_type: UtteranceType,
                       noisy_labels: List[str]) -> Tuple[List[ProcessedUtterance], List[NoisyProcessedUtterance]]:
    """
    Preprocesses a dataset by applying text normalization and updating bounds.
    Returns a list of ProcessedUtterance objects.
    """

    processed_utterances = []
    excluded_noisy_utterances = []

    gender_language_tools = GenderLanguageTools()

    id_counter = 0
    for dialogue in dialogues:
        dialogue_turns = []
        user_name = dialogue.user_utterances[0].user
        user_stance = Stance.from_agreement_value(dialogue.start_survey.agreement)
        dialogue_turns.append(("Bot", get_welcome_message(user_name, user_stance, dialogue.discussion_szenario)))
        for utterance in dialogue.utterances:
            check_bounds_correctness(utterance, dialogue.name)

            dialogue_turns.append((
                "User" if utterance.is_from_user() else "Bot", preprocess_text(utterance.text, gender_language_tools)
            ))

            if utterance_type == UtteranceType.User and not utterance.is_from_user():
                continue

            if utterance_type == UtteranceType.Bot and utterance.is_from_user():
                continue

            processed_utterance_text, processed_labels, processed_bounds = preprocess_utterance(utterance,
                                                                                                gender_language_tools)

            ucnd, reason = utterance_contains_noisy_data(utterance, noisy_labels)
            if ucnd:
                excluded_noisy_utterances.append(NoisyProcessedUtterance(
                    id=id_counter,
                    text=processed_utterance_text,
                    labels=processed_labels,
                    bounds=processed_bounds,
                    context=dialogue_turns,
                    discussion_scenario=dialogue.discussion_szenario,
                    scenario_description=preprocess_text(get_scenario_description(dialogue.discussion_szenario),
                                                         gender_language_tools),
                    scenario_question=preprocess_text(get_scenario_question(dialogue.discussion_szenario),
                                                      gender_language_tools),
                    reason=reason
                ))
            else:
                processed_utterances.append(ProcessedUtterance(
                    id=id_counter,
                    text=processed_utterance_text,
                    labels=processed_labels,
                    bounds=processed_bounds,
                    context=dialogue_turns,
                    discussion_scenario=dialogue.discussion_szenario,
                    scenario_description=preprocess_text(get_scenario_description(dialogue.discussion_szenario),
                                                         gender_language_tools),
                    scenario_question=preprocess_text(get_scenario_question(dialogue.discussion_szenario),
                                                      gender_language_tools)
                ))

            id_counter += 1

    return processed_utterances, excluded_noisy_utterances


def create_queries(processed_utterances: List[ProcessedUtterance], excluded_labels: List[str]) -> List[Query]:
    """
    Creates the queries from processed_utterances. Ensures that there are no duplicate queries (
    """
    queries = []
    for processed_utterance in processed_utterances:
        filtered_labels = [label for label in processed_utterance.labels if label not in excluded_labels]
        queries.append(Query(processed_utterance.id,
                             processed_utterance.text,
                             filtered_labels,
                             processed_utterance.discussion_scenario,
                             processed_utterance.context,
                             processed_utterance.scenario_description,
                             processed_utterance.scenario_question
                             ))
    # check for duplicates
    unique_queries = []
    for query in queries:
        if query in unique_queries:
            print(f"Duplicate query found: {query}. Will not be added to the dataset.")
        else:
            unique_queries.append(query)

    return unique_queries


def create_queries_relevant_passages_mapping_split(queries: List[Query], passages: List[Passage]) -> Dict[
    int, List[int]]:
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


def create_passages_from_utterances(processed_utterances: List[ProcessedUtterance], excluded_labels: List[str]) -> List[
    Passage]:
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
                                        passage_source=PassageSource.UserUtterance.value,
                                        retrieved_query_id=pu.id))
    return passages


def create_passages_from_argument_graph(argument_graph: ResponseTemplateCollection,
                                        discussion_scenario: DiscussionSzenario,
                                        excluded_labels: List[str]) -> List[Passage]:
    """
    Creates passages from the argument graph.
    """
    passages = []

    templates = argument_graph.arguments_templates + argument_graph.faq_question_templates
    templates = list(filter(lambda x: x.label not in excluded_labels, templates))

    for template in templates:
        passages.append(Passage(None, template.summary, template.label, discussion_scenario.value,
                                PassageSource.ArgumentgraphSummary.value, None))
        passages.append(Passage(None, template.full_text, template.label, discussion_scenario.value,
                                PassageSource.ArgumentgraphFullText.value, None))
        passages.extend([Passage(None, sample, template.label, discussion_scenario.value,
                                 PassageSource.ArgumentgraphSample.value, None) for sample in template.samples])
    return passages


def create_queries_trivial_passages_mapping_split(queries: List[Query], passages: List[Passage]) -> Dict[
    int, List[int]]:
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


def check_for_missing_passages(queries_relevant_passages_mapping: Dict[int, List[int]]) -> None:
    """
    Checks for missing passages in the queries_relevant_passages_mapping split.
    """
    for query_id, relevant_passages in queries_relevant_passages_mapping.items():
        if len(relevant_passages) == 0:
            raise ValueError(f"Query {query_id} has no relevant passages.")


def create_dataset_splits(dialogues: List[Dialogue],
                          utterance_type: UtteranceType,
                          argument_graphs: Dict[DiscussionSzenario, ResponseTemplateCollection],
                          noisy_labels: List[str]) \
        -> Tuple[List[Query], List[Passage], Dict[int, List[int]], Dict[int, List[int]], List[Tuple[Query, str]]]:
    """
    Creates the dataset splits for the information retrieval task. This consists of the following splits:
    - queries: A split containing the queries.
    - passages: A split containing the passages.
    - queries_relevant_passages_mapping: A mapping from query ids to relevant passage ids.
    - queries_trivial_passages_mapping: A mapping from query ids to trivial passage ids. (Passages that are part of the query itself)
    - noisy_queries: Queries containing no text that can be mapped to any part of the argument graph
    """

    processed_utterances, noisy_processed_utterances = preprocess_dataset(
        dialogues,
        utterance_type,
        noisy_labels
    )

    utterances_passages = create_passages_from_utterances(processed_utterances, noisy_labels)

    argument_graphs_passages = []
    for discussion_scenario, argument_graph in argument_graphs.items():
        argument_graphs_passages.extend(
            create_passages_from_argument_graph(argument_graph, discussion_scenario, noisy_labels))

    # im queries split hat jede query_id die reihenfolge der processed_utterances.
    queries = create_queries(processed_utterances, noisy_labels)
    noisy_queries = create_queries(noisy_processed_utterances, [])
    noisy_queries = list(zip(noisy_queries, [noisy_processed_utterance.reason for noisy_processed_utterance in noisy_processed_utterances]))

    # merge passages and assign ids
    passages = [Passage(idx, passage.text, passage.label, passage.discussion_scenario, passage.passage_source,
                        passage.retrieved_query_id) for idx, passage in
                enumerate(utterances_passages + argument_graphs_passages)]

    queries_relevant_passages_mapping = create_queries_relevant_passages_mapping_split(queries, passages)
    queries_trivial_passages_mapping = create_queries_trivial_passages_mapping_split(queries, passages)

    check_for_missing_passages(queries_relevant_passages_mapping)
    check_for_missing_passages(queries_trivial_passages_mapping)

    return queries, passages, queries_relevant_passages_mapping, queries_trivial_passages_mapping, noisy_queries


def create_dataset(config: DatasetConfig) -> None:
    save_path = config.dataset_path
    project_dir = config.project_dir
    utterance_type = config.utterance_type

    dialogues_autoai, dialogues_jurai, dialogues_medai, dialogues_refai = load_dialogues(project_dir)

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

    queries, passages, queries_relevant_passages_mapping, queries_trivial_passages_mapping, excluded_queries = create_dataset_splits(
        all_dialogues, utterance_type, argument_graphs, config.noisy_labels)

    # create hf dataset
    corpus_dataset = DatasetDict({
        "queries": Dataset.from_dict({"id": [query.id for query in queries],
                                      "text": [query.text for query in queries],
                                      "labels": [query.labels for query in queries],
                                      "discussion_scenario": [query.discussion_scenario for query in queries],
                                      "context": [query.context for query in queries],
                                      "scenario_description": [query.scenario_description for query in queries],
                                      "scenario_question": [query.scenario_question for query in queries]
                                      }),
        "passages": Dataset.from_dict({"id": [passage.id for passage in passages],
                                       "text": [passage.text for passage in passages],
                                       "label": [passage.label for passage in passages],
                                       "discussion_scenario": [passage.discussion_scenario for passage in passages],
                                       "passage_source": [passage.passage_source for passage in passages]
                                       }),
        "queries_relevant_passages_mapping": Dataset.from_dict({
            "query_id": [idx for idx, _ in queries_relevant_passages_mapping.items()],
            "passages_ids": [ids for _, ids in queries_relevant_passages_mapping.items()]
        }),
        "queries_trivial_passages_mapping": Dataset.from_dict({
            "query_id": [idx for idx, _ in queries_trivial_passages_mapping.items()],
            "passages_ids": [ids for _, ids in queries_trivial_passages_mapping.items()]
        }),
        "excluded_utterances": Dataset.from_dict({"id": [query.id for (query, reason) in excluded_queries],
                                                  "text": [query.text for (query, reason) in excluded_queries],
                                                  "labels": [query.labels for (query, reason) in excluded_queries],
                                                  "discussion_scenario": [query.discussion_scenario for (query, reason)
                                                                          in excluded_queries],
                                                  "context": [query.context for (query, reason) in excluded_queries],
                                                  "scenario_description": [query.scenario_description for
                                                                           (query, reason) in excluded_queries],
                                                  "scenario_question": [query.scenario_question for (query, reason) in
                                                                        excluded_queries],
                                                  "reason": [reason for (query, reason) in excluded_queries]
                                                  })
    })

    corpus_dataset.save_to_disk(save_path)


def load_dialogues(project_dir) -> Tuple[List[Dialogue], List[Dialogue], List[Dialogue], List[Dialogue]]:
    def load_and_merge_dialogues(processed_path: str, raw_base_path: str, filename: str,
                                 szenario: DiscussionSzenario, raw_subdir: str = None,
                                 version: str = "9", copy_start_survey: bool = True) -> List[Dialogue]:
        dialogues = load_dataset_from_excel_file(
            os.path.join(processed_path, filename),
            szenario
        )
        if copy_start_survey and raw_subdir:
            unprocessed = DialogueLoader.from_directory(
                dialogues_directory_path=os.path.join(raw_base_path, raw_subdir),
                version=version
            )
            dialogues = copy_start_survey_result_from_unprocessed_to_processed_dialogues(unprocessed, dialogues)
        return dialogues

    path_m1_curated = os.path.join(project_dir, "data", "external", "ethikchat_data-main", "mensateria_survey",
                                   "processed", "curated")
    path_m1_raw = os.path.join(project_dir, "data", "external", "ethikchat_data-main", "mensateria_survey", "raw")

    path_m2_curated = os.path.join(project_dir, "data", "external", "ethikchat_data-main", "mensateria_survey_2",
                                   "processed", "curated")
    path_m2_raw = os.path.join(project_dir, "data", "external", "ethikchat_data-main", "mensateria_survey_2", "raw")

    m1_dialogues_medai = load_and_merge_dialogues(path_m1_curated, path_m1_raw,
                                                  "mensateria_survey_medai_curated_dialogues.xlsx",
                                                  DiscussionSzenario.MEDAI, raw_subdir="medai", version="8")
    m1_dialogues_jurai = load_and_merge_dialogues(path_m1_curated, path_m1_raw,
                                                  "mensateria_survey_jurai_curated_dialogues.xlsx",
                                                  DiscussionSzenario.JURAI, raw_subdir="jurai", version="8")
    m1_dialogues_autoai = load_and_merge_dialogues(path_m1_curated, path_m1_raw,
                                                   "mensateria_survey_autoai_curated_dialogues.xlsx",
                                                   DiscussionSzenario.AUTOAI, raw_subdir="autoai", version="8")

    m2_dialogues_medai = load_and_merge_dialogues(path_m2_curated, path_m2_raw,
                                                  "mensateria_survey_2_medai_curated_dialogues.xlsx",
                                                  DiscussionSzenario.MEDAI, raw_subdir="medai")
    m2_dialogues_jurai = load_and_merge_dialogues(path_m2_curated, path_m2_raw,
                                                  "mensateria_survey_2_jurai_curated_dialogues.xlsx",
                                                  DiscussionSzenario.JURAI, raw_subdir="jurai")
    m2_dialogues_autoai = load_and_merge_dialogues(path_m2_curated, path_m2_raw,
                                                   "mensateria_survey_2_autoai_curated_dialogues.xlsx",
                                                   DiscussionSzenario.AUTOAI, raw_subdir="autoai")
    m2_dialogues_refai = load_and_merge_dialogues(path_m2_curated, path_m2_raw,
                                                  "mensateria_survey_2_refai_curated_dialogues.xlsx",
                                                  DiscussionSzenario.REFAI, raw_subdir="refai")

    m3_dialogues_medai = DialogueLoader.from_directory(
        dialogues_directory_path=os.path.join(project_dir, "data", "external", "ethikchat_data-main",
                                              "mensateria_survey_3", "processed", "medai"),
        version="webathen"
    )
    for d in m3_dialogues_medai:
        d.discussion_szenario = DiscussionSzenario.MEDAI

    dialogues_medai = m1_dialogues_medai + m2_dialogues_medai + m3_dialogues_medai
    dialogues_jurai = m1_dialogues_jurai + m2_dialogues_jurai
    dialogues_autoai = m1_dialogues_autoai + m2_dialogues_autoai
    dialogues_refai = m2_dialogues_refai

    return dialogues_autoai, dialogues_jurai, dialogues_medai, dialogues_refai


if __name__ == "__main__":

    # load dataset
    dataset_folder = "../../data/processed/"
    dataset_path = os.path.join(dataset_folder, "corpus_dataset_with_context_v1")

    if not os.path.exists(dataset_path):
        # Beispiel zum Erstellen eines Datensatzes. Mögliche Optionen von DatasetConfig sind im DocString beschrieben.
        create_dataset(
            DatasetConfig(
                dataset_path=dataset_path,
                project_dir="../../",
                utterance_type=UtteranceType.User,
                eval_size=0.5,
                validation_test_ratio=0.5
            )
        )

    # Beispiel zum Laden des Datensatzes + collate_function des DataLoaders um dynamisch ein Subset der negative passages zu laden.
    hf_dataset = load_from_disk(os.path.join(dataset_folder, "corpus_dataset_v1"))
    hf_dataset_with_context = load_from_disk(os.path.join(dataset_folder, "corpus_dataset_with_context_v1"))
    print()

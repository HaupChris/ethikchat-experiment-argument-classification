import ast
import os
import warnings

import pandas as pd
import re

from dataclasses import asdict, fields
from datasets import DatasetDict, Dataset, load_from_disk
from datetime import datetime
from typing import List, Tuple, Dict, Sequence, Any, Set

from ethikchat_argtoolkit.ArgumentGraph.response_template_collection import ResponseTemplateCollection
from ethikchat_argtoolkit.ArgumentGraph.stance import Stance

from ethikchat_argtoolkit.Dialogue.utterance import Utterance
from ethikchat_argtoolkit.Dialogue.discussion_szenario import DiscussionSzenario
from ethikchat_argtoolkit.Loading.dialogue_loader import DialogueLoader
from ethikchat_argtoolkit.Preprocessing.gender_language_tools import GenderLanguageTools
from ethikchat_argtoolkit.Dialogue.dialogue import Dialogue, UserUtterance, BotUtterance
from ethikchat_argtoolkit.Dialogue.dialogue_szenario import DialogueSzenario

from src.data.classes import UtteranceType, ProcessedUtterance, NoisyProcessedUtterance, PassageSource, Passage, Query, \
    NoisyQuery, DatasetConfig


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


def utterance_contains_noisy_data(utterance: Utterance, noisy_labels: Set[str]) -> Tuple[bool, str]:
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
                       noisy_labels: Set[str]) -> Tuple[List[ProcessedUtterance], List[NoisyProcessedUtterance]]:
    """
    Preprocesses a dataset by applying text normalization and updating bounds.
    Returns a list of ProcessedUtterance objects.
    """

    processed_utterances = []
    excluded_noisy_utterances = []
    gender_language_tools = GenderLanguageTools()
    id_counter = 0

    for dialogue in dialogues:
        user_stance = Stance.from_agreement_value(dialogue.start_survey.agreement)
        dialogue_turns = [("Bot", get_welcome_message(dialogue.user_utterances[0].user,
                                                      user_stance,
                                                      dialogue.discussion_szenario))]
        context_labels = [["Greeting_Message"]]

        for utterance in dialogue.utterances:
            check_bounds_correctness(utterance, dialogue.name)
            speaker = "User" if utterance.is_from_user() else "Bot"
            text = preprocess_text(utterance.text, gender_language_tools)

            dialogue_turns.append((speaker, text))
            context_labels.append(utterance.true_labels)


            # filter by type
            if utterance_type == UtteranceType.User and not utterance.is_from_user():
                continue

            if utterance_type == UtteranceType.Bot and utterance.is_from_user():
                continue

            proc_utterance_text, proc_labels, proc_bounds = preprocess_utterance(utterance, gender_language_tools)

            # make a shallow copy of dialogue context so not the context in the processed utterances is not a reference
            # to the whole context lists
            current_context = dialogue_turns.copy()
            current_context_labels = context_labels.copy()
            current_context.pop()
            current_context_labels.pop()

            noisy_flag, reason = utterance_contains_noisy_data(utterance, noisy_labels)

            common_kwargs = dict(
                id=id_counter,
                text=proc_utterance_text,
                labels=proc_labels,
                bounds=proc_bounds,
                context=current_context,
                discussion_scenario=dialogue.discussion_szenario,
                scenario_description=preprocess_text(
                    get_scenario_description(dialogue.discussion_szenario),
                    gender_language_tools),
                scenario_question=preprocess_text(
                    get_scenario_question(dialogue.discussion_szenario),
                    gender_language_tools),
                user_stance=user_stance,
                context_labels=current_context_labels,
                original_dialogue_id=dialogue.name
            )

            if noisy_flag:
                excluded_noisy_utterances.append(
                    NoisyProcessedUtterance(
                        **common_kwargs,
                        reason=reason
                    )
                )
            else:
                processed_utterances.append(
                    ProcessedUtterance(**common_kwargs)
                )

            id_counter += 1

    return processed_utterances, excluded_noisy_utterances


def create_queries(processed_utterances: List[ProcessedUtterance], excluded_labels: Set[str]) \
        -> Tuple[List[Query], List[int]]:
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
                             processed_utterance.scenario_question,
                             processed_utterance.user_stance,
                             processed_utterance.context_labels,
                             processed_utterance.original_dialogue_id
                             ))
    # check for duplicates
    unique_queries = []
    duplicate_query_ids = []
    for query in queries:
        if query in unique_queries:
            warnings.warn(f"Duplicate query found: {query.id} {query.text}. Will not be added to the dataset.")
            duplicate_query_ids.append(query.id)
        else:
            unique_queries.append(query)

    return unique_queries, duplicate_query_ids


def create_noisy_queries(noisy_processed_utterances: List[NoisyProcessedUtterance]) -> List[NoisyQuery]:
    """
    Creates the noisy queries from noisy processed_utterances. Ensures that there are no duplicates

    """
    queries, _ = create_queries(noisy_processed_utterances, [])
    noisy_queries = [
        NoisyQuery(
            reason=npu.reason,
            **asdict(query))
        for query, npu in list(zip(queries, noisy_processed_utterances))
    ]

    return noisy_queries


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


def create_passages_from_utterances(processed_utterances: List[ProcessedUtterance], excluded_labels: Set[str]) -> List[
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
                                        excluded_labels: Set[str]) -> List[Passage]:
    """
    Creates passages from the argument graph.
    """
    passages = []

    templates = argument_graph.user_intent_templates
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
                          noisy_labels: Set[str]) \
        -> Tuple[List[Query],
        List[Passage],
        Dict[int, List[int]],
        Dict[int, List[int]],
        List[NoisyQuery]]:
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
    queries, duplicate_queries_ids = create_queries(processed_utterances, noisy_labels)
    noisy_queries = create_noisy_queries(noisy_processed_utterances)

    # merge passages and assign ids
    passages = [Passage(idx, passage.text, passage.label, passage.discussion_scenario, passage.passage_source,
                        passage.retrieved_query_id) for idx, passage in
                enumerate(utterances_passages + argument_graphs_passages)
                if passage.retrieved_query_id not in duplicate_queries_ids]

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

    argument_graph_med = load_response_template_collection("s1", argument_graphs_dir="data/external/argument_graphs/")
    argument_graph_jur = load_response_template_collection("s2", argument_graphs_dir="data/external/argument_graphs/")
    argument_graph_auto = load_response_template_collection("s3", argument_graphs_dir="data/external/argument_graphs/")
    argument_graph_ref = load_response_template_collection("s4", argument_graphs_dir="data/external/argument_graphs/")

    argument_graphs = {
        DiscussionSzenario.MEDAI: argument_graph_med,
        DiscussionSzenario.JURAI: argument_graph_jur,
        DiscussionSzenario.AUTOAI: argument_graph_auto,
        DiscussionSzenario.REFAI: argument_graph_ref
    }

    all_possible_labels = [utterance.true_labels for dialogue in all_dialogues for utterance in dialogue.utterances]
    all_possible_labels = set([label for label_list in all_possible_labels for label in label_list])
    good_labels = set()
    for arg_graph in argument_graphs.values():
        good_labels.update(arg_graph.user_intent_labels)

    noisy_labels = all_possible_labels.difference(good_labels)

    queries, passages, queries_relevant_passages_mapping, queries_trivial_passages_mapping, noisy_queries = create_dataset_splits(
        all_dialogues, utterance_type, argument_graphs, noisy_labels)

    def build_ds(objs: Sequence[Any], attrs: Sequence[str]) -> Dataset:
        """Turn a sequence of objects into an HF Dataset by pulling out each attr."""
        return Dataset.from_dict({
            attr: [getattr(o, attr) for o in objs]
            for attr in attrs
        })

    # specify once which fields each split needs
    _query_fields = [f.name for f in fields(Query)]
    _passage_fields = [f.name for f in fields(Passage)]

    _noisy_query_fields = [f.name for f in fields(NoisyQuery)]

    corpus_dataset = DatasetDict({
        "queries": build_ds(queries, _query_fields),
        "passages": build_ds(passages, _passage_fields),
        "queries_relevant_passages_mapping": Dataset.from_dict({
            "query_id": list(queries_relevant_passages_mapping.keys()),
            "passages_ids": list(queries_relevant_passages_mapping.values())
        }),
        "queries_trivial_passages_mapping": Dataset.from_dict({
            "query_id": list(queries_trivial_passages_mapping.keys()),
            "passages_ids": list(queries_trivial_passages_mapping.values())
        }),
        # for noisy you need to pull the “reason” out of the tuple
        "noisy_queries": build_ds(noisy_queries, _noisy_query_fields)
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
    dataset_folder = "../../data/processed/with_context"
    dataset_path = os.path.join(dataset_folder, "corpus_dataset_v3")

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
    hf_dataset = load_from_disk(os.path.join(dataset_folder, "corpus_dataset_v3"))
    hf_dataset_with_context = load_from_disk(os.path.join(dataset_folder, "corpus_dataset_v2"))
    print()

import re
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Set
from src.data.create_corpus_dataset import load_dialogues



class DialogueStatistics:
    """Class for calculating comprehensive dialogue statistics."""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.all_dialogues = None
        self.df = None
        self._load_data()

    def _load_data(self):
        """Load all dialogue data."""
        dialogues_autoai, dialogues_jurai, dialogues_medai, dialogues_refai = load_dialogues(self.project_dir)
        self.all_dialogues = dialogues_medai + dialogues_autoai + dialogues_jurai + dialogues_refai
        self._create_dataframe()

    def _create_dataframe(self):
        """Create DataFrame from dialogue data."""
        columns = ["study", "bot_type", "dialogue", "topic", "utterance_id",
                   "is_user_utterance", "text", "labels", "boundaries"]
        data = []

        for dialogue in self.all_dialogues:
            study = dialogue.name[:6]
            bot_type = dialogue.bot_utterances[0].user if dialogue.bot_utterances else "unknown"

            for utterance in dialogue.utterances:
                data.append([
                    study, bot_type, dialogue.name, dialogue.discussion_szenario.value,
                    utterance.id_in_dialogue, utterance.is_from_user(),
                    utterance.text, utterance.true_labels, utterance.true_bounds
                ])

        self.df = pd.DataFrame(data=data, columns=columns)

    def get_label_categories(self) -> Dict[str, Set[str]]:
        """Extract and categorize all labels."""
        all_labels = defaultdict(set)

        for dialogue in self.all_dialogues:
            for utterance in dialogue.utterances:
                for label in utterance.true_labels:
                    if "FAQ" in label:
                        category = "FAQ"
                    elif any(x in label for x in ["NZ.K", "NZ.P", "CON_NZARG", "PRO_NZARG", "NZ.G"]):
                        category = "UF"
                    elif any(x in label for x in ["Z.K", "Z.P", "Z.C", "PRO_ZARG", "CON_ZARG", "Z.G"]):
                        category = "WF"
                    else:
                        category = "Misc"
                    all_labels[category].add(label)

        return dict(all_labels)

    def extract_distinct_arguments(self, labels: List[str]) -> List[str]:
        """Extract distinct arguments from labels."""
        if isinstance(labels, str):
            labels = eval(labels)

        arguments = [label for label in labels if re.match(r'^(Z\.|NZ\.|FAQ\.)(?!G)', label)]
        return list(set(arguments))

    def calculate_statistics(self, study: str = None, topic: str = None,
                             bot_type: str = None) -> Dict:
        """Calculate comprehensive statistics for given filters."""
        df = self.df.copy()

        # Apply filters
        if study:
            df = df[df["study"] == study]
        if topic:
            df = df[df["topic"] == topic]
        if bot_type:
            df = df[df["bot_type"] == bot_type]

        # Basic counts
        total_dialogues = df['dialogue'].nunique()
        total_user_utterances = df[df['is_user_utterance'] == True].shape[0]
        total_bot_utterances = df[df['is_user_utterance'] == False].shape[0]

        # Word statistics for user utterances
        user_utterances = df[df['is_user_utterance'] == True]['text']
        word_counts = user_utterances.str.split().str.len()
        avg_words = word_counts.mean() if not word_counts.empty else 0

        # Utterances per dialogue
        utterances_per_dialogue = df.groupby('dialogue').size()
        user_utterances_per_dialogue = df[df['is_user_utterance'] == True].groupby('dialogue').size()
        bot_utterances_per_dialogue = df[df['is_user_utterance'] == False].groupby('dialogue').size()

        avg_utterances_total = utterances_per_dialogue.mean() if not utterances_per_dialogue.empty else 0
        avg_utterances_user = user_utterances_per_dialogue.mean() if not user_utterances_per_dialogue.empty else 0
        avg_utterances_bot = bot_utterances_per_dialogue.mean() if not bot_utterances_per_dialogue.empty else 0

        # Distinct arguments per dialogue
        df['distinct_arguments_list'] = df['labels'].apply(self.extract_distinct_arguments)

        def union_of_arguments(dialogue_group):
            all_arguments = set()
            for args_list in dialogue_group['distinct_arguments_list']:
                all_arguments = all_arguments.union(set(args_list))
            return len(all_arguments)

        if not df.empty:
            distinct_args_per_dialogue = df.groupby('dialogue', group_keys=False).apply(
                union_of_arguments, include_groups=False)
            distinct_args_per_dialogue_by_user = df[df['is_user_utterance'] == True].groupby(
                'dialogue', group_keys=False).apply(union_of_arguments, include_groups=False)
            distinct_args_per_dialogue_by_bot = df[df['is_user_utterance'] == False].groupby(
                'dialogue', group_keys=False).apply(union_of_arguments, include_groups=False)

            avg_distinct_args = distinct_args_per_dialogue.mean()
            avg_distinct_args_user = distinct_args_per_dialogue_by_user.mean()
            avg_distinct_args_bot = distinct_args_per_dialogue_by_bot.mean()
        else:
            avg_distinct_args = avg_distinct_args_user = avg_distinct_args_bot = 0

        # Label distribution
        label_categories = self.get_label_categories()
        user_df = df[df['is_user_utterance'] == True].copy()

        def contains_label(labels, label_set):
            if isinstance(labels, str):
                labels = eval(labels)
            return any(label in label_set for label in labels)

        # Calculate label percentages
        if not user_df.empty:
            all_labels_set = label_categories.get("WF", set()).union(label_categories.get("UF", set())).union(label_categories.get("FAQ", set()))

            user_df.loc[:, 'contains_wf'] = user_df['labels'].apply(
                lambda labels: 1 if contains_label(labels, label_categories.get("WF", set())) else 0)
            user_df.loc[:, 'contains_uf'] = user_df['labels'].apply(
                lambda labels: 1 if contains_label(labels, label_categories.get("UF", set())) else 0)
            user_df.loc[:, 'contains_faq'] = user_df['labels'].apply(
                lambda labels: 1 if contains_label(labels, label_categories.get("FAQ", set())) else 0)
            user_df.loc[:, 'contains_other'] = user_df['labels'].apply(
                lambda labels: 0 if contains_label(labels, all_labels_set) else 1)

            perc_wf = user_df['contains_wf'].sum() / total_user_utterances if total_user_utterances > 0 else 0
            perc_uf = user_df['contains_uf'].sum() / total_user_utterances if total_user_utterances > 0 else 0
            perc_faq = user_df['contains_faq'].sum() / total_user_utterances if total_user_utterances > 0 else 0
            perc_other = user_df['contains_other'].sum() / total_user_utterances if total_user_utterances > 0 else 0
        else:
            perc_wf = perc_uf = perc_faq = perc_other = 0

        return {
            'total_dialogues': total_dialogues,
            'total_user_utterances': total_user_utterances,
            'total_bot_utterances': total_bot_utterances,
            'avg_words': avg_words,
            'avg_utterances_total': avg_utterances_total,
            'avg_utterances_user': avg_utterances_user,
            'avg_utterances_bot': avg_utterances_bot,
            'avg_distinct_args': avg_distinct_args,
            'avg_distinct_args_user': avg_distinct_args_user,
            'avg_distinct_args_bot': avg_distinct_args_bot,
            'perc_wf': perc_wf,
            'perc_uf': perc_uf,
            'perc_faq': perc_faq,
            'perc_other': perc_other
        }

    def generate_latex_row(self, study: str, topic: str, bot_type: str = None,
                           is_first_in_study: bool = False, study_row_count: int = 1) -> str:
        """Generate LaTeX table row for given parameters."""
        stats = self.calculate_statistics(study=study, topic=topic, bot_type=bot_type)

        if is_first_in_study:
            study_cell = f"\\multirow{{{study_row_count}}}{{*}}{{{study}}}"
        else:
            study_cell = ""

        return (f"{study_cell} & {topic} & {stats['total_dialogues']} & "
                f"{stats['total_user_utterances']} & {stats['avg_words']:.1f} & "
                f"{stats['avg_utterances_user']:.1f} & {stats['avg_distinct_args_user']:.1f} & "
                f"{stats['avg_distinct_args_bot']:.1f} & {stats['avg_distinct_args']:.1f} & "
                f"{stats['perc_wf']:.2f} & {stats['perc_uf']:.2f} & "
                f"{stats['perc_faq']:.2f} & {stats['perc_other']:.2f} \\\\")

    def generate_full_table(self) -> str:
        """Generate complete LaTeX table."""
        latex_rows = []

        # Study 1 (202301)
        topics_study1 = ["MEDAI", "JURAI", "AUTOAI"]
        for i, topic in enumerate(topics_study1):
            is_first = i == 0
            row = self.generate_latex_row("202301", topic, is_first_in_study=is_first,
                                          study_row_count=len(topics_study1))
            latex_rows.append(row)

        latex_rows.append("\\midrule")

        # Study 2 (202307)
        topics_study2 = ["MEDAI", "JURAI", "AUTOAI", "REFAI"]
        for i, topic in enumerate(topics_study2):
            is_first = i == 0
            row = self.generate_latex_row("202307", topic, is_first_in_study=is_first,
                                          study_row_count=len(topics_study2))
            latex_rows.append(row)

        latex_rows.append("\\midrule")

        # Study 3 (202402) - different bot types
        bot_types_study3 = ["EthiBot", "GenBot", "RAGBot"]
        for i, bot_type in enumerate(bot_types_study3):
            is_first = i == 0
            # For study 3, we might want to show MedAI topic or aggregate across topics
            stats = self.calculate_statistics(study="202402", bot_type=bot_type)

            if is_first:
                study_cell = f"\\multirow{{{len(bot_types_study3)}}}{{*}}{{3}}"
            else:
                study_cell = ""

            row = (f"{study_cell} & {bot_type} & {stats['total_dialogues']} & "
                   f"{stats['total_user_utterances']} & {stats['avg_words']:.1f} & "
                   f"{stats['avg_utterances_user']:.1f} & {stats['avg_distinct_args_user']:.1f} & "
                   f"{stats['avg_distinct_args_bot']:.1f} & {stats['avg_distinct_args']:.1f} & "
                   f"{stats['perc_wf']:.2f} & {stats['perc_uf']:.2f} & "
                   f"{stats['perc_faq']:.2f} & {stats['perc_other']:.2f} \\\\\\\\")
            latex_rows.append(row)

        latex_rows.append("\\midrule")

        # Total row for studies 1+2
        df_1_2 = self.df[self.df["study"].isin(["202301", "202307"])]
        df_total = self.df.copy()

        # Calculate stats for studies 1+2
        if not df_1_2.empty:
            # Basic counts
            studies_1_2_dialogues = df_1_2['dialogue'].nunique()
            studies_1_2_user_utterances = df_1_2[df_1_2['is_user_utterance'] == True].shape[0]

            # Word statistics
            user_utterances_1_2 = df_1_2[df_1_2['is_user_utterance'] == True]['text']
            word_counts_1_2 = user_utterances_1_2.str.split().str.len()
            avg_words_1_2 = word_counts_1_2.mean()

            # Utterances per dialogue
            user_utterances_per_dialogue_1_2 = df_1_2[df_1_2['is_user_utterance'] == True].groupby('dialogue').size()
            avg_utterances_user_1_2 = user_utterances_per_dialogue_1_2.mean()

            # Distinct arguments per dialogue for studies 1+2
            df_1_2['distinct_arguments_list'] = df_1_2['labels'].apply(self.extract_distinct_arguments)

            def union_of_arguments(dialogue_group):
                all_arguments = set()
                for args_list in dialogue_group['distinct_arguments_list']:
                    all_arguments = all_arguments.union(set(args_list))
                return len(all_arguments)

            distinct_args_per_dialogue_1_2 = df_1_2.groupby('dialogue', group_keys=False).apply(
                union_of_arguments, include_groups=False)
            distinct_args_per_dialogue_by_user_1_2 = df_1_2[df_1_2['is_user_utterance'] == True].groupby(
                'dialogue', group_keys=False).apply(union_of_arguments, include_groups=False)
            distinct_args_per_dialogue_by_bot_1_2 = df_1_2[df_1_2['is_user_utterance'] == False].groupby(
                'dialogue', group_keys=False).apply(union_of_arguments, include_groups=False)

            avg_distinct_args_1_2 = distinct_args_per_dialogue_1_2.mean()
            avg_distinct_args_user_1_2 = distinct_args_per_dialogue_by_user_1_2.mean()
            avg_distinct_args_bot_1_2 = distinct_args_per_dialogue_by_bot_1_2.mean()
        else:
            studies_1_2_dialogues = studies_1_2_user_utterances = 0
            avg_words_1_2 = avg_utterances_user_1_2 = 0
            avg_distinct_args_1_2 = avg_distinct_args_user_1_2 = avg_distinct_args_bot_1_2 = 0

        # Calculate stats for all studies (1+2+3)
        total_dialogues = df_total['dialogue'].nunique()
        total_user_utterances = df_total[df_total['is_user_utterance'] == True].shape[0]

        user_utterances_total = df_total[df_total['is_user_utterance'] == True]['text']
        word_counts_total = user_utterances_total.str.split().str.len()
        avg_words_total = word_counts_total.mean()

        user_utterances_per_dialogue_total = df_total[df_total['is_user_utterance'] == True].groupby('dialogue').size()
        avg_utterances_user_total = user_utterances_per_dialogue_total.mean()

        # Distinct arguments for all studies
        df_total['distinct_arguments_list'] = df_total['labels'].apply(self.extract_distinct_arguments)

        distinct_args_per_dialogue_total = df_total.groupby('dialogue', group_keys=False).apply(
            union_of_arguments, include_groups=False)
        distinct_args_per_dialogue_by_user_total = df_total[df_total['is_user_utterance'] == True].groupby(
            'dialogue', group_keys=False).apply(union_of_arguments, include_groups=False)
        distinct_args_per_dialogue_by_bot_total = df_total[df_total['is_user_utterance'] == False].groupby(
            'dialogue', group_keys=False).apply(union_of_arguments, include_groups=False)

        avg_distinct_args_total = distinct_args_per_dialogue_total.mean()
        avg_distinct_args_user_total = distinct_args_per_dialogue_by_user_total.mean()
        avg_distinct_args_bot_total = distinct_args_per_dialogue_by_bot_total.mean()

        # Calculate label distributions for studies 1+2
        label_categories = self.get_label_categories()
        user_df_1_2 = df_1_2[df_1_2['is_user_utterance'] == True].copy()

        def contains_label(labels, label_set):
            if isinstance(labels, str):
                labels = eval(labels)
            return any(label in label_set for label in labels)

        if not user_df_1_2.empty:
            all_labels_set_1_2 = label_categories.get("WF", set()).union(label_categories.get("UF", set())).union(
                label_categories.get("FAQ", set()))

            user_df_1_2.loc[:, 'contains_wf'] = user_df_1_2['labels'].apply(
                lambda labels: 1 if contains_label(labels, label_categories.get("WF", set())) else 0)
            user_df_1_2.loc[:, 'contains_uf'] = user_df_1_2['labels'].apply(
                lambda labels: 1 if contains_label(labels, label_categories.get("UF", set())) else 0)
            user_df_1_2.loc[:, 'contains_faq'] = user_df_1_2['labels'].apply(
                lambda labels: 1 if contains_label(labels, label_categories.get("FAQ", set())) else 0)
            user_df_1_2.loc[:, 'contains_other'] = user_df_1_2['labels'].apply(
                lambda labels: 0 if contains_label(labels, all_labels_set_1_2) else 1)

            perc_wf_1_2 = user_df_1_2[
                              'contains_wf'].sum() / studies_1_2_user_utterances if studies_1_2_user_utterances > 0 else 0
            perc_uf_1_2 = user_df_1_2[
                              'contains_uf'].sum() / studies_1_2_user_utterances if studies_1_2_user_utterances > 0 else 0
            perc_faq_1_2 = user_df_1_2[
                               'contains_faq'].sum() / studies_1_2_user_utterances if studies_1_2_user_utterances > 0 else 0
            perc_other_1_2 = user_df_1_2[
                                 'contains_other'].sum() / studies_1_2_user_utterances if studies_1_2_user_utterances > 0 else 0
        else:
            perc_wf_1_2 = perc_uf_1_2 = perc_faq_1_2 = perc_other_1_2 = 0

        # Calculate label distributions for all studies (1+2+3)
        user_df_total = df_total[df_total['is_user_utterance'] == True].copy()

        if not user_df_total.empty:
            all_labels_set_total = label_categories.get("WF", set()).union(label_categories.get("UF", set())).union(
                label_categories.get("FAQ", set()))

            user_df_total.loc[:, 'contains_wf'] = user_df_total['labels'].apply(
                lambda labels: 1 if contains_label(labels, label_categories.get("WF", set())) else 0)
            user_df_total.loc[:, 'contains_uf'] = user_df_total['labels'].apply(
                lambda labels: 1 if contains_label(labels, label_categories.get("UF", set())) else 0)
            user_df_total.loc[:, 'contains_faq'] = user_df_total['labels'].apply(
                lambda labels: 1 if contains_label(labels, label_categories.get("FAQ", set())) else 0)
            user_df_total.loc[:, 'contains_other'] = user_df_total['labels'].apply(
                lambda labels: 0 if contains_label(labels, all_labels_set_total) else 1)

            perc_wf_total = user_df_total[
                                'contains_wf'].sum() / total_user_utterances if total_user_utterances > 0 else 0
            perc_uf_total = user_df_total[
                                'contains_uf'].sum() / total_user_utterances if total_user_utterances > 0 else 0
            perc_faq_total = user_df_total[
                                 'contains_faq'].sum() / total_user_utterances if total_user_utterances > 0 else 0
            perc_other_total = user_df_total[
                                   'contains_other'].sum() / total_user_utterances if total_user_utterances > 0 else 0
        else:
            perc_wf_total = perc_uf_total = perc_faq_total = perc_other_total = 0

        # Generate the total rows
        total_1_2_row = (f"ADEA & Total & {studies_1_2_dialogues} & {studies_1_2_user_utterances} & "
                         f"{avg_words_1_2:.1f} & {avg_utterances_user_1_2:.1f} & {avg_distinct_args_user_1_2:.1f} & "
                         f"{avg_distinct_args_bot_1_2:.1f} & {avg_distinct_args_1_2:.1f} & "
                         f"{perc_wf_1_2:.2f} & {perc_uf_1_2:.2f} & {perc_faq_1_2:.2f} & {perc_other_1_2:.2f} \\\\")
        total_row = (f"1 + 2 + 3 & Total & {total_dialogues} & {total_user_utterances} & "
                     f"{avg_words_total:.1f} & {avg_utterances_user_total:.1f} & {avg_distinct_args_user_total:.1f} & "
                     f"{avg_distinct_args_bot_total:.1f} & {avg_distinct_args_total:.1f} & "
                     f"{perc_wf_total:.2f} & {perc_uf_total:.2f} & {perc_faq_total:.2f} & {perc_other_total:.2f} \\\\")
        latex_rows.append(total_1_2_row)
        latex_rows.append(total_row)

        return "\n".join(latex_rows)


def print_study_overview(stats_calculator: DialogueStatistics):
    """Print overview of studies and their compositions."""
    print("Study Overview:")
    print("=" * 50)

    studies = stats_calculator.df['study'].unique()
    for study in sorted(studies):
        study_data = stats_calculator.df[stats_calculator.df['study'] == study]
        topics = study_data['topic'].unique()
        bot_types = study_data['bot_type'].unique()

        print(f"\nStudy {study}:")
        print(f"  Topics: {', '.join(sorted(topics))}")
        print(f"  Bot types: {', '.join(sorted(bot_types))}")
        print(f"  Total dialogues: {study_data['dialogue'].nunique()}")
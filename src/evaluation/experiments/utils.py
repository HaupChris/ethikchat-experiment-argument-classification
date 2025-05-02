from typing import List
import pandas as pd



model_name_to_short_name = {
    "aari1995/German_Semantic_STS_V2": "GBERT Large-aari",
    "deutsche-telekom/gbert-large-paraphrase-euclidean": "GBERT Large-telekom",
    "T-Systems-onsite/cross-en-de-roberta-sentence-transformer": "XLMRoberta-EN-DE",
    "T-Systems-onsite/german-roberta-sentence-transformer-v2": "XLMRoberta-DE"
}


def pandas_df_to_latex(
        df: pd.DataFrame,
        use_tabularx: bool = True,
        columns: List[str] = None,
        column_format: str = None,
        rename_columns: dict = None,
        width: str = r"\textwidth",
        float_format: str = "{:.2f}".format,
) -> str:
    """
    Convert a pandas DataFrame to a LaTeX tabularx table with flexible options.

    Args:
        float_format ():
        columns ():
        df (pd.DataFrame): The DataFrame to convert.
        use_tabularx (bool): If True, use tabularx instead of tabular.
        column_format (str): Column format string (e.g., 'XXX' or 'lXr').
                             If None, auto-generate 'X' repeated.
        rename_columns (dict): Mapping to rename columns before export.
        width (str): Width for tabularx (default: \\textwidth).

    Returns:
        str: LaTeX code for the table.
    """
    df_copy: pd.DataFrame = df.copy(deep=True)

    # Rename columns if requested
    if rename_columns:
        df_copy.rename(columns=rename_columns, inplace=True)

    # Determine column format
    if column_format is None:
        num_cols = len(df_copy.columns) + int(df_copy.index.names[0] is not None)
        column_format = "C" * len(columns) if columns else "C" * num_cols
    # Generate LaTeX string
    latex_str = df_copy.to_latex(
        index=False,
        columns=columns,
        float_format=float_format if float_format else None,
        column_format=column_format,
        escape=False  # Allow LaTeX formatting in cell content if needed
    )

    # Replace tabular environment with tabularx if requested
    if use_tabularx:
        latex_str = latex_str.replace(r"\begin{tabular}", "\\begin{tabularx}{" + width + "}")
        latex_str = latex_str.replace(r"\end{tabular}", r"\end{tabularx}")

    return latex_str

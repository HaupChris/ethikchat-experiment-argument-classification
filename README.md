# Workflow for Conducting Experiments in the `de.uniwue.ethik-chat` Project

This workflow outlines the steps for conducting experiments in the `de.uniwue.ethik-chat` project, including setting up a virtual environment, installing required packages, and organizing experiment code.

## Prerequisites

To use this workflow, you will need the following software installed on your system:

- Python 3.8 or later
- Git
- Pip
- Virtualenv

You will also need access to your own GitLab repository, where you can host the `de.uniwue.ethik-chat.Data`, `de.uniwue.ethik-chat.NLP`, and `de.uniwue.ethik-chat.Dialogue` packages.

## Getting Started

To start conducting experiments in the `de.uniwue.ethik-chat` project, follow these steps:

1. Clone this repository to your local machine:
```bash
git clone https://gitlab2.informatik.uni-wuerzburg.de/de.uniwue.ethik-chat/de.uniwue.ethik-chat.Experiment
```

2. Create a new branch for your experiment:

```bash
git checkout -b your-branch-name
``` 

3. Create a new virtual environment for your experiment and activate it:

```bash 
virtualenv -p python3.9 venv
``` 

```bash
source venv/bin/activate
```

4. Install the required packages for your experiment by running the following command from the root directory of the `de.uniwue.ethik-chat.Experiment` repository:

```bash
pip install -e .
```

This command installs the `de.uniwue.ethik-chat.Data`, `de.uniwue.ethik-chat.NLP`, and `de.uniwue.ethik-chat.Dialogue` packages from your GitLab repository, as well as the `de.uniwue.ethik-chat.Experiment` package itself. The `-e` option tells pip to install the packages in "editable" mode, which means that any changes you make to the packages will be reflected in your environment immediately.

5. If you have a `requirements.txt` file for your experiment, you can install the required packages by running the following command from the root directory of the `de.uniwue.ethik-chat.Experiment` repository:

```bash
pip install -r requirements.txt
```

This will install the required packages listed in the `requirements.txt` file. You can use this method instead of Step 4 if you prefer not to install the packages in "editable" mode.




6. Put your data in the `data/` folder, Your code goes into the `src/` folder. You can use the `src/main.py` file as a starting point for your experiment code.
    `models/` is for storing your trained models. When you are done with your experiments, write a brief report in the `reports/` folder and commit your changes to your branch.



8. Run your experiment using the packages provided by `de.uniwue.ethik-chat.Data`, `de.uniwue.ethik-chat.NLP`, and `de.uniwue.ethik-chat.Dialogue`:


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


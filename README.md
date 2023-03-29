# Workflow for Conducting Experiments in the `de.uniwue.ethik-chat` Project

This repository is a template for conducting experiments in the `de.uniwue.ethik-chat` project. It contains a `Makefile` with commands for installing required packages, running experiments, and generating reports. It also contains a `requirements.txt` file for installing required packages, and a `src/main.py` file for running experiments.

This workflow outlines the steps for conducting experiments in the `de.uniwue.ethik-chat` project, including setting up a virtual environment, installing required packages, and organizing experiment code.


## Prerequisites

To use this workflow, you will need the following software installed on your system:

- Python 3.8 or later
- Git
- Pip
- Virtualenv


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

3. Use the `Makefile` to create a virtual environment, install required packages and download the data by running the following command in the root directory of the repository.Since the data is stored in a private gitlab repository, your gitlab account needs to have access to the repository.
If it has, you have to create a private access token and insert it into the file `.gitlab_personal_access_token` in the root directory of the repository.
Then run the following command in the root directory of the repository:
```bash
  make
```


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


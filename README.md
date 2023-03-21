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




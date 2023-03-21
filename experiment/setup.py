from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess


class InstallCommand(install):
    """Custom install command to clone repositories"""
    def run(self):
        # Clone eb.Data into data/eb.Data
        subprocess.check_call(['git', 'clone', 'https://gitlab2.informatik.uni-wuerzburg.de/de.uniwue.ethik-chat/de.uniwue.ethik-chat.Data', 'data/de.uniwue.ethik-chat.Data'])
        # Clone de.uniwue.ethik-chat.NLP into src/de.uniwue.ethik-chat.NLP
        subprocess.check_call(['git', 'clone', 'https://gitlab2.informatik.uni-wuerzburg.de/de.uniwue.ethik-chat/de.uniwue.ethik-chat.NLP', 'src/de.uniwue.ethik-chat.NLP'])
        # Clone de.uniwue.ethik-chat.Dialogue into src/de.uniwue.ethik-chat.Dialogue
        subprocess.check_call(['git', 'clone', 'https://gitlab2.informatik.uni-wuerzburg.de/de.uniwue.ethik-chat/de.uniwue.ethik-chat.Dialogue', 'src/de.uniwue.ethik-chat.Dialogue'])
        # Run standard install command
        install.run(self)

setup(
    name="de.uniwue.ethik-chat.Experiment",
    version="1.0",
    description="A package for conducting experiments in the de.uniwue.ethik-chat project",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "de.uniwue.ethik-chat.Data==latest",
        "de.uniwue.ethik-chat.NLP==latest",
        "de.uniwue.ethik-chat.Dialogue==latest"
    ],
    cmdclass={
        'install': InstallCommand,
    },
    data_files=[
        ("de.uniwue.ethik-chat/Data", "data/")
    ]
)

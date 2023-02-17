# ECE490
Repo to collaborate within for UIUC's ECE490, Spring 2023

## Installation
### Installation from PyPI
This should be as simple as:
```bash
python -m pip install uiuc-490-sp23
```
### Installation from Source
This project uses [Poetry](https://python-poetry.org/) Install it via
[their instructions](https://python-poetry.org/docs/#installing-with-the-official-installer)
and run:
```bash
poetry install
```
from this directory (preferably from within a `venv`).

## Executing
Once installed, individual assignments can be ran by:
```bash
python -m uiuc_490_sp23.assignment<n>
```
where `<n>` is the assignment number. These will have a CLI implemented using argparse; if you're not sure,
just run the above with `--help` to get a full list of commands.
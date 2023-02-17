# ECE490
Repo to collaborate within for UIUC's ECE490, Spring 2023

## Installation
This uses [Poetry](https://python-poetry.org/) for building/installation. Install it via their instructions and run:
```python
poetry install
```
from this directory (preferably from within a `venv`).

## Executing
Once installed, individual assignments can be ran by:
```python
python -m uiuc-490-sp23.assignment<n>
```
where `<n>` is the assignment number. These will have a CLI implemented using argparse; if you're not sure,
just run the above with `--help` to get a full list of commands.
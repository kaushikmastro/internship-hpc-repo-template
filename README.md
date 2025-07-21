# internship-repo-template

This repository template is a base repository for the CHM internship
projects. This is to fork only, don't use it to implement projects.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency
management. To install:

### On macOS:
1. Install pipx: `brew install pipx && pipx ensurepath`
2. Install Poetry: `pipx install poetry`
3. Install dependencies: `poetry install`
4. Activate virtual environment: `poetry shell`

### Other systems:
For installation on other systems, see [pipx installation docs](https://pipx.pypa.io/stable/installation/)
and [Poetry documentation](https://python-poetry.org/docs/).

## Run Project

Instructions for running the project should be added here after forking.

## Running Tests

Execute tests using pytest:
```bash
poetry run pytest
```

## Contributing

Please read [contributing.md](contributing.md) for guidelines on how to
contribute to this codebase.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE)
file.

## About Poetry

Poetry is a modern dependency management and packaging tool for Python. It
handles virtual environments, dependency resolution, and package publishing
automatically. The `pyproject.toml` file defines project metadata and
dependencies.

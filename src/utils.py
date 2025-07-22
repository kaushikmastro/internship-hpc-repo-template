"""Utility functions for configuration management and data processing.

This module provides helper functions for loading and processing configuration
files, particularly YAML files with Pydantic model validation.
"""

from typing import Type, TypeVar

import yaml
from pydantic import BaseModel

# Define a generic type variable that must inherit from BaseModel
T = TypeVar("T", bound=BaseModel)


def load_config_from_yaml(filepath: str, model: Type[T]) -> T:
    """Load and validate YAML configuration using a Pydantic model.

    Reads a YAML file from the specified filepath and creates an instance
    of the provided Pydantic model with the loaded data. This ensures
    type safety and validation of configuration files.

    Args:
        filepath: Path to the YAML configuration file
        model: Pydantic model class to validate and structure the data

    Returns:
        An instance of the provided model populated with YAML data

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is malformed or invalid
        pydantic.ValidationError: If the data doesn't match model schema

    Example:
        >>> from pydantic import BaseModel
        >>> class Config(BaseModel):
        ...     name: str
        ...     port: int
        >>> config = load_config_from_yaml('config.yaml', Config)
        >>> print(config.name, config.port)
    """
    with open(filepath, "r") as file:
        data = yaml.safe_load(file)

    # Handle empty YAML files where safe_load returns None
    if data is None:
        data = {}

    return model(**data)

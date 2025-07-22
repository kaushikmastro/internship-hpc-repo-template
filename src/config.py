"""Configuration classes for ModelAssistant and training.

This module defines Pydantic models for configuring the ModelAssistant
and training processes, supporting both supervised and inference modes.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field
from transformers import TrainingArguments


class ModelAssistantConfig(BaseModel):
    """Configuration for ModelAssistant initialization."""

    model_path: str = Field(
        default="gpt2",
        description="Path or name of the pre-trained model to load",
    )
    generation_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 50,
            "pad_token_id": 50256,  # GPT-2 EOS token
        },
        description="Parameters for text generation",
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    # Core training parameters
    dataset_path: str = Field(..., description="Path to the training dataset")
    output_dir: str = Field(
        ..., description="Directory to save model and training artifacts"
    )

    # Model configuration
    assistant_config: ModelAssistantConfig = Field(
        default_factory=ModelAssistantConfig,
        description="Configuration for the model assistant",
    )

    # Training arguments using Hugging Face TrainingArguments
    # Documentation:
    # https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
    # noqa: E501
    training_args: TrainingArguments = Field(
        default_factory=lambda: TrainingArguments(
            output_dir="./results",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            logging_dir="./logs",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=3,
            learning_rate=1e-5,
            # next three will ensure that the best model
            # regarding the eval_loss metric is saved
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
        description="Hugging Face TrainingArguments instance",
    )

    # Data processing parameters
    max_length: int = Field(
        default=512, description="Maximum sequence length for tokenization"
    )
    train_test_split: float = Field(
        default=0.8,
        description="Fraction of data to use for training (rest for eval)",
    )


class InferenceConfig(BaseModel):
    """Configuration for model inference."""

    model_path: str = Field(..., description="Path to the trained model")
    generation_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 100,
            "pad_token_id": 50256,
        },
        description="Parameters for text generation",
    )
    batch_size: int = Field(default=1, description="Batch size for inference")

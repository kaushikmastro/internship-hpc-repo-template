import torch
import pandas as pd
from transformer_lens import HookedTransformer
from huggingface_hub import login
from typing import List, Tuple
import torch.nn.functional as F
import numpy as np
import sys
import re
import random


class EmotionalMIPipeline:
    """
    A pipeline for zero-shot emotion classification using a Llama2 model,
    with advanced logit lens and causal analysis capabilities.
    """
    def __init__(self, model_name: str, hf_token: str, device: str = "cuda"):
        """
        Initializes the model, tokenizer, and device.
        """
        self.device = device
        self.emotion_labels = {
            'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3,
            'fear': 4, 'surprise': 5
        }
        self.id_to_emotion = {id: label for label, id in self.emotion_labels.items()}
        self.model = None
        self.tokenizer = None
        self._load_model(model_name, hf_token)

    def _load_model(self, model_name: str, hf_token: str):
        """Logs in to Hugging Face and loads the pre-trained model."""
        try:
            print("Logging into Hugging Face...")
            login(token=hf_token)
            print(f"Loading model: {model_name}...")
            self.model = HookedTransformer.from_pretrained(
                model_name,
                fold_ln=False,
                center_unembed=False,
                center_writing_weights=False,
                device=self.device
            )
            self.tokenizer = self.model.tokenizer
            print(f"Model and tokenizer loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            raise

    def _generate_prompt(self, text: str) -> str:
        """Generates a constraint-based prompt for emotion classification."""
        emotions_str = ', '.join(self.emotion_labels.keys())
        return (f"{text}\n"
                "What is the single emotion of this text? "
                f"You must choose one and only one from the following list: {emotions_str}. "
                "The emotion is:")

    def _classify_emotion_from_prompt(self, text: str) -> str:
        """Classifies the emotion of a given text using constraint-based prompting."""
        prompt = self._generate_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=10,
                do_sample=False,
            )

        predicted_tokens = self.tokenizer.decode(
            output_tokens[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )

        cleaned_output = predicted_tokens.strip().lower()
        for emotion in self.emotion_labels.keys():
            if emotion in cleaned_output.strip(".,:;").lower():
                return emotion
        return "unknown"

    def get_token_ids(self, text: str) -> List[int]:
        """Gets the token IDs for a text string, handling spaces."""
        try:
            encoded_with_space = self.tokenizer.encode(f" {text}", add_special_tokens=False)
            if encoded_with_space:
                return encoded_with_space
            encoded_without_space = self.tokenizer.encode(text, add_special_tokens=False)
            if encoded_without_space:
                return encoded_without_space
        except Exception as e:
            print(f"Tokenization error for '{text}': {e}", file=sys.stderr)
        raise ValueError(f"Could not find any tokens for '{text}'.")

    def get_rank(self, logits: torch.Tensor, token_ids: List[int]) -> int:
        """
        Calculates the rank of the given tokens in the logits tensor.
        """
        sorted_indices = torch.argsort(logits, descending=True)
        ranks = []
        for t in token_ids:
            if t in sorted_indices:
                rank = (sorted_indices == t).nonzero().item() + 1
                ranks.append(rank)
            else:
                return len(logits) + 1
        return min(ranks) if ranks else len(logits) + 1

    def calculate_prompt_ranks(self, prompt_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes a DataFrame of prompts by calculating the minimum rank for
        each prompt across all layers.
        """
        categorized_prompts = []
        for index, row in prompt_df.iterrows():
            try:
                true_emotion_text = row['emotion']
                prompt_text = row['constrained prompt']
                true_ids = self.get_token_ids(true_emotion_text)
                input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt')
                final_token_idx = input_ids.shape[-1] - 1
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(prompt_text)

                min_true_rank_across_layers = float('inf')
                min_rank_layer_idx = -1
                
                for layer_idx in range(self.model.cfg.n_layers):
                    mlp_out_contribution = cache[("mlp_out", layer_idx)][0, final_token_idx, :]
                    mlp_logits = self.model.unembed(mlp_out_contribution)
                    true_rank = self.get_rank(mlp_logits, true_ids)
                    
                    if true_rank < min_true_rank_across_layers:
                        min_true_rank_across_layers = true_rank
                        min_rank_layer_idx = layer_idx
                        
                categorized_prompts.append({
                    'prompt': prompt_text,
                    'rank last layer': self.get_rank(self.model.unembed(cache['resid_post', self.model.cfg.n_layers - 1][0, final_token_idx, :]), true_ids),
                    'min_true_rank': min_true_rank_across_layers,
                    'min_rank_layer': min_rank_layer_idx
                })
            except Exception as e:
                print(f"Skipping analysis for row {index}. Error: {e}", file=sys.stderr)
                continue

        if not categorized_prompts:
            return pd.DataFrame()
            
        return pd.DataFrame(categorized_prompts)

    def categorize_prompts(self, ranked_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the median of 'min_true_rank' as a threshold and
        categorizes the type of hallucination for each prompt.
        """
        if ranked_df.empty:
            print("Input DataFrame is empty. Cannot categorize prompts.", file=sys.stderr)
            return ranked_df

        try:
            knowledge_threshold = np.median(ranked_df['min_true_rank'])
        except KeyError as e:
            print(f"DataFrame is missing the required column: {e}", file=sys.stderr)
            return ranked_df

        ranked_df['threshold'] = knowledge_threshold
        ranked_df['hallucination type'] = ranked_df['min_true_rank'].apply(
            lambda min_rank: "Extraction" if min_rank <= knowledge_threshold else "Enrichment"
        )
        return ranked_df
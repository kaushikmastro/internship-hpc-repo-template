import pandas as pd
import torch
from typing import List, Tuple
import re
import random
from .emotional_mi_pipeline import EmotionalMIPipeline

class CausalValidationnalysis:
    def __init__(self, pipeline: EmotionalMIPipeline):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.tokenizer = pipeline.tokenizer
        self.device = pipeline.device

    def calculate_calibration_stats(self, prompts: List[str]) -> Tuple[float, torch.Tensor]:
        """
        Calculates sigma (3 * empirical std dev) and the mean embedding of a set of prompts.
        """
        all_embeddings = []
        embedding_layer = self.model.embed
        
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            if input_ids.numel() == 0:
                continue
            with torch.no_grad():
                token_embeddings = embedding_layer(input_ids)
            mean_embedding = torch.mean(token_embeddings, dim=1)
            all_embeddings.append(mean_embedding.squeeze())
            
        if not all_embeddings:
            raise ValueError("No valid prompts found to calculate calibration stats.")
            
        stacked_embeddings = torch.stack(all_embeddings)
        empirical_std_dev = torch.std(stacked_embeddings, dim=0, unbiased=False)
        mean_embedding = torch.mean(stacked_embeddings, dim=0)
        sigma = 3 * torch.mean(empirical_std_dev).item()
        
        return sigma, mean_embedding

    def _get_target_token_indices(self, prompt: str, target_tokens: List[str]) -> List[int]:
        """
        Finds the token indices for all occurrences of the target tokens in the
        user's portion of a prompt, handling sub-word tokenization.
        """
        instruction_separator = "\nwhat the single emotion of this text?"
        
        if instruction_separator in prompt:
            user_text = prompt.split(instruction_separator, 1)[0].strip()
        else:
            user_text = prompt.strip()

        tokenized_output = self.tokenizer(user_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = tokenized_output['offset_mapping']
        
        target_indices = set()
        
        pattern = r'\b(' + '|'.join(re.escape(word) for word in target_tokens) + r')\b'
        
        for match in re.finditer(pattern, user_text.lower()):
            start_char, end_char = match.span()
            
            for i, (token_start_char, token_end_char) in enumerate(offsets):
                if max(start_char, token_start_char) < min(end_char, token_end_char):
                    target_indices.add(i)
                    
        return sorted(list(target_indices))

    def _get_random_token_index(self, prompt: str) -> int:
        """
        Finds a random token index from the prompt to inject noise.
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        if not input_ids or len(input_ids) < 2:
            return None
        
        # Exclude the last token which is the target for prediction
        return random.randint(0, len(input_ids) - 2)

    def perform_causal_analysis_y_prime(self, hallu_df: pd.DataFrame, sigma_value: float, num_noise_samples: int) -> pd.DataFrame:
        """
        Performs causal analysis by adding scaled Gaussian noise to the embeddings of
        the first token (subject token) of each prompt.
        """
        print(f"\n--- Running Causal Analysis (y' baseline) with {num_noise_samples} samples ---")
        results_df = pd.DataFrame(columns=['prompt_text', 'true_emotion', 'predicted_emotion', 'num_truth_inducing_samples', 'truthful_y_primes'])

        with torch.no_grad():
            for index, row in hallu_df.iterrows():
                prompt_to_analyze = row['constrained prompt'].strip()
                true_emotion = row['emotion']
                predicted_emotion = row['predicted emotion']
                
                try:
                    _, original_cache = self.model.run_with_cache(prompt_to_analyze)
                    original_embeddings = original_cache['embed'].clone().detach()
                except Exception as e:
                    print(f"Error getting original embeddings for prompt {index}: {e}")
                    continue
                
                del original_cache
                torch.cuda.empty_cache()

                try:
                    true_id = self.tokenizer.encode(true_emotion, add_special_tokens=False)[0]
                    predicted_id = self.tokenizer.encode(predicted_emotion, add_special_tokens=False)[0]
                except IndexError:
                    print(f"Skipping prompt {index}: Emotion token not found.")
                    continue

                truthful_y_primes = []
                subject_token_index = 0
                
                for _ in range(num_noise_samples):
                    noise = torch.randn_like(original_embeddings) * sigma_value
                    perturbed_embeddings = original_embeddings.clone()
                    perturbed_embeddings[0, subject_token_index, :] += noise[0, subject_token_index, :]
                    
                    def hook_fn_replace_embed(embed_output, hook):
                        return perturbed_embeddings
                    
                    new_logits = self.model.run_with_hooks(
                        input=self.tokenizer.encode(prompt_to_analyze, return_tensors='pt').to(self.device),
                        fwd_hooks=[('hook_embed', hook_fn_replace_embed)]
                    )
                    
                    final_logits = new_logits[0, -1, :]
                    y_prime = final_logits[predicted_id] - final_logits[true_id]
                    
                    if y_prime.item() < 1:
                        truthful_y_primes.append(y_prime.item())

                new_row = pd.DataFrame([{
                    'prompt_text': prompt_to_analyze,
                    'true_emotion': true_emotion,
                    'predicted_emotion': predicted_emotion,
                    'num_truth_inducing_samples': len(truthful_y_primes),
                    'truthful_y_primes': truthful_y_primes
                }])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                del original_embeddings
                torch.cuda.empty_cache()

        return results_df

    def perform_causal_analysis_emotional_tokens(self, prompt_df: pd.DataFrame, sigma_value: float, num_noise_samples: int, target_tokens: List[str]) -> pd.DataFrame:
        """
        Performs causal analysis by adding scaled Gaussian noise to the embeddings of
        emotionally charged tokens.
        """
        print(f"\n--- Running Causal Analysis (Emotional Tokens) with {num_noise_samples} samples ---")
        results_df = pd.DataFrame(columns=['prompt_text', 'true_emotion', 'predicted_emotion', 'num_truth_inducing_samples', 'truthful_y_primes'])

        with torch.no_grad():
            for index, row in prompt_df.iterrows():
                prompt_to_analyze = row['constrained prompt'].strip()
                true_emotion = row['emotion']
                predicted_emotion = row['predicted emotion']
                
                target_indices = self._get_target_token_indices(prompt_to_analyze, target_tokens)
                
                if not target_indices:
                    print(f"Skipping prompt {index}: No target tokens found.")
                    continue
                
                truthful_y_primes = []
                
                try:
                    _, original_cache = self.model.run_with_cache(prompt_to_analyze)
                    original_embeddings = original_cache['embed'].clone().detach()
                except Exception as e:
                    print(f"Error getting original embeddings for prompt {index}: {e}")
                    continue
                    
                del original_cache
                torch.cuda.empty_cache()

                try:
                    true_id_with_space = self.tokenizer.encode(" " + true_emotion, add_special_tokens=False)[0]
                    predicted_id_with_space = self.tokenizer.encode(" " + predicted_emotion, add_special_tokens=False)[0]
                    true_id_no_space = self.tokenizer.encode(true_emotion, add_special_tokens=False)[0]
                    predicted_id_no_space = self.tokenizer.encode(predicted_emotion, add_special_tokens=False)[0]
                except IndexError:
                    print(f"Skipping prompt {index}: Emotion token not found.")
                    continue
                
                for _ in range(num_noise_samples):
                    noise = torch.randn_like(original_embeddings) * sigma_value
                    perturbed_embeddings = original_embeddings.clone()

                    for token_idx in target_indices:
                        perturbed_embeddings[0, token_idx, :] += noise[0, token_idx, :]
                    
                    def hook_fn_replace_embed(embed_output, hook):
                        return perturbed_embeddings
                    
                    new_logits = self.model.run_with_hooks(
                        input=self.tokenizer.encode(prompt_to_analyze, return_tensors='pt').to(self.device),
                        fwd_hooks=[('hook_embed', hook_fn_replace_embed)]
                    )
                    
                    final_logits = new_logits[0, -1, :]
                    
                    if (predicted_id_with_space in final_logits and true_id_with_space in final_logits):
                        y_prime = final_logits[predicted_id_with_space] - final_logits[true_id_with_space]
                    elif (predicted_id_no_space in final_logits and true_id_no_space in final_logits):
                        y_prime = final_logits[predicted_id_no_space] - final_logits[true_id_no_space]
                    else:
                        continue
                    
                    if y_prime.item() < 1:
                        truthful_y_primes.append(y_prime.item())

                new_row = pd.DataFrame([{
                    'prompt_text': prompt_to_analyze,
                    'true_emotion': true_emotion,
                    'predicted_emotion': predicted_emotion,
                    'num_truth_inducing_samples': len(truthful_y_primes),
                    'truthful_y_primes': truthful_y_primes
                }])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                del original_embeddings
                torch.cuda.empty_cache()

        return results_df

    def perform_causal_analysis_baseline(self, prompt_df: pd.DataFrame, sigma_value: float, num_noise_samples: int) -> pd.DataFrame:
        """
        Performs causal analysis by adding scaled Gaussian noise to the embeddings of
        randomly chosen tokens as a control experiment.
        """
        print(f"\n--- Running Causal Analysis (Random Token Baseline) with {num_noise_samples} samples ---")
        results_df = pd.DataFrame(columns=['prompt_text', 'true_emotion', 'predicted_emotion', 'num_truth_inducing_samples', 'truthful_y_primes'])

        with torch.no_grad():
            for index, row in prompt_df.iterrows():
                prompt_to_analyze = row['constrained prompt'].strip()
                true_emotion = row['emotion']
                predicted_emotion = row['predicted emotion']
                
                random_token_idx = self._get_random_token_index(prompt_to_analyze)
                
                if random_token_idx is None:
                    print(f"Skipping prompt {index}: Not enough tokens.")
                    continue
                
                truthful_y_primes = []
                
                try:
                    _, original_cache = self.model.run_with_cache(prompt_to_analyze)
                    original_embeddings = original_cache['embed'].clone().detach()
                except Exception as e:
                    print(f"Error getting original embeddings for prompt {index}: {e}")
                    continue
                    
                del original_cache
                torch.cuda.empty_cache()

                try:
                    true_id_with_space = self.tokenizer.encode(" " + true_emotion, add_special_tokens=False)[0]
                    predicted_id_with_space = self.tokenizer.encode(" " + predicted_emotion, add_special_tokens=False)[0]
                    true_id_no_space = self.tokenizer.encode(true_emotion, add_special_tokens=False)[0]
                    predicted_id_no_space = self.tokenizer.encode(predicted_emotion, add_special_tokens=False)[0]
                except IndexError:
                    print(f"Skipping prompt {index}: Emotion token not found.")
                    continue
                
                for _ in range(num_noise_samples):
                    noise = torch.randn_like(original_embeddings) * sigma_value
                    perturbed_embeddings = original_embeddings.clone()
                    
                    perturbed_embeddings[0, random_token_idx, :] += noise[0, random_token_idx, :]
                    
                    def hook_fn_replace_embed(embed_output, hook):
                        return perturbed_embeddings
                    
                    new_logits = self.model.run_with_hooks(
                        input=self.tokenizer.encode(prompt_to_analyze, return_tensors='pt').to(self.device),
                        fwd_hooks=[('hook_embed', hook_fn_replace_embed)]
                    )
                    
                    final_logits = new_logits[0, -1, :]
                    
                    if (predicted_id_with_space in final_logits and true_id_with_space in final_logits):
                        y_prime = final_logits[predicted_id_with_space] - final_logits[true_id_with_space]
                    elif (predicted_id_no_space in final_logits and true_id_no_space in final_logits):
                        y_prime = final_logits[predicted_id_no_space] - final_logits[true_id_no_space]
                    else:
                        continue
                    
                    if y_prime.item() < 1:
                        truthful_y_primes.append(y_prime.item())

                new_row = pd.DataFrame([{
                    'prompt_text': prompt_to_analyze,
                    'true_emotion': true_emotion,
                    'predicted_emotion': predicted_emotion,
                    'num_truth_inducing_samples': len(truthful_y_primes),
                    'truthful_y_primes': truthful_y_primes
                }])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                del original_embeddings
                torch.cuda.empty_cache()

        return results_df
import pandas as pd
import torch
from .emotional_mi_pipeline import EmotionAnalysisPipeline

class LogitLensAnalysis:
    def __init__(self, pipeline: EmotionalMIPipeline):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.tokenizer = pipeline.tokenizer
        self.device = pipeline.device

    def analyze_logit_prompts_attention(self, prompt_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes a DataFrame of prompts by calculating logit scores and ranks for each layer,
        based on the output of the attention mechanism only.
        """
        all_metrics = []
        for _, row in prompt_df.iterrows():
            try:
                true_ids = self.pipeline.get_token_ids(row['emotion'])
                predicted_ids = self.pipeline.get_token_ids(row['predicted emotion'])
                input_ids = self.tokenizer.encode(row['constrained prompt'], return_tensors='pt')
                final_token_idx = input_ids.shape[-1] - 1
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(row['constrained prompt'])

                for layer_idx in range(self.model.cfg.n_layers):
                    attn_out_contribution = cache[("attn_out", layer_idx)][0, final_token_idx, :]
                    attn_logits = self.model.unembed(attn_out_contribution)
                    true_logit_raw = attn_logits[true_ids].sum().item()
                    predicted_logit_raw = attn_logits[predicted_ids].sum().item()
                    logit_difference = predicted_logit_raw - true_logit_raw
                    
                    all_metrics.append({
                        'layer': layer_idx,
                        'true_logit_raw': true_logit_raw,
                        'predicted_logit_raw': predicted_logit_raw,
                        'logit_difference': logit_difference
                    })
            except Exception as e:
                print(f"Skipping analysis for prompt: {row['constrained prompt']}\nError: {e}")
                continue

        if not all_metrics:
            print("No metrics were generated.")
            return pd.DataFrame()
        
        all_metrics_df = pd.DataFrame(all_metrics)
        average_metrics_df = all_metrics_df.groupby('layer').mean().reset_index()
        return average_metrics_df

    def analyze_logit_prompts_mlp(self, prompt_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes a DataFrame of prompts by calculating MLP logit scores and ranks for each layer.
        """
        all_metrics = []
        for _, row in prompt_df.iterrows():
            try:
                true_ids = self.pipeline.get_token_ids(row['emotion'])
                predicted_ids = self.pipeline.get_token_ids(row['predicted emotion'])
                input_ids = self.tokenizer.encode(row['constrained prompt'], return_tensors='pt')
                final_token_idx = input_ids.shape[-1] - 1
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(row['constrained prompt'])

                for layer_idx in range(self.model.cfg.n_layers):
                    mlp_out_contribution = cache[("mlp_out", layer_idx)][0, final_token_idx, :]
                    mlp_logits = self.model.unembed(mlp_out_contribution)
                    true_logit_raw = mlp_logits[true_ids].sum().item()
                    predicted_logit_raw = mlp_logits[predicted_ids].sum().item()
                    logit_difference = predicted_logit_raw - true_logit_raw
                    
                    all_metrics.append({
                        'layer': layer_idx,
                        'true_logit_raw': true_logit_raw,
                        'predicted_logit_raw': predicted_logit_raw,
                        'logit_difference': logit_difference
                    })
            except Exception as e:
                print(f"Skipping analysis for prompt: {row['constrained prompt']}\nError: {e}")
                continue

        if not all_metrics:
            return pd.DataFrame()
        
        all_metrics_df = pd.DataFrame(all_metrics)
        average_metrics_df = all_metrics_df.groupby('layer').mean().reset_index()
        return average_metrics_df

    def analyze_logit_attention_distinction(self, prompt_df: pd.DataFrame, distractor_count: int = 100) -> pd.DataFrame:
        """
        Computes the relative attention-extracted attribute information, I_a^(l)(o).
        """
        all_metrics = []
        unembedding_matrix = self.model.unembed.W_U.squeeze()
        
        for _, row in prompt_df.iterrows():
            try:
                true_emotion_text = row['emotion']
                prompt_text = row['constrained prompt']
                true_ids = self.pipeline.get_token_ids(true_emotion_text)
                
                input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt')
                final_token_idx = input_ids.shape[-1] - 1
                
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(prompt_text)

                for layer_idx in range(self.model.cfg.n_layers):
                    mlp_out_contribution = cache[("mlp_out", layer_idx)][0, final_token_idx, :]
                    mlp_logits = self.model.unembed(mlp_out_contribution)
                    _, top_distractor_ids = torch.topk(mlp_logits, k=distractor_count)
                    
                    true_unembedding_vector = unembedding_matrix[true_ids].mean(dim=0)
                    distractor_unembedding_vectors = unembedding_matrix[top_distractor_ids]
                    mean_distractor_vector = distractor_unembedding_vectors.mean(dim=0)
                    distinction_vector = true_unembedding_vector - mean_distractor_vector
                    
                    attn_out_contribution = cache[("attn_out", layer_idx)][0, final_token_idx, :]
                    distinction_score = torch.dot(attn_out_contribution, distinction_vector).item()
                    
                    all_metrics.append({
                        'layer': layer_idx,
                        'distinction_score': distinction_score,
                    })
            except Exception as e:
                print(f"Skipping analysis for prompt: {row['constrained prompt']}\nError: {e}", file=sys.stderr)
                continue

        if not all_metrics:
            return pd.DataFrame()
        
        all_metrics_df = pd.DataFrame(all_metrics)
        average_metrics_df = all_metrics_df.groupby('layer').mean().reset_index()
        return average_metrics_df

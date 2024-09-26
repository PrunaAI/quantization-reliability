from src.data.FKTC_datasets import load_dataset_from_name
from src.reliability.response_generator import ResponseGenerator
from src.models import base_models

import os
import pandas as pd
import numpy as np
from sklearn import metrics

def evaluate_reliability(
    exp_id: str,
    model_name: str,
    dataset_name: str,
    taxonomy_type: str,
    typo_type: str,
    typo_intensity: int,
    strategy: str,
    max_new_tokens: int,
    temperature: float,
    use_beam_search: bool,
    n_repeats: int,
    n_beams: int,
    max_entries: int = None
  ):
  # LOAD DATASET
    qa_dataset = load_dataset_from_name(
      dataset_name,
      max_relations=1,
      max_entries=max_entries,
      taxonomy_type=taxonomy_type,
      typo_type=typo_type,
      typo_intensity=typo_intensity
    )

    # INITIALIZE RESULTS LIST
    results = []

    n_steps = 0
    total_steps = len(qa_dataset) * (n_repeats if not use_beam_search else 1)
    generator = ResponseGenerator(model_name)
    for query_idx, (query, true_answer) in enumerate(qa_dataset):
        run_results = generator.generate_response(query, strategy, dataset_name, true_answer, max_new_tokens, temperature, use_beam_search, n_repeats=n_repeats, n_beams=n_beams)
        for result_dict in run_results:
            print(f"  TOTAL: {n_steps + 1}/{total_steps}, MODEL: {model_name}, QUERY: {query_idx}, STRATEGY: {strategy}, MAX_NEW_TOKENS: {max_new_tokens}, RUN: {result_dict['run']}/{n_repeats}")
            # Store the results in the list
            results.append({
                "Query ID": query_idx,
                "Query": query,
                "Answer": true_answer,
                "Run": result_dict['run'],
                "Generated Response": result_dict['output_text'],
                "Cleaned": result_dict['cleaned'],
                "P": result_dict['beam_prob'],
                "P_adj": result_dict['beam_prob_adj'],
                "Entropy": result_dict['entropy'],
                "Is Correct": result_dict['is_correct'],
                "Token Probabilities": result_dict['token_probs']
            })
            print(f"    IS_CORRECT: {result_dict['is_correct']}, CLEANED: {result_dict['cleaned']}, PROB: {result_dict['beam_prob']:.2f}, ADJ_PROB: {result_dict['beam_prob_adj']:.2f}, ENTROPY: {result_dict['entropy']:.2f}")
            n_steps += 1
    
    # Generate custom file name based on parameters
    beam_search_str = "beam" if use_beam_search else "sample"
    strategy_str = strategy.replace(" ", "_").lower()  # Replace spaces with underscores for file names
    file_base = f"{model_name}_{dataset_name}_{taxonomy_type}_{beam_search_str}_{max_new_tokens}_tokens_{temperature}_temp_{strategy_str}"

    # Optional: Create a directory for saving the results if not already existing
    results_path = "/nfs/homedirs/daro/git/quantization-reliability/results"
    save_dir = os.path.join(results_path, "reliability_eval")
    os.makedirs(save_dir, exist_ok=True)

    # Generate file paths
    exp_path = os.path.join(save_dir, f"reliability_eval_{exp_id}")
    os.makedirs(exp_path, exist_ok=True)
    
    raw_table_path = os.path.join(exp_path, f"{file_base}_raw_table_{exp_id}.xlsx")
    scores_table_path = os.path.join(exp_path, f"{file_base}_scores_{exp_id}.xlsx")
    
    df_results = pd.DataFrame(results)
    
    # Calculate P_sem as the proportion of True values in 'Is Correct' per group
    df_results['P_sem'] = df_results.groupby(['Query ID'])['Is Correct'].transform('mean')

    # Define custom AUC calculation
    def custom_auc_roc(corrects, scores):
        fpr, tpr, thresholds = metrics.roc_curve(corrects, scores)
        return metrics.auc(fpr, tpr)
    
    def calculate_scores(group):
        y_true = group['Is Correct'].values

        # Calculate various scores
        scores_dict = {}
        metrics_to_calculate = {
            'sample': 'P',
            'adj': 'P_adj',
            'sem': 'P_sem'
        }

        for key, score_column in metrics_to_calculate.items():
            y_scores = group[score_column].values
            if len(set(y_true)) > 1:  # Ensure at least two classes are present
                aucroc = custom_auc_roc(y_true, y_scores)
                aucpr = metrics.average_precision_score(y_true, y_scores)
                brier_score = metrics.brier_score_loss(y_true, y_scores)
                # Add check for log_loss
                if len(set(y_scores)) > 1:
                    log_loss = metrics.log_loss(y_true, y_scores)
                else:
                    log_loss = np.nan
            else:
                aucroc = np.nan
                aucpr = np.nan
                brier_score = np.nan
                log_loss = np.nan

            accuracy = np.mean(y_true)
            entropy = -np.sum(y_scores * np.log2(y_scores + 1e-10))  # Added small constant to avoid log(0)

            scores_dict[f'AUCROC_{key}'] = aucroc
            scores_dict[f'AUCPR_{key}'] = aucpr
            scores_dict[f'Brier_{key}'] = brier_score
            scores_dict[f'LogLoss_{key}'] = log_loss
            scores_dict[f'Entropy_{key}'] = entropy
            
        scores_dict[f'Accuracy'] = accuracy
        
        return pd.Series(scores_dict)

    # Apply the calculate_scores function to the entire DataFrame
    df_scores = calculate_scores(df_results)

    # Convert df_scores to a DataFrame with a single row for consistent saving format
    df_scores = df_scores.to_frame().T

    # Save the original detailed results to an Excel file
    df_results.to_excel(raw_table_path, index=False)
    df_scores.to_excel(scores_table_path, index=False)
    
    return df_results, df_scores
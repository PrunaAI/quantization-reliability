import os
import pandas as pd
from typing import List, Tuple
from collections import defaultdict
from itertools import product

class ExcelUnifier:
    def __init__(self, save_dir: str, exp_id: str):
        self.save_dir = save_dir
        self.exp_id = exp_id
        self.exp_dir = os.path.join(save_dir, f"reliability_eval_{exp_id}")
        self.unified_scores_table_path = os.path.join(save_dir, f"unified_scores_table_{exp_id}_complete.xlsx")
        self.exclude_files = []
        self.unified_scores_table = pd.DataFrame()

    def unify_excel_files(self):
        print(f"Unifying Excel files in {self.exp_dir}")
        print("Number of files:", len(os.listdir(self.exp_dir)))
        
        # Dictionary to store unique values for each feature
        unique_values = defaultdict(set)
        
        # Set to store all found parameter combinations
        found_combinations = set()

        total_files = 0
        for filename in os.listdir(self.exp_dir):
            if self._is_valid_file(filename):
                file_info = self._parse_filename(filename)
                if file_info[2] == "0":  # Check if taxonomy_type is "0"
                    df = self._load_and_process_file(filename, file_info)
                    self._append_to_unified_table(df)
                    total_files += 1
                    # Collect unique values
                    unique_values['taxonomy_type'].add(file_info[2])
                    unique_values['beam_search'].add(file_info[3])
                    unique_values['model'].add(file_info[0])
                    unique_values['dataset_name'].add(file_info[1])
                    unique_values['temperature'].add(file_info[5])
                    unique_values['strategy'].add(file_info[6])
                    unique_values['max_new_tokens'].add(file_info[4])
                    
                    # Add the combination to the found set
                    found_combinations.add(tuple(file_info))
        
        print(f"Total number of files processed: {total_files}")
        self._save_unified_table()
        self._print_results()
        
        # Print unique values and their counts
        print("\nUnique values for each feature:")
        for feature, values in unique_values.items():
            print(f"{feature.capitalize()}:")
            print(f"  Unique values: {values}")
            print(f"  Number of unique values: {len(values)}")
            print()

        # Generate all possible combinations
        all_combinations = set(product(
            unique_values['model'],
            unique_values['dataset_name'],
            unique_values['taxonomy_type'],
            unique_values['beam_search'],
            unique_values['max_new_tokens'],
            unique_values['temperature'],
            unique_values['strategy']
        ))

        # Find the missing combination
        missing_combinations = all_combinations - found_combinations
        
        if missing_combinations:
            print("Missing parameter combination:")
            for combo in missing_combinations:
                print(f"Model: {combo[0]}")
                print(f"Dataset Name: {combo[1]}")
                print(f"Taxonomy Type: {combo[2]}")
                print(f"Beam Search: {combo[3]}")
                print(f"Max New Tokens: {combo[4]}")
                print(f"Temperature: {combo[5]}")
                print(f"Strategy: {combo[6]}")
        else:
            print("No missing parameter combinations found.")

    def _is_valid_file(self, filename: str) -> bool:
        return (filename.endswith(f"{self.exp_id}.xlsx") and
                filename not in self.exclude_files and
                'scores' in filename)

    def _parse_filename(self, filename: str) -> Tuple[str, str, str, str, int, float, str]:
        parts = filename.replace(".xlsx", "").split('_')
        return (
            parts[0],  # model
            parts[1],  # dataset_name
            parts[2],  # taxonomy_type
            parts[3],  # beam_search
            int(parts[4]),  # max_new_tokens
            float(parts[6]),  # temperature
            '_'.join(parts[8:]).replace("scores", "").replace(self.exp_id, "").replace("___", "_").strip('_')  # strategy
        )

    def _load_and_process_file(self, filename: str, file_info: Tuple) -> pd.DataFrame:
        file_path = os.path.join(self.exp_dir, filename)
        df = pd.read_excel(file_path)
        model, dataset_name, taxonomy_type, beam_search, max_new_tokens, temperature, strategy = file_info
        df['Model'] = model
        df['Dataset Name'] = dataset_name
        df['Taxonomy Type'] = taxonomy_type
        df['Beam Search'] = beam_search
        df['Max New Tokens'] = max_new_tokens
        df['Temperature'] = temperature
        df['Strategy'] = strategy
        return df

    def _append_to_unified_table(self, df: pd.DataFrame):
        self.unified_scores_table = pd.concat([self.unified_scores_table, df], ignore_index=True)

    def _save_unified_table(self):  
        desired_scores_column_order = [
            'Model', 'Dataset Name', 'Taxonomy Type', 'Strategy', 'Beam Search', 'Max New Tokens', 'Temperature',
            'Accuracy', 'AUCROC_sample', 'AUCPR_sample', 'AUCROC_adj', 'AUCPR_adj',
            'AUCROC_sem', 'AUCPR_sem'
        ]
        self.unified_scores_table = self.unified_scores_table[desired_scores_column_order]
        self.unified_scores_table.to_excel(self.unified_scores_table_path, index=False)

    def _print_results(self):
        print(self.unified_scores_table.columns)
        print(f"Unified scores table saved to {self.unified_scores_table_path}")

# Example usage:
if __name__ == "__main__":
    exp_id = "09-17-1"
    unifier = ExcelUnifier("results/reliability_eval", exp_id)
    unifier.unify_excel_files()
from typing import Dict


class ScoresCalculator:
    """
    Class to calculate scores from model outputs.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        inference_config: Dict = {},
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.inference_config = inference_config if inference_config is not None else {}

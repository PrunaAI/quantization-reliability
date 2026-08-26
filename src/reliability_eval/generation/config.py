from dataclasses import dataclass
from typing import Any, Dict, Optional

import wandb

from src.reliability_eval.common.experiment import GenerationExperimentConfig
from src.reliability_eval.common.params import ExperimentConfigParam, GenerationConfigParam
from src.reliability_eval.common.generation import GenerationStrategy
from src.reliability_eval.generation.strategy import GenerationStrategyMapper
from src.model_loading.common.tokenizer_utils import get_actual_tokenizer


@dataclass
class BaseGenerationConfig:
    """Base generation configuration with default values."""
    max_new_tokens: int = 25
    do_sample: bool = False
    use_cache: bool = True
    length_penalty: float = 1.0
    output_scores: bool = True
    output_hidden_states: bool = False
    output_attentions: bool = False
    output_logits: bool = True
    return_dict_in_generate: bool = True
    pad_token_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts config to dictionary using enum keys."""
        return {
            GenerationConfigParam.MAX_NEW_TOKENS.value: self.max_new_tokens,
            GenerationConfigParam.DO_SAMPLE.value: self.do_sample,
            GenerationConfigParam.USE_CACHE.value: self.use_cache,
            GenerationConfigParam.LENGTH_PENALTY.value: self.length_penalty,
            GenerationConfigParam.OUTPUT_SCORES.value: self.output_scores,
            GenerationConfigParam.OUTPUT_HIDDEN_STATES.value: self.output_hidden_states,
            GenerationConfigParam.OUTPUT_ATTENTIONS.value: self.output_attentions,
            GenerationConfigParam.OUTPUT_LOGITS.value: self.output_logits,
            GenerationConfigParam.RETURN_DICT_IN_GENERATE.value: self.return_dict_in_generate,
            GenerationConfigParam.PAD_TOKEN_ID.value: self.pad_token_id,
        }
        
        
class GenerationConfigHandler:
    """Handles generation configuration setup and validation."""

    def __init__(self, tokenizer):
        """Initializes handler with tokenizer for token-specific configurations."""
        self.tokenizer = tokenizer
        self.actual_tokenizer = get_actual_tokenizer(tokenizer)
        self._config = None

    def create_config(
        self,
        generation_experiment_config: Optional[GenerationExperimentConfig] = None
    ) -> Dict:
        """Creates complete generation configuration from input settings."""
        self._config = generation_experiment_config or GenerationExperimentConfig()
        return self._create_full_generation_config()

    def _create_full_generation_config(self) -> Dict:
        """Creates complete generation configuration from current settings."""
        base_config = self._create_base_config()
        extended_config = self._add_model_specific_config(base_config)
        strategy_config = self._add_strategy_specific_config(extended_config)
        self._validate_config(strategy_config)
        
        return strategy_config

    def _create_base_config(self) -> Dict[str, Any]:
        """Creates base configuration with standard parameters."""
        return BaseGenerationConfig(
            max_new_tokens=self._config.max_new_tokens,
            pad_token_id=self.actual_tokenizer.eos_token_id
        ).to_dict()

    def _add_model_specific_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Adds model-specific parameters to configuration."""
        model_config = {
            # Low-level generation params
            GenerationConfigParam.NUM_BEAMS.value: self._config.num_beams,
            GenerationConfigParam.NUM_RETURN_SEQUENCES.value: self._config.num_return_sequences,
            # High-level experiment params
            ExperimentConfigParam.GENERATION_STRATEGY.value: self._config.generation_strategy,
            ExperimentConfigParam.DATASET_NAME.value: self._config.dataset_name,
            ExperimentConfigParam.NUM_SHOTS.value: self._config.num_shots,
            ExperimentConfigParam.PROMPT_STRATEGY.value: self._config.prompt_strategy,
            ExperimentConfigParam.NUM_REPEATS.value: self._config.num_repeats,
            ExperimentConfigParam.TOP_K_CONCENTRATION.value: self._config.top_k_concentration
        }
        config.update({k: v for k, v in model_config.items() if v is not None})
        return config

    def _add_strategy_specific_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Adds strategy-specific parameters to configuration."""
        strategy = self._config.generation_strategy
        strategy_config = GenerationStrategyMapper.get_strategy_config(
            strategy=strategy,
            config=self._get_strategy_params()
        )
        config.update(strategy_config)
        return config
        
    def _get_strategy_params(self) -> Dict[str, Any]:
        """Gets strategy-specific parameters from current configuration."""
        return {
            GenerationConfigParam.NUM_BEAMS: self._config.num_beams,
            GenerationConfigParam.NUM_RETURN_SEQUENCES: self._config.num_return_sequences,
            GenerationConfigParam.NUM_BEAM_GROUPS: self._config.num_beam_groups,
            GenerationConfigParam.PENALTY_ALPHA: self._config.penalty_alpha,
            GenerationConfigParam.TOP_K: self._config.top_k,
            GenerationConfigParam.TOP_P: self._config.top_p,
            GenerationConfigParam.TEMPERATURE: self._config.temperature
        }

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validates generation configuration parameters."""
        self._validate_sampling_parameters(config)
        self._validate_beam_search_compatibility(config)

    def _validate_sampling_parameters(self, config: Dict[str, Any]) -> None:
        """Checks compatibility of sampling parameters."""
        has_top_k = config.get(GenerationConfigParam.TOP_K) is not None
        has_top_p = config.get(GenerationConfigParam.TOP_P) is not None
        if has_top_k and has_top_p:
            raise ValueError("Only one of top_k or top_p should be specified")

    def _validate_beam_search_compatibility(self, config: Dict[str, Any]) -> None:
        """Checks compatibility of beam search parameters."""
        has_top_k = config.get(GenerationConfigParam.TOP_K) is not None
        num_beams = config.get(GenerationConfigParam.NUM_BEAMS, 1)
        if has_top_k and num_beams > 1:
            raise ValueError("Top-k sampling is not compatible with beam search")
        
        
def create_generation_config(
    base_config: BaseGenerationConfig,
    strategy: GenerationStrategy,
    additional_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Creates complete generation configuration."""
    config = base_config.to_dict()
    strategy_config = GenerationStrategyMapper.get_strategy_config(strategy, additional_config or {})
    config.update(strategy_config)
    if additional_config:
        config.update({k: v for k, v in additional_config.items() if v is not None})
    return config
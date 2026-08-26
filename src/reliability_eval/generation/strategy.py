from typing import Any, Dict
from src.reliability_eval.common.params import GenerationConfigParam
from src.reliability_eval.common.generation import GenerationStrategy


class GenerationStrategyMapper:
    """Maps generation strategies to their specific configurations."""
    
    @staticmethod
    def get_greedy_search_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns greedy search configuration."""
        return {
            GenerationConfigParam.DO_SAMPLE.value: False,
            GenerationConfigParam.NUM_BEAMS.value: 1,
            GenerationConfigParam.TEMPERATURE.value: config.get(GenerationConfigParam.TEMPERATURE) or 0.7
        }
    
    @staticmethod
    def get_contrastive_search_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns contrastive search configuration."""
        return {
            GenerationConfigParam.PENALTY_ALPHA.value: config.get(GenerationConfigParam.PENALTY_ALPHA) or 0.6,
            GenerationConfigParam.TOP_K.value: config.get(GenerationConfigParam.TOP_K) or 4,
            GenerationConfigParam.TEMPERATURE.value: config.get(GenerationConfigParam.TEMPERATURE) or 0.7
        }
    
    @staticmethod
    def get_multinomial_sampling_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns multinomial sampling configuration."""
        return {
            GenerationConfigParam.DO_SAMPLE.value: True,
            GenerationConfigParam.NUM_BEAMS.value: 1,
            GenerationConfigParam.TEMPERATURE.value: config.get(GenerationConfigParam.TEMPERATURE) or 0.7
        }
    
    @staticmethod
    def get_beam_search_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns beam search configuration."""
        return {
            GenerationConfigParam.DO_SAMPLE.value: False,
            GenerationConfigParam.NUM_BEAMS.value: config.get(GenerationConfigParam.NUM_BEAMS) or 5,
            GenerationConfigParam.NUM_RETURN_SEQUENCES.value: config.get(GenerationConfigParam.NUM_RETURN_SEQUENCES) or 1,
            GenerationConfigParam.TEMPERATURE.value: config.get(GenerationConfigParam.TEMPERATURE) or 0.7
        }
        
    @staticmethod
    def get_beam_search_with_sampling_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns beam search with sampling configuration."""
        return {
            GenerationConfigParam.DO_SAMPLE.value: True,
            GenerationConfigParam.NUM_BEAMS.value: config.get(GenerationConfigParam.NUM_BEAMS) or 5,
            GenerationConfigParam.NUM_RETURN_SEQUENCES.value: config.get(GenerationConfigParam.NUM_RETURN_SEQUENCES) or 1,
            GenerationConfigParam.TEMPERATURE.value: config.get(GenerationConfigParam.TEMPERATURE) or 0.7
        }
        
    @staticmethod
    def get_diverse_beam_search_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns diverse beam search configuration."""
        return {
            GenerationConfigParam.DO_SAMPLE.value: False,
            GenerationConfigParam.NUM_BEAMS.value: config.get(GenerationConfigParam.NUM_BEAMS) or 4,
            GenerationConfigParam.NUM_RETURN_SEQUENCES.value: config.get(GenerationConfigParam.NUM_RETURN_SEQUENCES) or 1,
            GenerationConfigParam.NUM_BEAM_GROUPS.value: config.get(GenerationConfigParam.NUM_BEAM_GROUPS) or 2,
            GenerationConfigParam.TEMPERATURE.value: config.get(GenerationConfigParam.TEMPERATURE) or 0.7
        }
    
    @staticmethod
    def get_top_k_sampling_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns top-k sampling configuration."""
        return {
            GenerationConfigParam.DO_SAMPLE.value: True,
            GenerationConfigParam.TOP_K.value: config.get(GenerationConfigParam.TOP_K) or 4,
            GenerationConfigParam.TEMPERATURE.value: config.get(GenerationConfigParam.TEMPERATURE) or 0.7
        }
        
    @staticmethod
    def get_top_p_sampling_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns top-p sampling configuration."""
        return {
            GenerationConfigParam.DO_SAMPLE.value: True,
            GenerationConfigParam.TOP_P.value: config.get(GenerationConfigParam.TOP_P) or 0.9,
            GenerationConfigParam.TEMPERATURE.value: config.get(GenerationConfigParam.TEMPERATURE) or 0.7
        }
    
    @classmethod
    def get_strategy_config(
        cls,
        strategy: GenerationStrategy,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Returns configuration for specified generation strategy."""
        strategy_map = {
            GenerationStrategy.GREEDY_SEARCH: cls.get_greedy_search_config,
            GenerationStrategy.CONTRASTIVE_SEARCH: cls.get_contrastive_search_config,
            GenerationStrategy.MULTINOMIAL_SAMPLING: cls.get_multinomial_sampling_config,
            GenerationStrategy.BEAM_SEARCH: cls.get_beam_search_config,
            GenerationStrategy.BEAM_SEARCH_WITH_SAMPLING: cls.get_beam_search_with_sampling_config,
            GenerationStrategy.DIVERSE_BEAM_SEARCH: cls.get_diverse_beam_search_config,
            GenerationStrategy.TOP_K_SAMPLING: cls.get_top_k_sampling_config,
            GenerationStrategy.TOP_P_SAMPLING: cls.get_top_p_sampling_config
        }
        return strategy_map[strategy](config)

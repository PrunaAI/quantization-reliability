from typing import Dict, Optional, Union
from src.model_loading.common.identifier import ModelIdentifier
from src.model_loading.common.paths import ModelPaths
from src.model_loading.registry.models import Models
from src.model_loading.registry.string_utils import ModelStringifier

class EnhancedModelRegistry:
    """Enhanced model registry with string-based access"""
    def __init__(self, base_registry):
        self._base_registry = base_registry
        self._string_to_model: Dict[str, ModelIdentifier] = {}
        self._initialize_string_mappings()
        
    def _initialize_string_mappings(self):
        """Initialize mappings between strings and ModelIdentifiers"""
        all_models = Models.get_all_models()
        for model in all_models.values():
            model_str = ModelStringifier.to_string(model)
            self._string_to_model[model_str] = model
            
    def get_model_by_string(self, model_str: str) -> Optional[Union[ModelIdentifier, ModelPaths]]:
        """Get model identifier or paths from string representation"""
        # First check if this string is already in our mapping
        model = self._string_to_model.get(model_str)
        if model:
            return model
            
        # If not found, try parsing the string
        model = ModelStringifier.from_string(model_str)
        if model:
            # You might want to get the actual registered model that matches
            # this identifier, rather than just returning a new instance
            for registered_id, registered_model in self._string_to_model.items():
                if model == registered_model:
                    return registered_model
            # If we didn't find a match, return the newly created model
            return model
            
        # If string couldn't be parsed into a valid model, return None
        return None
        
    def get_all_model_strings(self) -> list[str]:
        """Get all available model strings"""
        return sorted(self._string_to_model.keys())

import torch
from transformers import StoppingCriteria


class SingleTokenStoppingCriteria(StoppingCriteria):
    """Stops generation if the last generated token is in the terminator list."""
    def __init__(self, terminator_ids: list):
        self.terminator_ids = terminator_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Get the last token ID for each sequence in the batch
        last_token = input_ids[:, -1]
        # Check if any of the last tokens are in our terminator list
        return any(token.item() in self.terminator_ids for token in last_token)

    def __len__(self):
        return 1
      
class PerBeamStoppingCriteria(StoppingCriteria):
    """Implements per-beam stopping with masking for terminated beams."""
    
    def __init__(self, terminator_ids: list):
        super().__init__()
        self.terminator_ids = terminator_ids
        self.beam_stopped = None  # Will be initialized on first call
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Initialize beam_stopped tensor if not already done
        if self.beam_stopped is None:
            self.beam_stopped = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        
        # Get the last token for each sequence in the batch
        last_tokens = input_ids[:, -1]
        
        # Update stopped beams based on terminator tokens
        for idx, token in enumerate(last_tokens):
            if not self.beam_stopped[idx]:  # Only check active beams
                if token.item() in self.terminator_ids:
                    self.beam_stopped[idx] = True
        
        # Modify scores for stopped beams if provided
        if scores is not None and self.beam_stopped.any():
            # Set scores for stopped beams to -inf except for padding token
            for idx in range(scores[-1].shape[0]):
                if self.beam_stopped[idx]:
                    # Set all logits to -inf except padding token
                    scores[-1][idx, :] = float('-inf')
                    scores[-1][idx, kwargs.get('pad_token_id', 0)] = 0.0  # Allow only padding for stopped beams
        
        # Return True only if all beams are stopped
        return self.beam_stopped.all().item()

    def __len__(self):
        return 1
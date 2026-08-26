import torch
import logging
from typing import List, Tuple, Optional, Dict, Any

from lm_eval.api.model import LM

logger = logging.getLogger(__name__)


class CustomModelWrapper(LM):
    def __init__(self, model, tokenizer, model_name: str,
                 batch_size: int = 1, is_multimodal: bool = False, processor=None):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self._batch_size = batch_size
        self.MULTIMODAL = is_multimodal
        self.processor = processor

        if processor is not None and hasattr(processor, "tokenizer"):
            self.tokenizer = processor.tokenizer
        elif hasattr(tokenizer, "tokenizer"):
            self.tokenizer = tokenizer.tokenizer
            self.processor = tokenizer
        else:
            self.tokenizer = tokenizer

        if hasattr(model, "device"):
            self._device = model.device
        elif hasattr(model, "hf_device_map"):
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Side-channel: (task_name, doc_id) -> (mean_log_prob, mean_entropy)
        self._generation_logprobs: Dict[Tuple[Optional[str], Optional[int]], Tuple[float, float]] = {}

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return getattr(self.model.config, "max_position_embeddings", 2048)

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def tok_encode(self, string: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _compute_logprobs_from_scores(self, gen_out, generated_tokens):
        if not gen_out.scores or len(generated_tokens) == 0:
            return 0.0, 0.0
        scores = torch.stack(gen_out.scores, dim=0).squeeze(1)[:50]
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=-1e4)
        lp = torch.log_softmax(scores.float(), dim=-1)
        safe_toks = generated_tokens[:50].clamp(0, scores.shape[-1] - 1).to(lp.device)
        mean_log_prob = lp[torch.arange(len(safe_toks), device=lp.device), safe_toks].mean().item()
        mean_entropy = -(lp.exp() * lp).sum(dim=-1).mean().item()
        return mean_log_prob, mean_entropy

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        results = []
        for req in requests:
            ctx, cont = req.args[0], req.args[1]
            ctx_toks = self.tok_encode(ctx, add_special_tokens=True)
            full_toks = self.tok_encode(ctx + cont, add_special_tokens=True)
            cont_toks = full_toks[len(ctx_toks):]
            if not cont_toks:
                results.append((0.0, True))
                continue
            input_ids = torch.tensor([full_toks], dtype=torch.long, device=self._device)
            with torch.no_grad():
                logits = self.model(input_ids).logits[0]
                lp = torch.log_softmax(logits, dim=-1)
            ll = sum(lp[len(ctx_toks) + i - 1, t].item() for i, t in enumerate(cont_toks))
            is_greedy = all(
                torch.argmax(lp[len(ctx_toks) + i - 1]).item() == t
                for i, t in enumerate(cont_toks)
            )
            results.append((ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests) -> List[float]:
        results = []
        for req in requests:
            toks = self.tok_encode(req.args[0], add_special_tokens=True)
            if len(toks) <= 1:
                results.append(0.0)
                continue
            input_ids = torch.tensor([toks], dtype=torch.long, device=self._device)
            with torch.no_grad():
                lp = torch.log_softmax(self.model(input_ids).logits[0], dim=-1)
            results.append(sum(lp[i - 1, toks[i]].item() for i in range(1, len(toks))))
        return results

    def generate_until(self, requests) -> List[str]:
        results = []
        for req in requests:
            ctx = req.args[0]
            kwargs = req.args[1] if len(req.args) > 1 else {}
            input_ids = torch.tensor(
                [self.tok_encode(ctx, add_special_tokens=True)],
                dtype=torch.long, device=self._device
            )
            do_sample = kwargs.get("do_sample", False)
            with torch.no_grad():
                gen_out = self.model.generate(
                    input_ids,
                    max_new_tokens=kwargs.get("max_gen_toks", self.max_gen_toks),
                    temperature=kwargs.get("temperature", 1.0) if do_sample else 1.0,
                    top_p=kwargs.get("top_p", 1.0) if do_sample else 1.0,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            gen_toks = gen_out.sequences[0][len(input_ids[0]):]
            text = self.tok_decode(gen_toks.tolist())
            for stop in kwargs.get("until", []):
                if stop in text:
                    text = text.split(stop)[0]
                    break
            key = (getattr(req, "task_name", None), getattr(req, "doc_id", None))
            self._generation_logprobs[key] = self._compute_logprobs_from_scores(gen_out, gen_toks)
            results.append(text)
        return results

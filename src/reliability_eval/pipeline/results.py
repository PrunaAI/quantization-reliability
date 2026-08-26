from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class MetricResults:
    """Container for dataset metric results."""
    accuracy: Optional[float] = None
    aucpr: Optional[float] = None
    aucroc: Optional[float] = None
    brier: Optional[float] = None
    mean_scores: Optional[float] = None
    

@dataclass
class EvaluationResults:
    """Container for evaluation results including scores and metrics."""
    scores: Dict[str, torch.Tensor]
    metrics: Dict[str, MetricResults]
    

@dataclass
class GroupedResults:
    """Container for results grouped by perturbation type and intensity."""
    overall_results: EvaluationResults
    grouped_results: Dict[Tuple[str, int], EvaluationResults]
    
    
@dataclass
class ProcessedMetrics:
    """Simplified container for metric values."""
    aucpr: Optional[float]
    aucroc: Optional[float]
    brier: Optional[float]
    mean_scores: Optional[float]


@dataclass
class ProcessedResult:
    """Container for processed single result entry."""
    perturbation_type: Optional[str]
    perturbation_intensity: Optional[int]
    pipeline_scores: Dict[str, List[float]]
    pipeline_metrics: Dict[str, ProcessedMetrics]

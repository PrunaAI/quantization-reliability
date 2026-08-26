import torch
import numpy as np
from sklearn import metrics
import wandb
from src.loggers.wandb_utils import safe_wandb_log
from typing import Dict, List, Optional, Tuple, Any
import json

import logging
from src.reliability_eval.common.score_types import DatasetMetricTypes
from src.reliability_eval.pipeline.context import MetricsConfig
from src.reliability_eval.pipeline.results import MetricResults

logger = logging.getLogger("reliability_eval")

class MetricsCalculator:
    """Handles computation of evaluation metrics with enhanced logging and wandb tracking."""
    
    def __init__(self, config: MetricsConfig):
        """Initialize with metrics configuration."""
        self.config = config
        
    def compute_metrics(self, scores: torch.Tensor, accuracy: torch.Tensor) -> MetricResults:
        """
        Computes metrics for given scores and accuracy values with detailed logging.
        
        Args:
            scores: Tensor containing score values for evaluation
            accuracy: Tensor containing accuracy/ground truth values
            
        Returns:
            MetricResults object with computed metrics
        """
        logger.debug("Computing metrics")
        
        # Log tensor shapes and types for debugging
        tensor_info = {
            "scores_shape": list(scores.shape),
            "scores_dtype": str(scores.dtype),
            "accuracy_shape": list(accuracy.shape),
            "accuracy_dtype": str(accuracy.dtype)
        }
        
        logger.info(f"Input tensor info: {json.dumps(tensor_info)}")
        safe_wandb_log({"metrics/tensor_info": tensor_info})
        
        # Convert tensors to numpy arrays for processing
        scores_np = scores.cpu().detach().numpy()
        accuracy_np = accuracy.cpu().detach().numpy()
        
        # Analyze NaN values in the input arrays
        nan_stats = self._analyze_nan_values(scores_np, accuracy_np)
        
        # Log NaN statistics
        logger.warning(f"NaN statistics: {json.dumps(nan_stats)}")
        safe_wandb_log({"metrics/nan_stats": nan_stats})
        
        # Log histograms of values to wandb
        try:
            safe_wandb_log({
                "metrics/scores_histogram": wandb.Histogram(scores_np),
                "metrics/accuracy_histogram": wandb.Histogram(accuracy_np)
            })
        except Exception as e:
            logger.warning(f"Failed to log histograms to wandb: {str(e)}")
        
        # Calculate metrics with enhanced error tracking
        return self._calculate_metrics(scores_np, accuracy_np, nan_stats)
    
    def _analyze_nan_values(self, scores: np.ndarray, accuracy: np.ndarray) -> Dict[str, Any]:
        """
        Analyze NaN values in the input arrays and return detailed statistics.
        
        Args:
            scores: Numpy array of scores
            accuracy: Numpy array of accuracy values
            
        Returns:
            Dict containing statistics about NaN values
        """
        # Check for NaN values in scores
        scores_nan_mask = np.isnan(scores)
        scores_nan_count = np.sum(scores_nan_mask)
        scores_total = scores.size
        
        # Check for NaN values in accuracy
        accuracy_nan_mask = np.isnan(accuracy)
        accuracy_nan_count = np.sum(accuracy_nan_mask)
        accuracy_total = accuracy.size
        
        # Identify positions where both arrays have NaN or only one has NaN
        both_nan_mask = scores_nan_mask & accuracy_nan_mask
        both_nan_count = np.sum(both_nan_mask)
        
        scores_only_nan_mask = scores_nan_mask & ~accuracy_nan_mask
        scores_only_nan_count = np.sum(scores_only_nan_mask)
        
        accuracy_only_nan_mask = ~scores_nan_mask & accuracy_nan_mask
        accuracy_only_nan_count = np.sum(accuracy_only_nan_mask)
        
        # Get indices where NaN values occur
        scores_nan_indices = np.where(scores_nan_mask)[0]
        accuracy_nan_indices = np.where(accuracy_nan_mask)[0]
        
        # Track if we have all NaN values
        all_scores_nan = scores_nan_count == scores_total
        all_accuracy_nan = accuracy_nan_count == accuracy_total
        
        # Calculate percentage of NaN values
        scores_nan_percent = (scores_nan_count / scores_total) * 100 if scores_total > 0 else 0
        accuracy_nan_percent = (accuracy_nan_count / accuracy_total) * 100 if accuracy_total > 0 else 0
        
        # Get some sample indices for NaN values (up to 10)
        sample_nan_indices = {
            "scores": scores_nan_indices[:10].tolist() if len(scores_nan_indices) > 0 else [],
            "accuracy": accuracy_nan_indices[:10].tolist() if len(accuracy_nan_indices) > 0 else []
        }
        
        # Collect non-NaN statistics for context
        valid_mask = ~scores_nan_mask & ~accuracy_nan_mask
        valid_count = np.sum(valid_mask)
        
        # Compute basic stats for non-NaN values
        if valid_count > 0:
            valid_scores = scores[valid_mask]
            valid_accuracy = accuracy[valid_mask]
            score_stats = {
                "min": float(np.min(valid_scores)),
                "max": float(np.max(valid_scores)),
                "mean": float(np.mean(valid_scores)),
                "std": float(np.std(valid_scores))
            }
            accuracy_stats = {
                "min": float(np.min(valid_accuracy)),
                "max": float(np.max(valid_accuracy)),
                "mean": float(np.mean(valid_accuracy)),
                "std": float(np.std(valid_accuracy))
            }
        else:
            score_stats = {"min": None, "max": None, "mean": None, "std": None}
            accuracy_stats = {"min": None, "max": None, "mean": None, "std": None}
        
        # Return comprehensive NaN statistics
        return {
            "scores_nan_count": int(scores_nan_count),
            "scores_total": int(scores_total),
            "scores_nan_percent": float(scores_nan_percent),
            "accuracy_nan_count": int(accuracy_nan_count), 
            "accuracy_total": int(accuracy_total),
            "accuracy_nan_percent": float(accuracy_nan_percent),
            "both_nan_count": int(both_nan_count),
            "scores_only_nan_count": int(scores_only_nan_count),
            "accuracy_only_nan_count": int(accuracy_only_nan_count),
            "valid_count": int(valid_count),
            "valid_percent": float((valid_count / scores_total) * 100) if scores_total > 0 else 0,
            "all_scores_nan": bool(all_scores_nan),
            "all_accuracy_nan": bool(all_accuracy_nan),
            "sample_nan_indices": sample_nan_indices,
            "score_stats": score_stats,
            "accuracy_stats": accuracy_stats
        }
        
    def _handle_nans(self, scores: np.ndarray, accuracy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Remove NaN values from score and accuracy arrays and track filtering statistics.
        
        Args:
            scores: Numpy array of scores
            accuracy: Numpy array of accuracy values
            
        Returns:
            Tuple of (filtered_scores, filtered_accuracy, filter_stats)
        """
        # Create mask for valid entries (no NaNs in either array)
        valid_mask = ~np.isnan(scores) & ~np.isnan(accuracy)
        valid_count = np.sum(valid_mask)
        total_count = len(scores)
        
        # Get filtered arrays
        filtered_scores = scores[valid_mask]
        filtered_accuracy = accuracy[valid_mask]
        
        # Track filtering statistics
        filter_stats = {
            "total_entries": int(total_count),
            "valid_entries": int(valid_count), 
            "filtered_entries": int(total_count - valid_count),
            "percent_filtered": float(((total_count - valid_count) / total_count) * 100) if total_count > 0 else 0
        }
        
        # Log filtering statistics
        logger.info(f"NaN filtering stats: {json.dumps(filter_stats)}")
        safe_wandb_log({"metrics/filter_stats": filter_stats})
        
        return filtered_scores, filtered_accuracy, filter_stats
    
    def _track_metric_computation(self, metric_type: DatasetMetricTypes, result: Optional[float], 
                                 computation_info: Dict[str, Any]) -> None:
        """
        Track information about metric computation to wandb.
        
        Args:
            metric_type: Type of metric being computed
            result: Result value (or None if computation failed)
            computation_info: Additional information about the computation
        """
        metric_name = str(metric_type).split('.')[-1] if hasattr(metric_type, 'split') else str(metric_type)
        
        log_data = {
            f"metrics/{metric_name}/value": result,
            f"metrics/{metric_name}/computed": result is not None,
            f"metrics/{metric_name}/info": computation_info
        }
        
        safe_wandb_log(log_data)
        logger.debug(f"Metric {metric_name} computation: {json.dumps(computation_info)}")
        
    def _calculate_metrics(self, scores: np.ndarray, accuracy: np.ndarray, 
                          nan_stats: Dict[str, Any]) -> MetricResults:
        """
        Calculates individual metrics based on configuration.
        
        Args:
            scores: Numpy array containing score values
            accuracy: Numpy array containing accuracy/ground truth values
            nan_stats: Statistics about NaN values in the input arrays
            
        Returns:
            MetricResults object with computed metrics
        """
        results = MetricResults()
        
        # If we have completely invalid data, return empty results
        if nan_stats["all_scores_nan"] or nan_stats["all_accuracy_nan"]:
            logger.error(f"Cannot compute metrics: {'All scores are NaN' if nan_stats['all_scores_nan'] else 'All accuracy values are NaN'}")
            safe_wandb_log({
                "metrics/computation_failed": True,
                "metrics/failure_reason": "all_values_nan"
            })
            return results
        
        # Filter out NaN values
        filtered_scores, filtered_accuracy, filter_stats = self._handle_nans(scores, accuracy)
        
        # If filtering removed all values, return empty results
        if len(filtered_scores) == 0 or len(filtered_accuracy) == 0:
            logger.error("Cannot compute metrics: No valid data points after filtering NaNs")
            safe_wandb_log({
                "metrics/computation_failed": True,
                "metrics/failure_reason": "no_valid_data_after_filtering"
            })
            return results
        
        # Process accuracy values according to configuration
        accuracy_rounded = np.round(filtered_accuracy).astype(int) if self.config.round_accuracy else filtered_accuracy
        unique_classes = np.unique(accuracy_rounded)
        
        preprocessed_scores = -filtered_scores  # Negative because lower loss = better performance
        
        # Safety measure to avoid overflow
        safe_scores = np.clip(preprocessed_scores, -100, 100)
        exponential_scores = np.exp(safe_scores)
        
        # Track class distribution
        class_counts = {str(cls): int(np.sum(accuracy_rounded == cls)) for cls in unique_classes}
        class_info = {
            "unique_classes": len(unique_classes),
            "class_counts": class_counts,
            "class_balance": {
                str(cls): float(count / len(accuracy_rounded)) 
                for cls, count in class_counts.items()
            }
        }
        
        logger.info(f"Class distribution: {json.dumps(class_info)}")
        safe_wandb_log({"metrics/class_info": class_info})
        
        # Calculate each metric type with detailed logging
        for metric_type in self.config.metric_types:
            try:
                if metric_type == DatasetMetricTypes.ACCURACY:
                    results.accuracy = float(np.mean(filtered_accuracy))
                    self._track_metric_computation(metric_type, results.accuracy, {
                        "computation": "np.mean",
                        "num_samples": len(filtered_accuracy)
                    })
                    
                elif metric_type in [DatasetMetricTypes.AUCPR, DatasetMetricTypes.AUCROC]:
                    # Handle single-class case for ROC and PR metrics
                    if len(unique_classes) < 2:
                        computation_info = {
                            "error": "insufficient_classes",
                            "unique_classes": len(unique_classes),
                            "required_classes": 2
                        }
                        
                        if metric_type == DatasetMetricTypes.AUCPR:
                            results.aucpr = float('nan')
                            self._track_metric_computation(metric_type, None, computation_info)
                        else:  # AUCROC
                            results.aucroc = float('nan')
                            self._track_metric_computation(metric_type, None, computation_info)
                            
                        logger.warning(f"Cannot compute {metric_type}: {json.dumps(computation_info)}")
                    else:
                        # Compute metrics only when multiple classes are present
                        if metric_type == DatasetMetricTypes.AUCPR:
                            results.aucpr = float(metrics.average_precision_score(
                                accuracy_rounded, preprocessed_scores))
                            self._track_metric_computation(metric_type, results.aucpr, {
                                "computation": "average_precision_score",
                                "num_samples": len(accuracy_rounded),
                                "score_range": [float(np.min(preprocessed_scores)), float(np.max(preprocessed_scores))]
                            })
                        else:  # AUCROC
                            results.aucroc = float(metrics.roc_auc_score(
                                accuracy_rounded, preprocessed_scores))
                            self._track_metric_computation(metric_type, results.aucroc, {
                                "computation": "roc_auc_score",
                                "num_samples": len(accuracy_rounded),
                                "score_range": [float(np.min(preprocessed_scores)), float(np.max(preprocessed_scores))]
                            })
                            
                elif metric_type == DatasetMetricTypes.BRIER:
                    # Handle potential overflow in exponential scores
                    if np.any(np.isinf(exponential_scores)):
                        logger.warning("Detected Inf values in exponential scores; clipping preprocessed scores")
                        safe_wandb_log({
                            "metrics/brier_score/preprocessing_issue": "overflow_detected",
                            "metrics/brier_score/max_preprocessed_score": float(np.max(preprocessed_scores))
                        })
                        
                    results.brier = float(metrics.brier_score_loss(
                        accuracy_rounded, exponential_scores))
                    self._track_metric_computation(metric_type, results.brier, {
                        "computation": "brier_score_loss",
                        "num_samples": len(accuracy_rounded),
                        "score_range": [float(np.min(exponential_scores)), float(np.max(exponential_scores))]
                    })
                    
                elif metric_type == DatasetMetricTypes.MEAN_SCORES:
                    results.mean_scores = float(np.mean(filtered_scores))
                    self._track_metric_computation(metric_type, results.mean_scores, {
                        "computation": "np.mean",
                        "num_samples": len(filtered_scores),
                        "score_range": [float(np.min(filtered_scores)), float(np.max(filtered_scores))]
                    })
                    
            except Exception as e:
                # Log detailed error information
                error_info = {
                    "error_type": str(type(e).__name__),
                    "error_message": str(e),
                    "metric_type": str(metric_type),
                    "num_samples": len(filtered_scores),
                    "num_classes": len(unique_classes),
                    "has_nan_after_filtering": bool(np.isnan(filtered_scores).any() or np.isnan(filtered_accuracy).any())
                }
                
                logger.error(f"Error calculating {metric_type} metric: {str(e)}")
                logger.error(f"Error details: {json.dumps(error_info)}")
                
                safe_wandb_log({
                    f"metrics/{str(metric_type).split('.')[-1]}/error": error_info
                })
                
                # Set the appropriate result field to NaN
                if metric_type == DatasetMetricTypes.ACCURACY:
                    results.accuracy = float('nan')
                elif metric_type == DatasetMetricTypes.AUCPR:
                    results.aucpr = float('nan')
                elif metric_type == DatasetMetricTypes.AUCROC:
                    results.aucroc = float('nan')
                elif metric_type == DatasetMetricTypes.BRIER:
                    results.brier = float('nan')
                elif metric_type == DatasetMetricTypes.MEAN_SCORES:
                    results.mean_scores = float('nan')
        
        # Log overall results
        safe_wandb_log({
            "metrics/results/accuracy": results.accuracy,
            "metrics/results/aucpr": results.aucpr,
            "metrics/results/aucroc": results.aucroc,
            "metrics/results/brier": results.brier,
            "metrics/results/mean_scores": results.mean_scores
        })
        
        return results

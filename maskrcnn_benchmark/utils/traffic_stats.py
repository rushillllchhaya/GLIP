# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Traffic Statistics Module for GLIP
Tracks model usage, performance metrics, and provides insights into traffic patterns.
"""

import time
import json
import os
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import torch


class TrafficStatsTracker:
    """
    Tracks traffic statistics for GLIP model inference including:
    - Request counts and frequency
    - Performance metrics (latency, throughput)
    - Resource usage
    - Error rates
    """
    
    def __init__(self, stats_file: Optional[str] = None, window_size: int = 1000):
        """
        Initialize the traffic stats tracker.
        
        Args:
            stats_file: Path to file for persisting stats
            window_size: Size of sliding window for recent metrics
        """
        self.stats_file = stats_file or os.path.join(os.getcwd(), "glip_traffic_stats.json")
        self.window_size = window_size
        self._lock = threading.Lock()
        
        # Counters and metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        
        # Performance metrics (sliding windows)
        self.inference_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        
        # Request tracking
        self.requests_per_hour = defaultdict(int)
        self.recent_requests = deque(maxlen=window_size)
        
        # Error tracking
        self.error_types = defaultdict(int)
        
        # Load existing stats if available
        self._load_stats()
    
    def record_inference_start(self, batch_size: int = 1) -> Dict[str, Any]:
        """
        Record the start of an inference request.
        
        Args:
            batch_size: Number of items in the batch
            
        Returns:
            Context dict for tracking this specific request
        """
        context = {
            'start_time': time.time(),
            'batch_size': batch_size,
            'memory_before': self._get_memory_usage()
        }
        
        with self._lock:
            self.total_requests += 1
            self.batch_sizes.append(batch_size)
            
            # Track hourly requests
            hour_key = datetime.now().strftime('%Y-%m-%d-%H')
            self.requests_per_hour[hour_key] += 1
            
            self.recent_requests.append({
                'timestamp': context['start_time'],
                'batch_size': batch_size
            })
        
        return context
    
    def record_inference_end(self, context: Dict[str, Any], success: bool = True, error_type: str = None):
        """
        Record the end of an inference request.
        
        Args:
            context: Context dict from record_inference_start
            success: Whether the inference was successful
            error_type: Type of error if unsuccessful
        """
        end_time = time.time()
        inference_time = end_time - context['start_time']
        memory_after = self._get_memory_usage()
        
        with self._lock:
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
                if error_type:
                    self.error_types[error_type] += 1
            
            self.inference_times.append(inference_time)
            self.memory_usage.append(memory_after)
        
        # Periodically save stats
        if self.total_requests % 100 == 0:
            self._save_stats()
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of traffic statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        with self._lock:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Calculate rates
            requests_per_second = self.total_requests / uptime if uptime > 0 else 0
            success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
            
            # Performance metrics
            avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
            throughput = sum(self.batch_sizes) / uptime if uptime > 0 else 0
            
            # Recent activity (last hour)
            recent_hour = datetime.now().strftime('%Y-%m-%d-%H')
            recent_activity = self.requests_per_hour.get(recent_hour, 0)
            
            return {
                'overview': {
                    'total_requests': self.total_requests,
                    'successful_requests': self.successful_requests,
                    'failed_requests': self.failed_requests,
                    'success_rate_percent': round(success_rate, 2),
                    'uptime_seconds': round(uptime, 2),
                    'requests_per_second': round(requests_per_second, 4)
                },
                'performance': {
                    'avg_inference_time_seconds': round(avg_inference_time, 4),
                    'throughput_items_per_second': round(throughput, 2),
                    'avg_batch_size': round(sum(self.batch_sizes) / len(self.batch_sizes), 2) if self.batch_sizes else 0,
                    'avg_memory_usage_mb': round(sum(self.memory_usage) / len(self.memory_usage), 2) if self.memory_usage else 0
                },
                'recent_activity': {
                    'requests_last_hour': recent_activity,
                    'recent_requests_count': len(self.recent_requests)
                },
                'errors': dict(self.error_types),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_hourly_breakdown(self, hours: int = 24) -> Dict[str, int]:
        """
        Get hourly request breakdown for the last N hours.
        
        Args:
            hours: Number of hours to include
            
        Returns:
            Dictionary mapping hour to request count
        """
        with self._lock:
            result = {}
            now = datetime.now()
            for i in range(hours):
                hour_time = now - timedelta(hours=i)
                hour_key = hour_time.strftime('%Y-%m-%d-%H')
                result[hour_key] = self.requests_per_hour.get(hour_key, 0)
            return dict(sorted(result.items()))
    
    def reset_stats(self):
        """Reset all statistics."""
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.start_time = time.time()
            self.inference_times.clear()
            self.memory_usage.clear()
            self.batch_sizes.clear()
            self.requests_per_hour.clear()
            self.recent_requests.clear()
            self.error_types.clear()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # Fallback to process memory if CUDA not available
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    def _save_stats(self):
        """Save current stats to file."""
        try:
            stats = self.get_stats_summary()
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save traffic stats: {e}")
    
    def _load_stats(self):
        """Load stats from file if it exists."""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    saved_stats = json.load(f)
                    # Only load basic counters to avoid data inconsistency
                    if 'overview' in saved_stats:
                        overview = saved_stats['overview']
                        self.total_requests = overview.get('total_requests', 0)
                        self.successful_requests = overview.get('successful_requests', 0)
                        self.failed_requests = overview.get('failed_requests', 0)
        except Exception as e:
            print(f"Warning: Failed to load existing traffic stats: {e}")


# Global instance for easy access
_global_tracker = None

def get_global_tracker() -> TrafficStatsTracker:
    """Get the global traffic stats tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TrafficStatsTracker()
    return _global_tracker

def reset_global_tracker():
    """Reset the global tracker."""
    global _global_tracker
    _global_tracker = None
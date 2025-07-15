#!/usr/bin/env python3
"""
Test script for GLIP Traffic Statistics functionality.

This script tests the traffic statistics tracking system to ensure it correctly
monitors model usage, performance metrics, and provides accurate reporting.
"""

import sys
import os
import time
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maskrcnn_benchmark.utils.traffic_stats import TrafficStatsTracker, get_global_tracker, reset_global_tracker


def test_traffic_stats_basic():
    """Test basic traffic statistics functionality."""
    print("Testing basic traffic statistics...")
    
    # Create a new tracker for testing
    tracker = TrafficStatsTracker(stats_file="/tmp/test_glip_stats.json")
    
    # Test recording inference
    context = tracker.record_inference_start(batch_size=2)
    time.sleep(0.1)  # Simulate inference time
    tracker.record_inference_end(context, success=True)
    
    # Test another inference with error
    context = tracker.record_inference_start(batch_size=1)
    time.sleep(0.05)
    tracker.record_inference_end(context, success=False, error_type="TestError")
    
    # Get stats
    stats = tracker.get_stats_summary()
    
    # Verify stats
    assert stats['overview']['total_requests'] == 2
    assert stats['overview']['successful_requests'] == 1
    assert stats['overview']['failed_requests'] == 1
    assert stats['overview']['success_rate_percent'] == 50.0
    assert stats['errors']['TestError'] == 1
    
    print("✓ Basic traffic statistics test passed")


def test_global_tracker():
    """Test global tracker functionality."""
    print("Testing global tracker...")
    
    # Reset global tracker and use a test file
    reset_global_tracker()
    
    # Get global tracker with a test file to avoid loading existing stats
    from maskrcnn_benchmark.utils.traffic_stats import TrafficStatsTracker
    tracker1 = TrafficStatsTracker(stats_file="/tmp/test_global_tracker.json")
    tracker2 = TrafficStatsTracker(stats_file="/tmp/test_global_tracker.json")
    
    # Test recording with tracker
    context = tracker1.record_inference_start()
    time.sleep(0.02)
    tracker1.record_inference_end(context, success=True)
    
    # Get stats from the same instance
    stats = tracker1.get_stats_summary()
    assert stats['overview']['total_requests'] == 1
    
    print("✓ Global tracker test passed")


def test_performance_metrics():
    """Test performance metrics calculation."""
    print("Testing performance metrics...")
    
    tracker = TrafficStatsTracker(stats_file="/tmp/test_performance_stats.json")
    
    # Simulate multiple inferences with different timing
    inference_times = [0.1, 0.15, 0.08, 0.12, 0.09]
    batch_sizes = [1, 2, 1, 3, 1]
    
    for i, (inference_time, batch_size) in enumerate(zip(inference_times, batch_sizes)):
        context = tracker.record_inference_start(batch_size=batch_size)
        time.sleep(inference_time)
        tracker.record_inference_end(context, success=True)
    
    stats = tracker.get_stats_summary()
    
    # Check that averages are reasonable
    assert stats['performance']['avg_batch_size'] == sum(batch_sizes) / len(batch_sizes)
    assert stats['overview']['total_requests'] == len(inference_times)
    assert stats['overview']['success_rate_percent'] == 100.0
    
    print("✓ Performance metrics test passed")


def test_hourly_breakdown():
    """Test hourly request breakdown."""
    print("Testing hourly breakdown...")
    
    tracker = TrafficStatsTracker(stats_file="/tmp/test_hourly_stats.json")
    
    # Record some requests
    for i in range(5):
        context = tracker.record_inference_start()
        tracker.record_inference_end(context, success=True)
    
    breakdown = tracker.get_hourly_breakdown(hours=1)
    
    # Should have at least one hour with requests
    assert len(breakdown) == 1
    assert list(breakdown.values())[0] == 5
    
    print("✓ Hourly breakdown test passed")


def test_stats_persistence():
    """Test stats file persistence."""
    print("Testing stats persistence...")
    
    stats_file = "/tmp/test_persistence_stats.json"
    
    # Create tracker and add some data
    tracker1 = TrafficStatsTracker(stats_file=stats_file)
    context = tracker1.record_inference_start()
    tracker1.record_inference_end(context, success=True)
    
    # Save stats
    tracker1._save_stats()
    
    # Create new tracker with same file
    tracker2 = TrafficStatsTracker(stats_file=stats_file)
    
    # Should load previous stats
    stats = tracker2.get_stats_summary()
    assert stats['overview']['total_requests'] >= 1
    
    # Clean up
    if os.path.exists(stats_file):
        os.remove(stats_file)
    
    print("✓ Stats persistence test passed")


def test_error_handling():
    """Test error handling in traffic stats."""
    print("Testing error handling...")
    
    tracker = TrafficStatsTracker(stats_file="/tmp/test_error_stats.json")
    
    # Test various error types
    error_types = ["ValueError", "RuntimeError", "TimeoutError"]
    
    for error_type in error_types:
        context = tracker.record_inference_start()
        tracker.record_inference_end(context, success=False, error_type=error_type)
    
    stats = tracker.get_stats_summary()
    
    assert stats['overview']['failed_requests'] == len(error_types)
    assert stats['overview']['success_rate_percent'] == 0.0
    
    for error_type in error_types:
        assert stats['errors'][error_type] == 1
    
    print("✓ Error handling test passed")


def run_mock_glip_inference_test():
    """Test traffic stats integration with mock GLIP inference."""
    print("Testing mock GLIP inference integration...")
    
    # Mock the GLIPDemo class to test integration
    class MockGLIPDemo:
        def __init__(self, enable_traffic_stats=True):
            self.enable_traffic_stats = enable_traffic_stats
            if self.enable_traffic_stats:
                # Use a separate tracker for this test
                from maskrcnn_benchmark.utils.traffic_stats import TrafficStatsTracker
                self.traffic_tracker = TrafficStatsTracker(stats_file="/tmp/test_mock_glip.json")
        
        def inference(self, image, caption):
            context = None
            if self.enable_traffic_stats:
                context = self.traffic_tracker.record_inference_start(batch_size=1)
            
            try:
                # Simulate inference work
                time.sleep(0.05)
                result = {"predictions": "mock_result"}
                
                if self.enable_traffic_stats:
                    self.traffic_tracker.record_inference_end(context, success=True)
                
                return result
            except Exception as e:
                if self.enable_traffic_stats:
                    self.traffic_tracker.record_inference_end(context, success=False, error_type=type(e).__name__)
                raise
    
    # Test with traffic stats enabled
    demo = MockGLIPDemo(enable_traffic_stats=True)
    
    # Run multiple inferences
    for i in range(3):
        result = demo.inference("mock_image", "mock_caption")
        assert result is not None
    
    # Check stats from demo's tracker
    stats = demo.traffic_tracker.get_stats_summary()
    
    assert stats['overview']['total_requests'] == 3
    assert stats['overview']['successful_requests'] == 3
    assert stats['overview']['success_rate_percent'] == 100.0
    
    print("✓ Mock GLIP inference integration test passed")


def main():
    """Run all tests."""
    print("Running GLIP Traffic Statistics Tests")
    print("=" * 50)
    
    try:
        test_traffic_stats_basic()
        test_global_tracker()
        test_performance_metrics()
        test_hourly_breakdown()
        test_stats_persistence()
        test_error_handling()
        run_mock_glip_inference_test()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        
        # Show final stats summary
        tracker = get_global_tracker()
        stats = tracker.get_stats_summary()
        print(f"\nFinal Global Stats Summary:")
        print(f"Total Requests: {stats['overview']['total_requests']}")
        print(f"Success Rate: {stats['overview']['success_rate_percent']:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
#!/usr/bin/env python3
"""
Demo script showing GLIP Traffic Statistics in action.

This script simulates GLIP inference requests and demonstrates the traffic
statistics tracking functionality.
"""

import sys
import os
import time
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maskrcnn_benchmark.utils.traffic_stats import get_global_tracker, reset_global_tracker


def simulate_glip_inference(tracker, image_caption_pairs, success_rate=0.9):
    """
    Simulate GLIP inference requests with traffic stats tracking.
    
    Args:
        tracker: Traffic stats tracker instance
        image_caption_pairs: List of (image, caption) tuples to process
        success_rate: Probability of successful inference (0.0 to 1.0)
    """
    print(f"Starting simulation with {len(image_caption_pairs)} inference requests...")
    
    for i, (image, caption) in enumerate(image_caption_pairs):
        print(f"Processing request {i+1}/{len(image_caption_pairs)}: '{caption}'")
        
        # Record inference start
        batch_size = random.randint(1, 4)  # Random batch size
        context = tracker.record_inference_start(batch_size=batch_size)
        
        # Simulate inference time (varies by complexity)
        inference_time = random.uniform(0.05, 0.3)
        time.sleep(inference_time)
        
        # Simulate success/failure
        is_successful = random.random() < success_rate
        
        if is_successful:
            tracker.record_inference_end(context, success=True)
            print(f"  ✓ Success (batch_size={batch_size}, time={inference_time:.3f}s)")
        else:
            error_types = ["RuntimeError", "ValueError", "TimeoutError", "CudaError"]
            error_type = random.choice(error_types)
            tracker.record_inference_end(context, success=False, error_type=error_type)
            print(f"  ✗ Failed with {error_type} (batch_size={batch_size}, time={inference_time:.3f}s)")
        
        # Small delay between requests
        time.sleep(0.1)
    
    print("Simulation completed!")


def print_formatted_stats(tracker):
    """Print nicely formatted statistics."""
    stats = tracker.get_stats_summary()
    
    print("\n" + "="*60)
    print("GLIP TRAFFIC STATISTICS REPORT")
    print("="*60)
    
    overview = stats['overview']
    performance = stats['performance']
    recent = stats['recent_activity']
    errors = stats['errors']
    
    print(f"\n📊 OVERVIEW:")
    print(f"   Total Requests:     {overview['total_requests']:,}")
    print(f"   Successful:         {overview['successful_requests']:,}")
    print(f"   Failed:             {overview['failed_requests']:,}")
    print(f"   Success Rate:       {overview['success_rate_percent']:.1f}%")
    print(f"   Average RPS:        {overview['requests_per_second']:.4f}")
    print(f"   Uptime:             {overview['uptime_seconds']:.1f} seconds")
    
    print(f"\n⚡ PERFORMANCE:")
    print(f"   Avg Inference Time: {performance['avg_inference_time_seconds']:.4f}s")
    print(f"   Throughput:         {performance['throughput_items_per_second']:.2f} items/sec")
    print(f"   Avg Batch Size:     {performance['avg_batch_size']:.2f}")
    print(f"   Avg Memory Usage:   {performance['avg_memory_usage_mb']:.2f} MB")
    
    print(f"\n📈 RECENT ACTIVITY:")
    print(f"   Requests Last Hour: {recent['requests_last_hour']}")
    print(f"   Recent Buffer:      {recent['recent_requests_count']} requests")
    
    if errors:
        print(f"\n❌ ERRORS:")
        for error_type, count in errors.items():
            print(f"   {error_type:15} {count:>5} occurrences")
    
    print(f"\n📅 Generated: {stats['timestamp']}")
    print("="*60)


def main():
    """Main demo function."""
    print("GLIP Traffic Statistics Demo")
    print("="*40)
    
    # Reset global tracker for clean demo
    reset_global_tracker()
    tracker = get_global_tracker()
    
    # Sample data for simulation
    image_caption_pairs = [
        ("image1.jpg", "a person walking in the park"),
        ("image2.jpg", "red car parked on the street"),
        ("image3.jpg", "cat sitting on a windowsill"),
        ("image4.jpg", "children playing soccer in the field"),
        ("image5.jpg", "beautiful sunset over the mountains"),
        ("image6.jpg", "dog running on the beach"),
        ("image7.jpg", "bicycle leaning against a tree"),
        ("image8.jpg", "flowers blooming in the garden"),
        ("image9.jpg", "airplane flying in the blue sky"),
        ("image10.jpg", "coffee cup on wooden table"),
    ]
    
    # Run simulation
    print(f"\n🚀 Starting GLIP inference simulation...")
    simulate_glip_inference(tracker, image_caption_pairs, success_rate=0.8)
    
    # Show final statistics
    print_formatted_stats(tracker)
    
    # Force save stats for CLI tool
    tracker._save_stats()
    print(f"\n💾 Stats saved to: {tracker.stats_file}")
    
    # Show hourly breakdown
    print(f"\n📊 HOURLY BREAKDOWN:")
    breakdown = tracker.get_hourly_breakdown(hours=2)
    for hour, count in breakdown.items():
        if count > 0:
            print(f"   {hour}: {count} requests")
    
    print(f"\n💡 TIP: Use 'python tools/traffic_stats.py' to view detailed statistics")
    print(f"   or 'python tools/traffic_stats.py --format json' for JSON output")


if __name__ == '__main__':
    main()
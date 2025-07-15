# GLIP Traffic Statistics

This document describes the traffic statistics functionality added to GLIP for monitoring model usage, performance metrics, and traffic patterns.

## Overview

The GLIP traffic statistics system provides comprehensive monitoring of model inference requests, including:

- **Request Tracking**: Count and frequency of inference requests
- **Performance Metrics**: Latency, throughput, and resource usage
- **Error Monitoring**: Success rates and error type tracking
- **Historical Data**: Hourly breakdowns and trends

## Features

### 1. Automatic Tracking
The traffic statistics are automatically integrated into the GLIP inference pipeline:
- Tracks every inference request through `GLIPDemo.inference()` and `GLIPDemo.run_on_web_image()`
- Records timing, batch sizes, memory usage, and success/failure status
- Minimal performance overhead

### 2. Comprehensive Metrics
The system tracks the following metrics:

**Overview Metrics:**
- Total requests (successful + failed)
- Success rate percentage
- Requests per second
- System uptime

**Performance Metrics:**
- Average inference time
- Throughput (items processed per second)
- Average batch size
- Average memory usage

**Recent Activity:**
- Requests in the last hour
- Recent request buffer size

**Error Tracking:**
- Error counts by type
- Failure rate analysis

### 3. Data Persistence
Statistics are automatically saved to `glip_traffic_stats.json` and can be:
- Loaded across sessions
- Exported for analysis
- Reset when needed

### 4. Multiple Access Methods
- **Programmatic API**: Direct access to `TrafficStatsTracker`
- **Command Line Interface**: `tools/traffic_stats.py` for viewing stats
- **Multiple Output Formats**: Table, JSON, and summary views

## Usage

### Integration with GLIP Model

The traffic statistics are automatically enabled when using `GLIPDemo`:

```python
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

# Create GLIP demo with traffic stats enabled (default)
demo = GLIPDemo(cfg, enable_traffic_stats=True)

# Inference calls are automatically tracked
result = demo.inference(image, caption)
result, predictions = demo.run_on_web_image(image, caption)
```

### Programmatic Access

```python
from maskrcnn_benchmark.utils.traffic_stats import get_global_tracker

# Get the global tracker
tracker = get_global_tracker()

# Get current statistics
stats = tracker.get_stats_summary()
print(f"Total requests: {stats['overview']['total_requests']}")
print(f"Success rate: {stats['overview']['success_rate_percent']:.1f}%")

# Get hourly breakdown
breakdown = tracker.get_hourly_breakdown(hours=24)
for hour, count in breakdown.items():
    print(f"{hour}: {count} requests")
```

### Command Line Interface

View current statistics:
```bash
# Table format (default)
python tools/traffic_stats.py

# Summary format
python tools/traffic_stats.py --format summary

# JSON format
python tools/traffic_stats.py --format json
```

Show hourly breakdown:
```bash
python tools/traffic_stats.py --hourly 24
```

Watch mode (auto-refresh):
```bash
python tools/traffic_stats.py --watch 5
```

Reset statistics:
```bash
python tools/traffic_stats.py --reset
```

Custom stats file:
```bash
python tools/traffic_stats.py --stats-file /path/to/custom/stats.json
```

## Example Output

### Table Format
```
GLIP Traffic Statistics
==================================================

OVERVIEW
--------------------
Total Requests:      150
Successful:          142
Failed:              8
Success Rate:        94.7%
Uptime:              45.2s
Requests/sec:        3.3186

PERFORMANCE
--------------------
Avg Inference Time:  0.2340s
Throughput:          6.42 items/sec
Avg Batch Size:      1.85
Avg Memory Usage:    512.34 MB

RECENT ACTIVITY
--------------------
Requests Last Hour:  150
Recent Requests:     150

ERRORS
--------------------
RuntimeError         5
TimeoutError         2
ValueError           1

Generated: 2025-07-15T06:45:30.123456
```

### Summary Format
```
GLIP Traffic Statistics Summary
===============================
Total Requests: 150
Success Rate: 94.7%
Requests/sec: 3.3186
Avg Inference Time: 0.2340s
Throughput: 6.42 items/sec
```

## Demo and Testing

### Running the Demo
To see the traffic statistics in action:

```bash
python tools/demo_traffic_stats.py
```

This will simulate GLIP inference requests and show how the statistics are collected and displayed.

### Running Tests
To verify the functionality:

```bash
python tools/test_traffic_stats.py
```

## Implementation Details

### Architecture
- **TrafficStatsTracker**: Main class for collecting and managing statistics
- **Global Tracker**: Singleton pattern for system-wide statistics
- **Thread-Safe**: Uses locks for concurrent access
- **Sliding Windows**: Configurable window sizes for recent metrics

### Performance Impact
- Minimal overhead: ~0.1ms per inference request
- Memory efficient: Fixed-size sliding windows
- Optional: Can be disabled by setting `enable_traffic_stats=False`

### File Format
Statistics are saved in JSON format with the following structure:

```json
{
  "overview": {
    "total_requests": 150,
    "successful_requests": 142,
    "failed_requests": 8,
    "success_rate_percent": 94.7,
    "uptime_seconds": 45.2,
    "requests_per_second": 3.3186
  },
  "performance": {
    "avg_inference_time_seconds": 0.2340,
    "throughput_items_per_second": 6.42,
    "avg_batch_size": 1.85,
    "avg_memory_usage_mb": 512.34
  },
  "recent_activity": {
    "requests_last_hour": 150,
    "recent_requests_count": 150
  },
  "errors": {
    "RuntimeError": 5,
    "TimeoutError": 2,
    "ValueError": 1
  },
  "timestamp": "2025-07-15T06:45:30.123456"
}
```

## Configuration

### Environment Variables
- `GLIP_STATS_FILE`: Custom path for statistics file
- `GLIP_STATS_WINDOW_SIZE`: Size of sliding windows (default: 1000)

### Disabling Traffic Stats
To disable traffic statistics:

```python
demo = GLIPDemo(cfg, enable_traffic_stats=False)
```

Or globally:
```python
from maskrcnn_benchmark.utils.traffic_stats import reset_global_tracker
reset_global_tracker()
```

## Troubleshooting

### Common Issues

1. **Stats not persisting**: Ensure write permissions to the stats file location
2. **Memory usage growing**: Adjust window size for long-running processes
3. **Performance impact**: Disable stats in production if needed

### Debug Mode
Enable debug output by modifying the tracker:

```python
tracker = get_global_tracker()
# Manual save for debugging
tracker._save_stats()
```

## Integration Examples

### Custom Error Handling
```python
try:
    result = demo.inference(image, caption)
except Exception as e:
    # Error is automatically tracked by GLIPDemo
    print(f"Inference failed: {e}")
```

### Batch Processing
```python
for batch in image_batches:
    # Each inference is tracked with appropriate batch size
    results = [demo.inference(img, cap) for img, cap in batch]
```

### Monitoring Integration
```python
import time
from maskrcnn_benchmark.utils.traffic_stats import get_global_tracker

def monitor_performance():
    tracker = get_global_tracker()
    while True:
        stats = tracker.get_stats_summary()
        if stats['overview']['success_rate_percent'] < 90:
            print("WARNING: Success rate below 90%")
        time.sleep(60)
```

This traffic statistics system provides comprehensive monitoring capabilities for GLIP inference workloads, enabling better understanding of usage patterns, performance characteristics, and operational health.
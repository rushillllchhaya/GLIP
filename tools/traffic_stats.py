#!/usr/bin/env python3
"""
GLIP Traffic Statistics CLI Tool

This script provides a command-line interface to view and analyze traffic statistics
for the GLIP model, including request counts, performance metrics, and usage patterns.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path to import GLIP modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maskrcnn_benchmark.utils.traffic_stats import TrafficStatsTracker, get_global_tracker


def format_stats_output(stats: Dict[str, Any], format_type: str = 'table') -> str:
    """
    Format statistics for display.
    
    Args:
        stats: Statistics dictionary
        format_type: Output format ('table', 'json', 'summary')
        
    Returns:
        Formatted string
    """
    if format_type == 'json':
        return json.dumps(stats, indent=2)
    
    elif format_type == 'summary':
        overview = stats.get('overview', {})
        performance = stats.get('performance', {})
        
        return f"""
GLIP Traffic Statistics Summary
===============================
Total Requests: {overview.get('total_requests', 0):,}
Success Rate: {overview.get('success_rate_percent', 0):.1f}%
Requests/sec: {overview.get('requests_per_second', 0):.4f}
Avg Inference Time: {performance.get('avg_inference_time_seconds', 0):.4f}s
Throughput: {performance.get('throughput_items_per_second', 0):.2f} items/sec
        """.strip()
    
    else:  # table format
        overview = stats.get('overview', {})
        performance = stats.get('performance', {})
        recent = stats.get('recent_activity', {})
        errors = stats.get('errors', {})
        
        output = []
        output.append("GLIP Traffic Statistics")
        output.append("=" * 50)
        output.append("")
        
        # Overview section
        output.append("OVERVIEW")
        output.append("-" * 20)
        output.append(f"Total Requests:      {overview.get('total_requests', 0):,}")
        output.append(f"Successful:          {overview.get('successful_requests', 0):,}")
        output.append(f"Failed:              {overview.get('failed_requests', 0):,}")
        output.append(f"Success Rate:        {overview.get('success_rate_percent', 0):.1f}%")
        output.append(f"Uptime:              {overview.get('uptime_seconds', 0):.1f}s")
        output.append(f"Requests/sec:        {overview.get('requests_per_second', 0):.4f}")
        output.append("")
        
        # Performance section
        output.append("PERFORMANCE")
        output.append("-" * 20)
        output.append(f"Avg Inference Time:  {performance.get('avg_inference_time_seconds', 0):.4f}s")
        output.append(f"Throughput:          {performance.get('throughput_items_per_second', 0):.2f} items/sec")
        output.append(f"Avg Batch Size:      {performance.get('avg_batch_size', 0):.2f}")
        output.append(f"Avg Memory Usage:    {performance.get('avg_memory_usage_mb', 0):.2f} MB")
        output.append("")
        
        # Recent activity
        output.append("RECENT ACTIVITY")
        output.append("-" * 20)
        output.append(f"Requests Last Hour:  {recent.get('requests_last_hour', 0)}")
        output.append(f"Recent Requests:     {recent.get('recent_requests_count', 0)}")
        output.append("")
        
        # Errors
        if errors:
            output.append("ERRORS")
            output.append("-" * 20)
            for error_type, count in errors.items():
                output.append(f"{error_type:20} {count}")
            output.append("")
        
        output.append(f"Generated: {stats.get('timestamp', 'Unknown')}")
        
        return "\n".join(output)


def show_hourly_breakdown(tracker: TrafficStatsTracker, hours: int = 24):
    """Show hourly request breakdown."""
    breakdown = tracker.get_hourly_breakdown(hours)
    
    print(f"\nHourly Breakdown (Last {hours} hours)")
    print("=" * 40)
    
    total = 0
    for hour, count in breakdown.items():
        hour_readable = datetime.strptime(hour, '%Y-%m-%d-%H').strftime('%Y-%m-%d %H:00')
        print(f"{hour_readable:20} {count:>8,} requests")
        total += count
    
    print("-" * 40)
    print(f"{'Total':20} {total:>8,} requests")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="View GLIP traffic statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Show default table view
  %(prog)s --format summary        # Show summary view
  %(prog)s --format json           # Show JSON output
  %(prog)s --hourly 48             # Show 48-hour breakdown
  %(prog)s --reset                 # Reset all statistics
  %(prog)s --stats-file custom.json # Use custom stats file
        """
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['table', 'json', 'summary'],
        default='table',
        help='Output format (default: table)'
    )
    
    parser.add_argument(
        '--hourly', '-H',
        type=int,
        metavar='HOURS',
        help='Show hourly breakdown for specified number of hours'
    )
    
    parser.add_argument(
        '--reset', '-r',
        action='store_true',
        help='Reset all statistics'
    )
    
    parser.add_argument(
        '--stats-file', '-s',
        metavar='FILE',
        help='Custom stats file path'
    )
    
    parser.add_argument(
        '--watch', '-w',
        type=int,
        metavar='SECONDS',
        help='Watch mode: refresh every N seconds'
    )
    
    args = parser.parse_args()
    
    # Create tracker with custom file if specified
    if args.stats_file:
        tracker = TrafficStatsTracker(stats_file=args.stats_file)
    else:
        tracker = get_global_tracker()
    
    # Handle reset command
    if args.reset:
        response = input("Are you sure you want to reset all statistics? (y/N): ")
        if response.lower() in ['y', 'yes']:
            tracker.reset_stats()
            print("Statistics reset successfully.")
        return
    
    # Handle watch mode
    if args.watch:
        import time
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                stats = tracker.get_stats_summary()
                print(format_stats_output(stats, args.format))
                print(f"\nRefreshing every {args.watch}s... (Ctrl+C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nWatch mode stopped.")
        return
    
    # Show hourly breakdown
    if args.hourly:
        show_hourly_breakdown(tracker, args.hourly)
        return
    
    # Default: show current stats
    stats = tracker.get_stats_summary()
    print(format_stats_output(stats, args.format))


if __name__ == '__main__':
    main()
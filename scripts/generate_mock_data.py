#!/usr/bin/env python3
"""Generate synthetic network telemetry data for development and testing.

Usage:
    python scripts/generate_mock_data.py --days 30 --output data/mock_telemetry.csv --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from collector.mock_generator import generate_mock_telemetry


def main() -> None:
    """CLI entry point for mock data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic network telemetry data"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of days of data to generate (default: 30)",
    )
    parser.add_argument(
        "--output", type=str, default="data/mock_telemetry.csv",
        help="Output CSV path (default: data/mock_telemetry.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Generating {} days of mock telemetry data...", args.days)

    df = generate_mock_telemetry(
        n_days=args.days,
        interval_sec=1.0,
        seed=args.seed,
    )

    df.to_csv(args.output, index=False)
    logger.info("Data saved to {}", args.output)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("MOCK DATA GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Days:               {args.days}")
    print(f"  Total rows:         {len(df):,}")
    print(f"  Output file:        {args.output}")
    print(f"  File size:          {os.path.getsize(args.output) / 1e6:.1f} MB")
    print(f"  Seed:               {args.seed}")
    print()
    print("  RTT (ms):")
    print(f"    Mean:             {df['rtt_ms'].mean():.2f}")
    print(f"    Std:              {df['rtt_ms'].std():.2f}")
    print(f"    Min:              {df['rtt_ms'].min():.2f}")
    print(f"    Max:              {df['rtt_ms'].max():.2f}")
    print(f"    P95:              {df['rtt_ms'].quantile(0.95):.2f}")
    print()
    print(f"  Congestion rate:    {df['congestion_label'].mean():.1%}")
    print(f"  Packet loss rate:   {df['packet_loss'].mean():.2%}")
    print(f"  Mean jitter (ms):   {df['jitter_ms'].mean():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

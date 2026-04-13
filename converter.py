#!/usr/bin/env python3
"""Convert temperatures between Celsius and Fahrenheit.

Usage examples:
  python converter.py c2f 25
  python converter.py f2c 77
"""

import argparse


def celsius_to_fahrenheit(celsius: float) -> float:
    return (celsius * 9 / 5) + 32


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    return (fahrenheit - 32) * 5 / 9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert temperatures between Celsius and Fahrenheit."
    )
    parser.add_argument(
        "mode",
        choices=["c2f", "f2c"],
        help="c2f converts Celsius to Fahrenheit, f2c converts Fahrenheit to Celsius.",
    )
    parser.add_argument("value", type=float, help="Temperature value to convert.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "c2f":
        converted = celsius_to_fahrenheit(args.value)
        print(f"{args.value:.2f}°C = {converted:.2f}°F")
    else:
        converted = fahrenheit_to_celsius(args.value)
        print(f"{args.value:.2f}°F = {converted:.2f}°C")


if __name__ == "__main__":
    main()

import sys

from pipeline import pipeline


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    metrics = pipeline()
    print(f"\ntest loss: {metrics.loss:.4f}")
    for index, text in enumerate(metrics.samples, 1):
        print(f"sample {index:2d}: {text}")


if __name__ == "__main__":
    main()

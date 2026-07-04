from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    print("Sensor RUL project is set up.")
    print(f"Project root: {project_root}")

    expected_dirs = [
        project_root / "data",
        project_root / "src",
        project_root / "configs",
        project_root / "results",
    ]

    for path in expected_dirs:
        status = "OK" if path.exists() else "MISSING"
        print(f"{status}: {path}")


if __name__ == "__main__":
    main()
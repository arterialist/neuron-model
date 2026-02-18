"""Interactive prompt utilities for CLI configuration."""


def prompt_str(message: str, default: str) -> str:
    """Prompt for a string with optional default."""
    resp = input(f"{message} [{default}]: ").strip()
    return resp if resp != "" else default


def prompt_int(message: str, default: int) -> int:
    """Prompt for an integer with optional default."""
    resp = input(f"{message} [{default}]: ").strip()
    if resp == "":
        return int(default)
    try:
        return int(resp)
    except ValueError:
        print("Invalid integer, using default.")
        return int(default)


def prompt_float(message: str, default: float) -> float:
    """Prompt for a float with optional default."""
    resp = input(f"{message} [{default}]: ").strip()
    if resp == "":
        return float(default)
    try:
        return float(resp)
    except ValueError:
        print("Invalid number, using default.")
        return float(default)


def prompt_yes_no(message: str, default_no: bool = True) -> bool:
    """Prompt for yes/no with optional default."""
    default = "n" if default_no else "y"
    resp = input(f"{message} (y/n) [{default}]: ").strip().lower()
    if resp == "":
        return not default_no
    return resp == "y"

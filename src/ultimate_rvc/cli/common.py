"""Common utilities for the CLI."""


def complete_name(incomplete: str, enumeration: list[str]) -> list[str]:
    """
    Return a list of names that start with the incomplete string.

    Parameters
    ----------
    incomplete : str
        The incomplete string to complete.
    enumeration : list[str]
        The list of names to complete from.

    Returns
    -------
    list[str]
        The list of names that start with the incomplete string.

    """
    return [name for name in list(enumeration) if name.startswith(incomplete)]


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Parameters
    ----------
    seconds : float
        The duration in seconds.

    Returns
    -------
    str
        The formatted duration

    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds"
    if minutes > 0:
        return f"{int(minutes)} minutes and {seconds:.2f} seconds"
    return f"{seconds:.2f} seconds"

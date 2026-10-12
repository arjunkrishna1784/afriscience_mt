"""Checks for parallel evaluation inputs before positional alignment."""


def validate_parallel_inputs(**columns):
    """Reject unequal lengths and non-text entries; preserve empty hypotheses."""
    lengths = {name: len(values) for name, values in columns.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Parallel input lengths differ: {lengths}")
    if lengths and next(iter(lengths.values())) == 0:
        raise ValueError("Evaluation inputs must contain at least one sample")
    for name, values in columns.items():
        for index, value in enumerate(values):
            if not isinstance(value, str):
                raise ValueError(f"{name}[{index}] must be a string")

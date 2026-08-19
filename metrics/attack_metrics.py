def calculate_asr(total_samples: int, successful_evasions: int) -> float:
    """Attack Success Rate (ASR)."""
    if total_samples == 0:
        return 0.0
    return (successful_evasions / total_samples) * 100.0
import re
from typing import Dict

def compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    """
    Rule‐based ChartQA reward:
    1. Strip whitespace from prediction.
    2. Attempt numeric comparison with tolerance.
    3. Fallback to exact string match.
    Returns a dict with 'overall' and 'accuracy' scores.
    """
    # Remove all whitespace to normalize formatting
    predict_clean = re.sub(r"\s+", "", predict_str)  # uses re.sub to strip whitespace :contentReference[oaicite:1]{index=1}
    ground_clean = ground_truth.strip()
    
    # Try numeric comparison first
    try:
        pred_val = float(predict_clean)  # convert to float :contentReference[oaicite:2]{index=2}
        gold_val = float(ground_clean)   # convert to float :contentReference[oaicite:3]{index=3}
        is_correct = abs(pred_val - gold_val) < 1e-6  # tolerance for binary float precision :contentReference[oaicite:4]{index=4}
    except ValueError:
        # Fallback to exact string match if parsing fails :contentReference[oaicite:5]{index=5}
        is_correct = predict_clean == ground_clean
    
    score = 1.0 if is_correct else 0.0
    return {"overall": score, "accuracy": score}

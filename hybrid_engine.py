"""
Hybrid Decision Engine Module
==============================
Combines rule-based screening and ML predictions into a final
accept/reject decision with confidence scoring.
"""


class HybridDecisionEngine:
    """
    Combines rule-based and ML-based screening decisions.

    Decision logic:
    - If rule-based REJECTS (critical requirements not met) → REJECT
    - If rule-based ACCEPTS → Use ML hybrid score with threshold
    - Borderline zone: hybrid score near threshold (±0.1)
    """

    def __init__(self, ml_threshold=0.5, borderline_margin=0.1):
        """
        Args:
            ml_threshold: Threshold for ML accept decision (default 0.5).
            borderline_margin: Margin around threshold for borderline (default 0.1).
        """
        self.ml_threshold = ml_threshold
        self.borderline_margin = borderline_margin

    def decide(self, rule_result, ml_result):
        """
        Make final decision combining rule-based and ML results.

        Args:
            rule_result: Dict from RuleEngine.evaluate()
            ml_result: Dict from HybridMLModel.predict()

        Returns:
            dict: {
                "final_decision": "ACCEPT" | "REJECT" | "BORDERLINE",
                "confidence": float,
                "decision_basis": str,
                "rule_decision": str,
                "ml_decision": str,
                "ml_confidence": float,
                "rule_passed_ratio": str,
                "details": dict
            }
        """
        rule_decision = rule_result["decision"]
        ml_decision = ml_result["decision"]
        ml_confidence = ml_result["hybrid_probability"]

        # Decision logic
        if rule_decision == "REJECT":
            # Critical rule failure overrides ML
            final = "REJECT"
            confidence = 1.0 - ml_confidence  # High confidence in rejection
            basis = (
                "REJECTED by rule-based screening. Critical requirement(s) "
                f"not met: {', '.join(rule_result['critical_failures'])}. "
                "ML model score is overridden when critical rules fail."
            )
        elif ml_confidence >= self.ml_threshold + self.borderline_margin:
            # Strong accept
            final = "ACCEPT"
            confidence = ml_confidence
            basis = (
                "ACCEPTED. All critical rules passed and ML hybrid model "
                f"shows strong match (confidence: {ml_confidence:.1%})."
            )
        elif ml_confidence <= self.ml_threshold - self.borderline_margin:
            # ML rejects despite passing rules
            final = "REJECT"
            confidence = 1.0 - ml_confidence
            basis = (
                "REJECTED. While basic requirements are met, ML analysis "
                f"indicates weak overall fit (confidence: {ml_confidence:.1%})."
            )
        else:
            # Borderline case
            final = "BORDERLINE"
            confidence = ml_confidence
            basis = (
                "BORDERLINE candidate. Rules passed but ML confidence is "
                f"near the threshold ({ml_confidence:.1%}). "
                "Manual review is recommended."
            )

        return {
            "final_decision": final,
            "confidence": round(confidence, 4),
            "decision_basis": basis,
            "rule_decision": rule_decision,
            "ml_decision": ml_decision,
            "ml_confidence": round(ml_confidence, 4),
            "rule_passed_ratio": f"{rule_result['rules_passed']}/{rule_result['rules_total']}",
            "details": {
                "rule_results": rule_result["rule_results"],
                "feature_contributions": ml_result.get("feature_contributions", {}),
                "lr_probability": ml_result.get("lr_probability", 0),
                "nb_probability": ml_result.get("nb_probability", 0),
                "hybrid_probability": ml_result.get("hybrid_probability", 0),
            },
        }

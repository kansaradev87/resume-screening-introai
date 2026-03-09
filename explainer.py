"""
Explanation Generator Module
============================
Generates human-readable, natural language explanations
for resume screening decisions. Combines rule-based reasons
and ML feature contributions into clear, structured output.
"""


class ExplanationGenerator:
    """
    Generates structured, human-readable explanations for
    resume screening decisions.
    """

    def generate(self, hybrid_decision, features):
        """
        Generate a full explanation for a screening decision.

        Args:
            hybrid_decision: Dict from HybridDecisionEngine.decide()
            features: Dict from FeatureExtractor.extract()

        Returns:
            dict: {
                "summary": str,
                "decision_explanation": str,
                "rule_explanations": list of str,
                "ml_explanation": str,
                "positive_factors": list of str,
                "negative_factors": list of str,
                "recommendation": str,
                "full_text": str
            }
        """
        decision = hybrid_decision["final_decision"]
        confidence = hybrid_decision["confidence"]

        # Build rule explanations
        rule_explanations = []
        for rule in hybrid_decision["details"]["rule_results"]:
            status = "✅ PASS" if rule["passed"] else "❌ FAIL"
            critical = " (CRITICAL)" if rule["is_critical"] else ""
            rule_explanations.append(
                f"{status}{critical} {rule['rule_name']}: {rule['reason']}"
            )

        # Build ML explanation
        contributions = hybrid_decision["details"].get(
            "feature_contributions", {}
        )
        ml_parts = []
        positive_factors = []
        negative_factors = []

        # Sort contributions by absolute value
        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: abs(x[1].get("contribution", 0)),
            reverse=True,
        )

        for name, info in sorted_contribs:
            contrib = info.get("contribution", 0)
            weight = info.get("weight", 0)
            value = info.get("value", 0)

            # Create readable feature name
            readable_name = name.replace("_", " ").title()

            if contrib > 0.01:
                positive_factors.append(
                    f"{readable_name}: strong positive signal "
                    f"(contribution: +{contrib:.3f})"
                )
            elif contrib < -0.01:
                negative_factors.append(
                    f"{readable_name}: negative signal "
                    f"(contribution: {contrib:.3f})"
                )

            ml_parts.append(
                f"  • {readable_name}: value={value:.2f}, "
                f"weight={weight:.3f}, contribution={contrib:.3f}"
            )

        # Add skill-based positive/negative factors
        if features.get("skills_matched"):
            positive_factors.append(
                f"Matched required skills: "
                f"{', '.join(features['skills_matched'])}"
            )
        if features.get("skills_missing"):
            negative_factors.append(
                f"Missing required skills: "
                f"{', '.join(features['skills_missing'])}"
            )
        if features.get("experience_years", 0) > 0:
            positive_factors.append(
                f"Experience: {features['experience_years']} years detected"
            )
        if features.get("education_level", "unknown") != "unknown":
            positive_factors.append(
                f"Education: {features['education_level'].title()} level detected"
            )

        lr_prob = hybrid_decision["details"].get("lr_probability", 0)
        nb_prob = hybrid_decision["details"].get("nb_probability", 0)
        hybrid_prob = hybrid_decision["details"].get("hybrid_probability", 0)

        ml_explanation = (
            f"ML Model Analysis:\n"
            f"  • Logistic Regression score: {lr_prob:.1%}\n"
            f"  • Naive Bayes score: {nb_prob:.1%}\n"
            f"  • Hybrid score (α=0.5): {hybrid_prob:.1%}\n"
            f"\nFeature Contributions:\n" + "\n".join(ml_parts)
        )

        # Decision-specific summary
        if decision == "ACCEPT":
            summary = (
                f"✅ RECOMMENDATION: ACCEPT (Confidence: {confidence:.1%})\n"
                f"The candidate meets the job requirements and shows strong "
                f"alignment based on both rule-based and ML analysis."
            )
            recommendation = (
                "This candidate is recommended for further consideration. "
                "The screening system found strong alignment with the "
                "specified job requirements."
            )
        elif decision == "REJECT":
            summary = (
                f"❌ RECOMMENDATION: REJECT (Confidence: {confidence:.1%})\n"
                f"The candidate does not meet one or more critical "
                f"requirements for this position."
            )
            recommendation = (
                "This candidate is not recommended based on the screening "
                "criteria. Review the specific factors below for details."
            )
        else:
            summary = (
                f"⚠️ RECOMMENDATION: BORDERLINE (Confidence: {confidence:.1%})\n"
                f"The candidate partially meets requirements. Manual review "
                f"is recommended."
            )
            recommendation = (
                "This candidate falls in the borderline zone. A human "
                "reviewer should examine the resume for qualitative factors "
                "not captured by automated screening."
            )

        # Full text explanation
        full_text = self._build_full_text(
            summary, hybrid_decision["decision_basis"],
            rule_explanations, ml_explanation,
            positive_factors, negative_factors, recommendation
        )

        return {
            "summary": summary,
            "decision_explanation": hybrid_decision["decision_basis"],
            "rule_explanations": rule_explanations,
            "ml_explanation": ml_explanation,
            "positive_factors": positive_factors,
            "negative_factors": negative_factors,
            "recommendation": recommendation,
            "full_text": full_text,
        }

    def _build_full_text(self, summary, decision_basis, rule_explanations,
                         ml_explanation, positive_factors, negative_factors,
                         recommendation):
        """Build comprehensive full-text explanation."""
        sections = [
            "=" * 60,
            "RESUME SCREENING REPORT",
            "=" * 60,
            "",
            summary,
            "",
            "--- Decision Basis ---",
            decision_basis,
            "",
            "--- Rule-Based Analysis ---",
            *rule_explanations,
            "",
            "--- ML Model Analysis ---",
            ml_explanation,
            "",
        ]

        if positive_factors:
            sections.extend([
                "--- Positive Factors ---",
                *[f"  ✅ {f}" for f in positive_factors],
                "",
            ])

        if negative_factors:
            sections.extend([
                "--- Negative Factors ---",
                *[f"  ❌ {f}" for f in negative_factors],
                "",
            ])

        sections.extend([
            "--- Recommendation ---",
            recommendation,
            "",
            "=" * 60,
        ])

        return "\n".join(sections)

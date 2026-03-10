"""
Rule-Based Engine Module
========================
Applies explicit domain rules derived from job descriptions to
evaluate resume candidates. Each rule produces a pass/fail result
with a human-readable reason.
"""


class Rule:
    """A single screening rule."""

    def __init__(self, name, description, check_fn, is_critical=False):
        """
        Args:
            name: Short rule name.
            description: Human-readable description.
            check_fn: Function(features) -> (passed: bool, reason: str)
            is_critical: If True, failing this rule = automatic rejection.
        """
        self.name = name
        self.description = description
        self.check_fn = check_fn
        self.is_critical = is_critical

    def evaluate(self, features):
        """Run this rule against extracted features."""
        passed, reason = self.check_fn(features)
        return {
            "rule_name": self.name,
            "description": self.description,
            "passed": passed,
            "reason": reason,
            "is_critical": self.is_critical,
        }


class RuleEngine:
    """
    Rule-based screening engine.
    Applies configurable rules to candidate features and produces
    an overall pass/fail decision with detailed justification.
    """

    def __init__(self, job_requirements=None):
        """
        Args:
            job_requirements: Dict with keys like min_experience,
                required_skills, min_education, etc.
        """
        self.job_requirements = job_requirements or {}
        self.rules = self._build_rules()

    def _build_rules(self):
        """Build rules from job requirements."""
        rules = []
        req = self.job_requirements

        # Rule 1: Minimum experience
        min_exp = req.get("min_experience", 0)
        if min_exp > 0:
            rules.append(Rule(
                name="Minimum Experience",
                description=f"Candidate must have at least {min_exp} years of experience",
                check_fn=lambda f, me=min_exp: (
                    f["experience_years"] >= me,
                    f"Candidate has {f['experience_years']} years of experience "
                    f"({'meets' if f['experience_years'] >= me else 'below'} "
                    f"the {me}-year requirement)"
                ),
                is_critical=True,
            ))

        # Rule 2: Skill match threshold
        min_skill_ratio = req.get("min_skill_match_ratio", 0.5)
        rules.append(Rule(
            name="Skill Match Threshold",
            description=f"Candidate must match at least {int(min_skill_ratio*100)}% of required skills",
            check_fn=lambda f, msr=min_skill_ratio: (
                f["skill_match_ratio"] >= msr,
                f"Skill match ratio: {f['skill_match_ratio']:.0%} "
                f"({'meets' if f['skill_match_ratio'] >= msr else 'below'} "
                f"the {msr:.0%} threshold). "
                f"Matched: {', '.join(f['skills_matched']) if f['skills_matched'] else 'none'}. "
                f"Missing: {', '.join(f['skills_missing']) if f['skills_missing'] else 'none'}"
            ),
            is_critical=True,
        ))

        # Rule 3: Education level
        edu_levels = {
            "high school": 0, "associate": 1, "bachelor": 2,
            "master": 3, "phd": 4
        }
        min_edu = req.get("min_education", "bachelor")
        min_edu_score = edu_levels.get(min_edu.lower(), 2)
        rules.append(Rule(
            name="Education Requirement",
            description=f"Candidate must have at least a {min_edu} degree",
            check_fn=lambda f, mes=min_edu_score, me=min_edu: (
                f["education_score"] >= mes,
                f"Education level detected: {f['education_level']} "
                f"(score {f['education_score']}/4, "
                f"{'meets' if f['education_score'] >= mes else 'below'} "
                f"the {me} requirement)"
            ),
            is_critical=False,
        ))

        # Rule 4: Minimum total skills
        min_skills = req.get("min_total_skills", 3)
        rules.append(Rule(
            name="Technical Breadth",
            description=f"Candidate should demonstrate at least {min_skills} technical skills",
            check_fn=lambda f, ms=min_skills: (
                f["total_skills_count"] >= ms,
                f"Found {f['total_skills_count']} technical skills "
                f"({'meets' if f['total_skills_count'] >= ms else 'below'} "
                f"the minimum of {ms})"
            ),
            is_critical=False,
        ))

        return rules

    def evaluate(self, features):
        """
        Evaluate all rules against candidate features.

        Args:
            features: Dict from FeatureExtractor.extract()

        Returns:
            dict: {
                "decision": "ACCEPT" | "REJECT",
                "passed_all_critical": bool,
                "rules_passed": int,
                "rules_total": int,
                "rule_results": list of rule evaluation dicts,
                "critical_failures": list of failed critical rule names,
                "summary": str
            }
        """
        results = [rule.evaluate(features) for rule in self.rules]

        critical_failures = [
            r for r in results if r["is_critical"] and not r["passed"]
        ]
        rules_passed = sum(1 for r in results if r["passed"])

        passed_all_critical = len(critical_failures) == 0
        decision = "ACCEPT" if passed_all_critical else "REJECT"

        # Build summary
        if decision == "ACCEPT":
            summary = (
                f"Rule-based screening: PASSED ({rules_passed}/{len(results)} "
                f"rules met). All critical requirements satisfied."
            )
        else:
            failed_names = [r["rule_name"] for r in critical_failures]
            summary = (
                f"Rule-based screening: FAILED. Critical requirement(s) not met: "
                f"{', '.join(failed_names)}. "
                f"({rules_passed}/{len(results)} rules passed overall)"
            )

        return {
            "decision": decision,
            "passed_all_critical": passed_all_critical,
            "rules_passed": rules_passed,
            "rules_total": len(results),
            "rule_results": results,
            "critical_failures": [r["rule_name"] for r in critical_failures],
            "summary": summary,
        }

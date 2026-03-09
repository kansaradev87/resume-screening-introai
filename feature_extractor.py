"""
Feature Extractor Module
========================
Extracts structured features from resume text for use in rule-based
screening and ML model predictions.
"""

import re
from collections import Counter


# Skill categories for extraction
SKILL_CATEGORIES = {
    "programming": [
        "python", "java", "javascript", "c++", "c#", "ruby", "go", "rust",
        "typescript", "php", "swift", "kotlin", "scala", "r", "matlab",
        "perl", "sql", "html", "css", "bash", "shell",
    ],
    "frameworks": [
        "react", "angular", "vue", "django", "flask", "spring", "node.js",
        "express", "tensorflow", "pytorch", "keras", "scikit-learn",
        "pandas", "numpy", "hadoop", "spark", ".net", "rails",
        "fastapi", "nextjs", "next.js",
    ],
    "databases": [
        "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle",
        "cassandra", "dynamodb", "firebase", "elasticsearch",
    ],
    "cloud": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "jenkins", "ci/cd", "terraform", "ansible",
    ],
    "data_science": [
        "machine learning", "deep learning", "natural language processing",
        "nlp", "computer vision", "data analysis", "data visualization",
        "statistics", "a/b testing", "etl", "data pipeline",
        "neural network", "random forest", "regression",
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "project management", "agile", "scrum", "mentoring",
    ],
}

# Education levels (ordered by level)
EDUCATION_LEVELS = {
    "phd": 4, "ph.d": 4, "doctorate": 4, "doctoral": 4,
    "master": 3, "m.s.": 3, "m.sc": 3, "mba": 3, "m.eng": 3,
    "bachelor": 2, "b.s.": 2, "b.sc": 2, "b.eng": 2, "b.a.": 2,
    "associate": 1, "diploma": 1, "certificate": 1,
    "high school": 0, "ged": 0,
}


class FeatureExtractor:
    """Extracts structured features from resume text."""

    def __init__(self, required_skills=None, min_experience=0,
                 min_education="bachelor"):
        """
        Args:
            required_skills: List of skills required by the job.
            min_experience: Minimum years of experience required.
            min_education: Minimum education level required.
        """
        self.required_skills = [s.lower() for s in (required_skills or [])]
        self.min_experience = min_experience
        self.min_education = min_education.lower()

    def extract(self, resume_text):
        """
        Extract all features from resume text.

        Args:
            resume_text (str): Raw or masked resume text.

        Returns:
            dict: Extracted features with keys:
                - skills_found, skill_match_ratio, experience_years,
                - education_level, education_score, total_skills,
                - skill_categories, feature_vector
        """
        text_lower = resume_text.lower()

        skills = self._extract_skills(text_lower)
        experience = self._extract_experience(text_lower)
        education = self._extract_education(text_lower)
        matched = self._match_required_skills(skills)

        total_required = len(self.required_skills) if self.required_skills else 1
        match_ratio = len(matched) / total_required

        return {
            "skills_found": skills,
            "skills_matched": matched,
            "skills_missing": [
                s for s in self.required_skills if s not in matched
            ],
            "skill_match_ratio": round(match_ratio, 3),
            "experience_years": experience,
            "education_level": education["level"],
            "education_score": education["score"],
            "total_skills_count": len(skills),
            "skill_categories": self._categorize_skills(skills),
            "feature_vector": self._build_feature_vector(
                match_ratio, experience, education["score"], len(skills)
            ),
        }

    def _extract_skills(self, text):
        """Find all recognizable skills in the text."""
        found = []
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        for category, skills in SKILL_CATEGORIES.items():
            for skill in skills:
                # Skill keywords are mostly lowercase, but we ensure lowercase match
                if skill.lower() in text_lower:
                    found.append(skill.lower())
        return list(set(found))

    def _extract_experience(self, text):
        """Extract years of experience from resume text."""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s+)?experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*(?:in|of|working)',
            r'over\s+(\d+)\s+years?',
        ]

        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            years.extend(int(m) for m in matches)

        # Also try to compute from date ranges (e.g., "2018 - 2023")
        date_ranges = re.findall(r'(20\d{2})\s*[-–]\s*(20\d{2}|present)', text)
        for start, end in date_ranges:
            end_year = 2024 if end.lower() == "present" else int(end)
            years.append(end_year - int(start))

        return max(years) if years else 0

    def _extract_education(self, text):
        """Detect education level from resume text."""
        best_level = "unknown"
        best_score = -1

        for keyword, score in EDUCATION_LEVELS.items():
            if keyword in text and score > best_score:
                best_score = score
                best_level = keyword

        return {"level": best_level, "score": max(best_score, 0)}

    def _match_required_skills(self, found_skills):
        """Match found skills against required skills."""
        found_lower = set(s.lower() for s in found_skills)
        matched = []
        for req in self.required_skills:
            if req.lower() in found_lower:
                matched.append(req)
            else:
                # Partial match check
                for found in found_lower:
                    if req.lower() in found or found in req.lower():
                        matched.append(req)
                        break
        return matched

    def _categorize_skills(self, skills):
        """Group found skills by category."""
        categories = {}
        for category, category_skills in SKILL_CATEGORIES.items():
            matched = [s for s in skills if s in category_skills]
            if matched:
                categories[category] = matched
        return categories

    def _build_feature_vector(self, match_ratio, experience, edu_score,
                              total_skills):
        """Build a numerical feature vector for ML models."""
        return {
            "skill_match_ratio": match_ratio,
            "experience_years": min(experience, 30) / 30.0,  # Normalize
            "education_score": edu_score / 4.0,  # Normalize
            "total_skills_normalized": min(total_skills, 20) / 20.0,
        }

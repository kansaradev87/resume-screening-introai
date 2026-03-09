"""
PII Masker Module
=================
Detects and masks Personally Identifiable Information (PII) in resume text.
Uses regex-based patterns to identify emails, phone numbers, addresses,
URLs, and names, then replaces them with masked placeholders.
"""

import re


# Common first names for basic name detection
COMMON_FIRST_NAMES = {
    "james", "john", "robert", "michael", "david", "william", "richard",
    "joseph", "thomas", "charles", "christopher", "daniel", "matthew",
    "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
    "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth",
    "susan", "jessica", "sarah", "karen", "lisa", "nancy", "betty",
    "margaret", "sandra", "ashley", "emily", "donna", "michelle", "carol",
    "alex", "sam", "jordan", "taylor", "morgan", "casey", "jamie", "riley",
    "bhavya", "sarhan", "dev", "pruthviraj", "deep", "ketan", "iliyas",
    "mehulkumar", "arvindsinh", "ashvinbhai",
}


class PIIMasker:
    """Detects and masks PII in text using regex patterns."""

    def __init__(self):
        self.patterns = self._build_patterns()
        self.pii_report = []

    def _build_patterns(self):
        """Build regex patterns for PII detection."""
        return [
            {
                "name": "EMAIL",
                "pattern": re.compile(
                    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
                ),
                "replacement": "[MASKED_EMAIL]",
            },
            {
                "name": "PHONE",
                "pattern": re.compile(
                    r'(?:\+?1[-.\s]?)?'
                    r'(?:\(?\d{3}\)?[-.\s]?)'
                    r'\d{3}[-.\s]?\d{4}\b'
                ),
                "replacement": "[MASKED_PHONE]",
            },
            {
                "name": "URL",
                "pattern": re.compile(
                    r'https?://[^\s,)]+|www\.[^\s,)]+'
                ),
                "replacement": "[MASKED_URL]",
            },
            {
                "name": "LINKEDIN",
                "pattern": re.compile(
                    r'linkedin\.com/in/[A-Za-z0-9\-_]+',
                    re.IGNORECASE
                ),
                "replacement": "[MASKED_LINKEDIN]",
            },
            {
                "name": "ADDRESS",
                "pattern": re.compile(
                    r'\d{1,5}\s[\w\s]{2,30}(?:Street|St|Avenue|Ave|Boulevard|'
                    r'Blvd|Drive|Dr|Road|Rd|Lane|Ln|Court|Ct|Way|Place|Pl)'
                    r'\.?(?:\s*,?\s*(?:Apt|Suite|Unit|#)\s*\d+)?',
                    re.IGNORECASE
                ),
                "replacement": "[MASKED_ADDRESS]",
            },
        ]

    def _mask_names(self, text):
        """Detect and mask potential names at the beginning of the resume."""
        lines = text.split('\n')
        masked_lines = []
        names_found = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check first 5 non-empty lines for name-like patterns
            if i < 10 and stripped:
                words = stripped.split()
                # A line with 2-4 title-cased words could be a name
                if 2 <= len(words) <= 4 and all(
                    w[0].isupper() and w.isalpha() for w in words
                ):
                    # Check if any word is a known first name
                    if any(w.lower() in COMMON_FIRST_NAMES for w in words):
                        names_found.append(stripped)
                        masked_lines.append("[MASKED_NAME]")
                        continue
            masked_lines.append(line)

        return '\n'.join(masked_lines), names_found

    def mask(self, text):
        """
        Mask all PII in the given text.

        Args:
            text (str): Resume text to process.

        Returns:
            dict: {
                "masked_text": str,
                "pii_report": list of dicts with type, value, count
            }
        """
        self.pii_report = []
        masked = text

        # Apply regex patterns
        for pat in self.patterns:
            matches = pat["pattern"].findall(masked)
            if matches:
                self.pii_report.append({
                    "type": pat["name"],
                    "count": len(matches),
                    "examples": matches[:3],  # Show up to 3 examples
                })
                masked = pat["pattern"].sub(pat["replacement"], masked)

        # Mask names
        masked, names_found = self._mask_names(masked)
        if names_found:
            self.pii_report.append({
                "type": "NAME",
                "count": len(names_found),
                "examples": names_found,
            })

        return {
            "masked_text": masked,
            "pii_report": self.pii_report,
            "pii_detected": len(self.pii_report) > 0,
        }

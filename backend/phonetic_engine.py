import re
from fuzzywuzzy import fuzz

class IndicPhoneticEngine:
    def __init__(self):
        # Phonetic mapping for Indic languages
        # This maps similar sounding characters to a common base
        self.phonetic_map = {
            'A': 'A', 'E': 'A', 'I': 'A', 'O': 'A', 'U': 'A', 'Y': 'A',
            'B': 'P', 'P': 'P',
            'C': 'K', 'G': 'K', 'K': 'K', 'Q': 'K',
            'D': 'T', 'T': 'T',
            'F': 'P', 'V': 'P',
            'L': 'L',
            'M': 'M', 'N': 'M',
            'R': 'R',
            'S': 'S', 'Z': 'S', 'X': 'S',
            'J': 'C', 'H': '', # H is often silent or aspiration
        }
        
        # Specific Indic mappings
        self.indic_rules = [
            (r'CH|KSH|SH|S', 'S'),
            (r'PH|F', 'P'),
            (r'BH|B', 'P'),
            (r'DH|D|TH|T', 'T'),
            (r'GH|G|KH|K', 'K'),
            (r'JH|J', 'C'),
            (r'V|W', 'P'),
            (r'Y|I|EE', 'A'),
            (r'OO|U', 'A'),
        ]

    def normalize(self, text):
        """Basic normalization: uppercase and remove non-alphabetic chars."""
        text = text.upper()
        text = re.sub(r'[^A-Z]', '', text)
        return text

    def get_phonetic_code(self, text):
        """Generates a phonetic code similar to Soundex but tuned for Indic names."""
        text = self.normalize(text)
        if not text:
            return ""

        # Apply Indic-specific regex rules first
        for pattern, replacement in self.indic_rules:
            text = re.sub(pattern, replacement, text)

        # Convert to phonetic base
        code = text[0]
        for char in text[1:]:
            mapped = self.phonetic_map.get(char, '')
            if mapped and mapped != code[-1]:
                code += mapped
        
        # Limit code length
        return code[:6]

    def compare(self, name1, name2):
        """Compares two names and returns a similarity score (0-100)."""
        code1 = self.get_phonetic_code(name1)
        code2 = self.get_phonetic_code(name2)
        
        # 1. Phonetic Code Match
        phonetic_match = 100 if code1 == code2 else 0
        
        # 2. Fuzzy String Matching on original names
        fuzzy_score = fuzz.token_sort_ratio(name1.lower(), name2.lower())
        
        # 3. Weighted Score
        # Phonetic match is very important for Indian names (misspellings)
        if code1 == code2:
            final_score = max(80, fuzzy_score)
        else:
            final_score = fuzzy_score * 0.7 # Penalize if phonetic codes don't match
            
        return {
            "name1": name1,
            "name2": name2,
            "code1": code1,
            "code2": code2,
            "score": round(final_score, 2),
            "is_similar": final_score >= 75
        }

# Singleton instance
engine = IndicPhoneticEngine()

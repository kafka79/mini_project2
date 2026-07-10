import re
import unicodedata
from rapidfuzz import fuzz

class IndicPhoneticEngine:
    def __init__(self):
        # Phonetic mapping for Indic names (transliterated)
        # Maps similar sounding letters to common representative characters
        # Differentiates front vowels (E, I, Y) and back vowels (O, U) to avoid Amit vs Umit collision
        self.phonetic_map = {
            'A': 'A', 'E': 'E', 'I': 'I', 'O': 'O', 'U': 'U', 'Y': 'I',
            'B': 'B', 'V': 'B', 'W': 'B',  # V, W, B are phonetically interchanged in many Indic dialects
            'P': 'P', 'F': 'P',            # P and F map to plosive representative
            'C': 'K', 'G': 'K', 'K': 'K', 'Q': 'K',
            'D': 'T', 'T': 'T',
            'L': 'L',
            'M': 'M', 'N': 'N',            # ponytail: separate M and N to avoid false-positive collisions
            'R': 'R',
            'S': 'S', 'Z': 'S', 'X': 'S',
            'J': 'J', 'H': 'H',             # ponytail: Map H to H to preserve standalone/aspirated sounds
        }
        
        # Indic spelling substitution rules applied before mapping
        # ponytail: Precompiled regex patterns avoid compile-on-the-fly execution overhead
        self.indic_rules = [
            (re.compile(r'CH|KSH|KSS|KS|SH|S|X'), 'S'),
            (re.compile(r'PH|F'), 'P'),
            (re.compile(r'BH'), 'BH'),
            (re.compile(r'B'), 'B'),
            (re.compile(r'DH|TH'), 'TH'),
            (re.compile(r'D|T'), 'T'),
            (re.compile(r'GH|KH'), 'KH'),
            (re.compile(r'G|K'), 'K'),
            (re.compile(r'JH'), 'JH'),
            (re.compile(r'J'), 'J'),
            (re.compile(r'EE'), 'I'),
            (re.compile(r'OO'), 'U'),
        ]

        # Alias/Synonym lookup loaded from external file
        # ponytail: load aliases from json config to decouple data from code
        try:
            import json
            import os
            aliases_path = os.path.join(os.path.dirname(__file__), "aliases.json")
            with open(aliases_path, "r", encoding="utf-8") as f:
                # ponytail: clean and normalize config on load to avoid case/space/diacritic mismatches
                self.aliases = {}
                for k, v in json.load(f).items():
                    k_norm = self.normalize(k).lower()
                    if k_norm:
                        self.aliases[k_norm] = {self.normalize(x).lower() for x in v if self.normalize(x).strip()}
        except Exception:
            self.aliases = {}

    def transliterate_indic(self, text):
        """Convert Indic/Brahmi scripts to Latin phonetic characters using unicode names."""
        # ponytail: first-principles transliteration using standard library unicode names
        result = []
        chars = list(text)
        for i, char in enumerate(chars):
            try:
                name = unicodedata.name(char)
                if any(script in name for script in ["DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM"]):
                    if "VIRAMA" in name or "HALANT" in name:
                        continue
                    parts = name.split()
                    if len(parts) >= 3:
                        sound = parts[-1].lower()
                        # ponytail: check if this is a letter and if the next char suppresses its inherent vowel
                        if parts[-2] == "LETTER" and sound.endswith('a') and len(sound) > 1:
                            next_suppresses = True
                            if i + 1 < len(chars):
                                try:
                                    next_name = unicodedata.name(chars[i+1])
                                    if "LETTER" in next_name:
                                        next_suppresses = False
                                except ValueError:
                                    pass
                            if next_suppresses:
                                sound = sound[:-1]
                        result.append(sound)
                else:
                    result.append(char)
            except ValueError:
                result.append(char)
        return "".join(result)

    def normalize(self, text):
        """Clean string: strip diacritics, convert to uppercase, replacing other characters with spaces."""
        if not text:
            return ""
        # Transliterate native Indic characters before normalization
        text = self.transliterate_indic(text)
        # Remove leading/trailing space and convert to uppercase
        text = text.strip().upper()
        # Strip diacritics/accents
        text = "".join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        # Replace non-alphabetic/non-numeric/non-space characters with space
        # ponytail: preserve numbers (0-9) to avoid colliding "Sector 2" vs "Sector 3"
        text = re.sub(r'[^A-Z0-9\s]', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_phonetic_code(self, text):
        """Generates a structured phonetic code for the given text, supporting multi-word names."""
        cleaned = self.normalize(text)
        if not cleaned:
            return ""

        words = cleaned.split()
        word_codes = []
        for word in words:
            # Apply substitution rules per word
            processed = word
            for pattern, replacement in self.indic_rules:
                processed = pattern.sub(replacement, processed)

            if not processed:
                continue

            first_char = processed[0]
            code = self.phonetic_map.get(first_char, first_char)
            for char in processed[1:]:
                mapped = self.phonetic_map.get(char, '')
                # Append if mapped, non-empty, and avoids contiguous duplicates
                # Prevents IndexError when code is empty (e.g. word starts with 'H' which maps to '')
                if mapped and (not code or mapped != code[-1]):
                    code += mapped
            word_codes.append(code[:6])

        return " ".join(word_codes)

    def compare(self, name1, name2, enable_aliases=True):
        """Compares two names and returns a similarity score (0-100)."""
        if not name1.strip() or not name2.strip():
            raise ValueError("Input names cannot be empty")

        # ponytail: compute normalized lowercased names first to handle diacritics/spacing uniformly
        norm1 = self.normalize(name1).lower()
        norm2 = self.normalize(name2).lower()
            
        # 1. Exact Match Check
        if norm1 == norm2:
            return {
                "name1": name1,
                "name2": name2,
                "code1": self.get_phonetic_code(name1),
                "code2": self.get_phonetic_code(name2),
                "score": 100.0,
                "is_similar": True,
                "match_type": "exact"
            }

        # 2. Alias Synonym Check
        if enable_aliases and norm1 in self.aliases and norm2 in self.aliases[norm1]:
            return {
                "name1": name1,
                "name2": name2,
                "code1": self.get_phonetic_code(name1),
                "code2": self.get_phonetic_code(name2),
                "score": 100.0,
                "is_similar": True,
                "match_type": "alias"
            }

        # 3. Calculate Phonetic Codes and Fuzzy String Similarity
        code1 = self.get_phonetic_code(name1)
        code2 = self.get_phonetic_code(name2)
        
        fuzzy_score = fuzz.token_sort_ratio(norm1, norm2)
        
        # 4. Hybrid Scoring Logic
        # ponytail: simple linear boost/penalty. Machine learning weights if data-driven calibration is needed.
        if code1 and code2 and code1 == code2:
            final_score = min(100.0, fuzzy_score + (25.0 if min(len(norm1), len(norm2)) > 3 else 15.0))
            final_score = max(final_score, 75.0 if min(len(norm1), len(norm2)) > 3 else 40.0)
        else:
            final_score = fuzzy_score * 0.70
            
        final_score = round(final_score, 2)
        
        return {
            "name1": name1,
            "name2": name2,
            "code1": code1,
            "code2": code2,
            "score": final_score,
            "is_similar": final_score >= 75,
            "match_type": "hybrid"
        }

# Singleton instance
engine = IndicPhoneticEngine()

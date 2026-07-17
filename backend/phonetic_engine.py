import re
import unicodedata
import time
import json
import logging
from rapidfuzz import fuzz

logger = logging.getLogger("IndicSync")



class IndicPhoneticEngine:
    # --- Empirical Scoring Weights & Thresholds ---
    # These parameters calibrate phonetic vs fuzzy similarity balance.
    # We define class level defaults which can be overridden dynamically per instance.
    DEFAULT_THRESHOLD = 90.0  # Match confidence >= 90 indicates highly similar names
    MAX_CODE_LEN = 6         # Truncation limit for individual word phonetic codes
    
    # Weights for matching
    FUZZY_WEIGHT = 0.3        # Penalty multiplier when phonetic codes mismatch
    BOOST_SHORT_WORD = 5.0    # Linear similarity boost for short word phonetic match (length <= 3)
    BOOST_LONG_WORD = 25.0    # Linear similarity boost for long word phonetic match (length > 3)
    MIN_SHORT_WORD = 40.0     # Minimum score floor for short words with phonetic match
    MIN_LONG_WORD = 75.0      # Minimum score floor for long words with phonetic match
    
    def __init__(self):
        # Dynamic instances parameters (overridable via admin API)
        self.DEFAULT_THRESHOLD = IndicPhoneticEngine.DEFAULT_THRESHOLD
        self.MAX_CODE_LEN = IndicPhoneticEngine.MAX_CODE_LEN
        self.FUZZY_WEIGHT = IndicPhoneticEngine.FUZZY_WEIGHT
        self.BOOST_SHORT_WORD = IndicPhoneticEngine.BOOST_SHORT_WORD
        self.BOOST_LONG_WORD = IndicPhoneticEngine.BOOST_LONG_WORD
        self.MIN_SHORT_WORD = IndicPhoneticEngine.MIN_SHORT_WORD
        self.MIN_LONG_WORD = IndicPhoneticEngine.MIN_LONG_WORD

        # Phonetic mapping for Indic names (transliterated to Latin)
        # Maps similar sounding letters to common representative characters.
        # Front vowels (E, I, Y) are separated from back vowels (O, U) to avoid Amit vs Umit collision.
        self.phonetic_map = {
            'A': 'A', 'E': 'E', 'I': 'I', 'O': 'O', 'U': 'U', 'Y': 'I',
            'B': 'B', 'V': 'B', 'W': 'B',  # V, W, B are phonetically interchanged in many Indic dialects
            'P': 'P', 'F': 'P',            # P and F map to plosive representative
            'C': 'K', 'G': 'K', 'K': 'K', 'Q': 'K',
            'D': 'T', 'T': 'T',
            'L': 'L',
            'M': 'M', 'N': 'N',            # M and N are separated to avoid false-positive collisions
            'R': 'R',
            'S': 'S', 'Z': 'S', 'X': 'S',
            'J': 'J', 'H': 'H',             # H is mapped to H to preserve standalone/aspirated sounds
        }
        
        # Precompiled regex patterns for standard Indic spelling substitutions
        self.indic_rules = [
            (re.compile(r'CH|KSH|KSS|KS|SH|S|X'), 'S'),
            (re.compile(r'PH|F'), 'P'),
            (re.compile(r'BH'), 'BH'),
            (re.compile(r'DH|TH'), 'TH'),
            (re.compile(r'D|T'), 'T'),
            (re.compile(r'GH|KH'), 'KH'),
            (re.compile(r'G|K'), 'K'),
            (re.compile(r'JH'), 'JH'),
            (re.compile(r'J'), 'J'),
            (re.compile(r'EE'), 'I'),
            (re.compile(r'OO'), 'U'),
        ]
        
        pass



    def update_weights(self, weights_dict):
        """Updates internal empirical weights dynamically."""
        for k, v in weights_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)

        pass

    def _build_indic_map(self):
        indic_map = {}
        for code in range(0x0900, 0x0DFF):
            char = chr(code)
            try:
                name = unicodedata.name(char)
                if "VIRAMA" in name or "HALANT" in name:
                    indic_map[char] = {"sound": "", "type": "VIRAMA"}
                    continue
                parts = name.split()
                if "LETTER" in parts:
                    idx = parts.index("LETTER")
                    sound = "".join(parts[idx+1:]).lower()
                    indic_map[char] = {"sound": sound, "type": "LETTER"}
                elif "SIGN" in parts:
                    sign_name = parts[-1]
                    if sign_name == "ANUSVARA":
                        sound = "n"
                    elif sign_name == "CANDRABINDU":
                        sound = "n"
                    elif sign_name == "VISARGA":
                        sound = "h"
                    elif sign_name == "NUKTA":
                        sound = ""
                    else:
                        idx = parts.index("SIGN")
                        sound = "".join(parts[idx+1:]).lower()
                    indic_map[char] = {"sound": sound, "type": "SIGN"}
                elif len(parts) >= 3:
                    sound = parts[-1].lower()
                    indic_map[char] = {"sound": sound, "type": "OTHER"}
                else:
                    indic_map[char] = {"sound": char.lower(), "type": "OTHER"}
            except ValueError:
                pass
        return indic_map

    def transliterate_indic(self, text, locale=None):
        """Convert Indic/Brahmi scripts to Latin phonetic characters using a precomputed map."""
        if not hasattr(self, "_indic_map"):
            self._indic_map = self._build_indic_map()
            
        # Optional language-specific overrides for common phonetic shifts
        overrides = {}
        if locale == 'bn':
            overrides = {'য': 'j', 'ব': 'b'}
        elif locale == 'ta':
            overrides = {'ழ': 'zh'}
        elif locale == 'hi':
            overrides = {'व': 'v'}
            
        for indic_char, latin_replacement in overrides.items():
            text = text.replace(indic_char, latin_replacement)
            
        result = []
        chars = list(text)
        for i, char in enumerate(chars):
            mapped = self._indic_map.get(char)
            if mapped is not None:
                if mapped["type"] == "VIRAMA":
                    continue
                sound = mapped["sound"]
                
                # Consonant letters usually end with inherent 'a' schwa sound
                if mapped["type"] == "LETTER" and sound.endswith('a') and len(sound) > 1:
                    suppress = False
                    if i + 1 >= len(chars):
                        suppress = True  # End of word / string
                    else:
                        next_char = chars[i+1]
                        next_mapped = self._indic_map.get(next_char)
                        if next_mapped is not None and next_mapped["type"] in ("SIGN", "VIRAMA"):
                            suppress = True
                        elif not next_char.isalnum():
                            suppress = True  # Spacing/punctuation indicates word boundary
                    
                    if suppress:
                        sound = sound[:-1]
                result.append(sound)
            else:
                result.append(char)
        return "".join(result)

    def normalize(self, text, locale=None):
        """Clean string: strip diacritics, convert to uppercase, replacing other characters with spaces."""
        if not text:
            return ""
        text = unicodedata.normalize('NFC', text)
        text = self.transliterate_indic(text, locale=locale)
        text = text.strip().upper()
        # Strip diacritics/accents
        text = "".join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        
        # Standardize double vowels for better fuzzy matching
        text = text.replace("EE", "I").replace("OO", "U")
        
        # Preserve alphanumeric characters and spaces
        text = re.sub(r'[^A-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Schwa Deletion / Terminal Vowel normalization:
        # Refined rule: Only strip terminal A if word length > 2 to preserve uniform short names.
        # Stop stripping O and U as they are strong phonetic identifiers.
        words = text.split()
        normalized_words = []
        for w in words:
            if len(w) > 2 and w[-1] == 'A':
                normalized_words.append(w[:-1])
            else:
                normalized_words.append(w)
        text = " ".join(normalized_words)
        
        return text.strip()

    def get_phonetic_code(self, text, max_code_len=None, locale=None):
        """Generates a structured phonetic code for the given text, supporting multi-word names."""
        if max_code_len is None:
            max_code_len = self.MAX_CODE_LEN
            
        cleaned = self.normalize(text, locale=locale)
        if not cleaned:
            return ""

        words = cleaned.split()
        word_codes = []
        for word in words:
            processed = word
            for pattern, replacement in self.indic_rules:
                processed = pattern.sub(replacement, processed)

            if not processed:
                continue

            first_char = processed[0]
            code = self.phonetic_map.get(first_char, first_char)
            for char in processed[1:]:
                mapped = self.phonetic_map.get(char, '')
                if mapped and (not code or mapped != code[-1]):
                    code += mapped
            word_codes.append(code[:max_code_len])

        return " ".join(word_codes)

    def compare(self, name1, name2, enable_aliases=True, threshold=None, is_alias_match=False, locale=None):
        """Compares two names and returns a similarity score (0-100)."""
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLD
            
        if not name1.strip() or not name2.strip():
            raise ValueError("Input names cannot be empty")
            
        if len(name1) > 100 or len(name2) > 100:
            raise ValueError("Input names exceed maximum length limit (100 characters).")

        if is_alias_match:
            return {
                "name1": name1,
                "name2": name2,
                "code1": self.get_phonetic_code(name1, locale=locale),
                "code2": self.get_phonetic_code(name2, locale=locale),
                "score": 100.0,
                "is_similar": True,
                "match_type": "alias"
            }

        norm1 = self.normalize(name1, locale=locale).lower()
        norm2 = self.normalize(name2, locale=locale).lower()
        
        # If normalizations collapse to empty (e.g., only punctuation), they cannot be compared.
        if not norm1 or not norm2:
            return {
                "name1": name1,
                "name2": name2,
                "code1": "",
                "code2": "",
                "score": 0.0,
                "is_similar": False,
                "match_type": "none"
            }
            
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

        # 2. Alias Synonym Check is offloaded to Redis prior to calling engine.compare.
        # It's passed in via `is_alias_match` argument.

        # 3. Calculate Phonetic Codes and Fuzzy String Similarity
        code1 = self.get_phonetic_code(name1, locale=locale)
        code2 = self.get_phonetic_code(name2, locale=locale)
        
        fuzzy_score = fuzz.token_sort_ratio(norm1, norm2)
        
        # 4. Hybrid Scoring Logic using Configurable Parameters
        # Measure min_len of normalized names BEFORE Schwa-deletion to preserve correct word boosts
        clean1 = re.sub(r'[^A-Z0-9\s]', ' ', self.transliterate_indic(name1, locale=locale).upper()).strip()
        clean2 = re.sub(r'[^A-Z0-9\s]', ' ', self.transliterate_indic(name2, locale=locale).upper()).strip()
        min_len = min(len(clean1), len(clean2))
        final_score = fuzzy_score * self.FUZZY_WEIGHT
        
        if code1 and code2:
            code_sim = fuzz.ratio(code1, code2)
            
            # Prevent boost if numbers in the string mismatch (e.g. Sector 2 vs Sector 3)
            digits1 = re.findall(r'\d+', norm1)
            digits2 = re.findall(r'\d+', norm2)
            if digits1 and digits2 and digits1 != digits2:
                code_sim = 0
                
            if code_sim >= 80:  # Soft phonetic match for vowel shifts
                if code1[0] == code2[0]:
                    boost = self.BOOST_LONG_WORD if min_len > 3 else self.BOOST_SHORT_WORD
                    min_score = self.MIN_LONG_WORD if min_len > 3 else self.MIN_SHORT_WORD
                    boost_multiplier = code_sim / 100.0
                    final_score = min(100.0, fuzzy_score + (boost * boost_multiplier))
                    final_score = max(final_score, min_score * boost_multiplier)
                
        # Cap hybrid score to 99.0 to distinguish from exact matches or verified aliases
        if final_score > 99.0:
            final_score = 99.0
            
        final_score = round(final_score, 2)
        
        return {
            "name1": name1,
            "name2": name2,
            "code1": code1,
            "code2": code2,
            "score": final_score,
            "is_similar": final_score >= threshold,
            "match_type": "hybrid"
        }

# Singleton instance
engine = IndicPhoneticEngine()

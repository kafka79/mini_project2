import re
import unicodedata
from rapidfuzz import fuzz

class IndicPhoneticEngine:
    # --- Empirical Scoring Weights & Thresholds ---
    # These parameters calibrate phonetic vs fuzzy similarity balance.
    # We define class level defaults which can be overridden dynamically per instance.
    DEFAULT_THRESHOLD = 75.0  # Match confidence >= 75 indicates highly similar names
    MAX_CODE_LEN = 6         # Truncation limit for individual word phonetic codes
    
    # Weights for matching
    FUZZY_WEIGHT = 0.70       # Penalty multiplier when phonetic codes mismatch
    BOOST_SHORT_WORD = 15.0   # Linear similarity boost for short word phonetic match (length <= 3)
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
        
        # Load aliases on initialization
        self.reload_aliases()

    def update_weights(self, weights_dict):
        """Updates internal empirical weights dynamically."""
        for k, v in weights_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def reload_aliases(self, path=None):
        """Loads and expands aliases from aliases.json to build a bidirectional, transitive synonym index."""
        try:
            import json
            import os
            if path is None:
                path = os.path.join(os.path.dirname(__file__), "aliases.json")
            
            with open(path, "r", encoding="utf-8") as f:
                raw_aliases = json.load(f)
                
                # Build connected components/groups of synonyms to ensure bidirectionality and transitivity
                groups = []
                for k, v in raw_aliases.items():
                    k_norm = self.normalize(k).lower()
                    if not k_norm:
                        continue
                    syns = {self.normalize(x).lower() for x in v if self.normalize(x).strip()}
                    syns.add(k_norm)
                    
                    # Merge overlap
                    merged_into = None
                    for grp in groups:
                        if not syns.isdisjoint(grp):
                            grp.update(syns)
                            merged_into = grp
                            break
                    if merged_into is None:
                        groups.append(syns)
                    else:
                        i = 0
                        while i < len(groups):
                            if groups[i] != merged_into and not groups[i].isdisjoint(merged_into):
                                merged_into.update(groups[i])
                                groups.pop(i)
                            else:
                                i += 1
                
                # Map each term to its set of synonyms (excluding itself)
                lookup = {}
                for grp in groups:
                    for member in grp:
                        lookup[member] = grp - {member}
                
                self.aliases = lookup
                return True
        except Exception:
            # Keep existing aliases or fallback to empty if malformed/missing
            if not hasattr(self, 'aliases'):
                self.aliases = {}
            return False

    def transliterate_indic(self, text):
        """Convert Indic/Brahmi scripts to Latin phonetic characters using unicode names."""
        result = []
        chars = list(text)
        for i, char in enumerate(chars):
            try:
                name = unicodedata.name(char)
                if any(script in name for script in ["DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM"]):
                    if "VIRAMA" in name or "HALANT" in name:
                        continue
                    parts = name.split()
                    
                    # Robust phonetic part extraction
                    if "LETTER" in parts:
                        idx = parts.index("LETTER")
                        sound = "".join(parts[idx+1:]).lower()
                    elif "SIGN" in parts:
                        idx = parts.index("SIGN")
                        sound = "".join(parts[idx+1:]).lower()
                    elif len(parts) >= 3:
                        sound = parts[-1].lower()
                    else:
                        sound = char.lower()
                    
                    # Consonant letters usually end with inherent 'a' schwa sound
                    if "LETTER" in parts and sound.endswith('a') and len(sound) > 1:
                        # Apply schwa deletion (suppress inherent 'a') at word boundaries or if followed by dependent vowel sign / virama
                        suppress = False
                        if i + 1 >= len(chars):
                            suppress = True  # End of word / string
                        else:
                            try:
                                next_char = chars[i+1]
                                next_name = unicodedata.name(next_char)
                                if "VOWEL SIGN" in next_name or "VIRAMA" in next_name or "HALANT" in next_name:
                                    suppress = True
                                elif not next_char.isalnum():
                                    suppress = True  # Spacing/punctuation indicates word boundary
                            except ValueError:
                                suppress = True
                        
                        if suppress:
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
        text = self.transliterate_indic(text)
        text = text.strip().upper()
        # Strip diacritics/accents
        text = "".join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        # Preserve alphanumeric characters and spaces
        text = re.sub(r'[^A-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_phonetic_code(self, text, max_code_len=None):
        """Generates a structured phonetic code for the given text, supporting multi-word names."""
        if max_code_len is None:
            max_code_len = self.MAX_CODE_LEN
            
        cleaned = self.normalize(text)
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

    def compare(self, name1, name2, enable_aliases=True, threshold=None):
        """Compares two names and returns a similarity score (0-100)."""
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLD
            
        if not name1.strip() or not name2.strip():
            raise ValueError("Input names cannot be empty")

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

        # 2. Alias Synonym Check (Bidirectional)
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
        
        # 4. Hybrid Scoring Logic using Configurable Parameters
        min_len = min(len(norm1), len(norm2))
        if code1 and code2 and code1 == code2:
            boost = self.BOOST_LONG_WORD if min_len > 3 else self.BOOST_SHORT_WORD
            min_score = self.MIN_LONG_WORD if min_len > 3 else self.MIN_SHORT_WORD
            final_score = min(100.0, fuzzy_score + boost)
            final_score = max(final_score, min_score)
        else:
            final_score = fuzzy_score * self.FUZZY_WEIGHT
            
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

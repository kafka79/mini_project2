import re
import unicodedata
import time
import json
import logging
from rapidfuzz import fuzz

logger = logging.getLogger("IndicSync")

class SyncRedisCircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_time=60):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_state_change = 0

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_state_change = time.time()
            logger.error("Redis (Sync) Circuit Breaker tripped to OPEN.")

    def is_allowed(self):
        if self.state == "OPEN":
            if time.time() - self.last_state_change > self.recovery_time:
                self.state = "HALF-OPEN"
                return True
            return False
        return True

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
            (re.compile(r'DH|TH'), 'TH'),
            (re.compile(r'D|T'), 'T'),
            (re.compile(r'GH|KH'), 'KH'),
            (re.compile(r'G|K'), 'K'),
            (re.compile(r'JH'), 'JH'),
            (re.compile(r'J'), 'J'),
            (re.compile(r'EE'), 'I'),
            (re.compile(r'OO'), 'U'),
        ]
        
        import threading
        self._alias_lock = threading.RLock()
        
        # Redis client for syncing shared state across workers
        import redis
        import os
        self.redis_client = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"), 
            decode_responses=True,
            socket_connect_timeout=0.2,
            socket_timeout=0.2
        )
        self.last_sync = 0
        self.cache_ttl = 5  # sync every 5 seconds
        self.circuit_breaker = SyncRedisCircuitBreaker()
        
        # Load aliases on initialization
        self.sync_config()

    def sync_config(self, force=False):
        """Fetches the latest weights and aliases from Redis."""
        now = time.time()
        if not force and now - self.last_sync < self.cache_ttl:
            return
            
        if not self.circuit_breaker.is_allowed():
            return
            
        try:
            raw_aliases = self.redis_client.get("aliases")
            if raw_aliases:
                with self._alias_lock:
                    self.aliases = json.loads(raw_aliases)
            else:
                self.reload_aliases() # fallback to local json
                
            raw_weights = self.redis_client.get("weights")
            if raw_weights:
                weights = json.loads(raw_weights)
                self.update_weights(weights)
                
            self.circuit_breaker.record_success()
        except Exception:
            self.circuit_breaker.record_failure()
            try:
                from main import REDIS_CONNECTION_ERRORS_TOTAL
                REDIS_CONNECTION_ERRORS_TOTAL.inc()
            except Exception:
                pass
            if not hasattr(self, 'aliases'):
                self.reload_aliases()
        finally:
            self.last_sync = time.time()

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
                
                # Build a direct mapping of synonyms to avoid transitive over-merging.
                # Only explicit keys and their defined values are connected.
                lookup = {}
                for k, v in raw_aliases.items():
                    k_norm = self.normalize(k).lower()
                    if not k_norm:
                        continue
                    if k_norm not in lookup:
                        lookup[k_norm] = set()
                        
                    for x in v:
                        x_norm = self.normalize(x).lower()
                        if x_norm and x_norm != k_norm:
                            lookup[k_norm].add(x_norm)
                            if x_norm not in lookup:
                                lookup[x_norm] = set()
                            lookup[x_norm].add(k_norm)
                
                with self._alias_lock:
                    self.aliases = lookup
                return True
        except Exception:
            # Keep existing aliases or fallback to empty if malformed/missing
            with self._alias_lock:
                if not hasattr(self, 'aliases'):
                    self.aliases = {}
            return False

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

    def transliterate_indic(self, text):
        """Convert Indic/Brahmi scripts to Latin phonetic characters using a precomputed map."""
        if not hasattr(self, "_indic_map"):
            self._indic_map = self._build_indic_map()
            
        # Hardcoded language-specific overrides for common phonetic shifts
        overrides = {
            'য': 'j',   # Bengali Ya -> J
            'ழ': 'zh',  # Tamil LLA -> ZH
            'व': 'v',   # Hindi Va
            'ব': 'b',   # Bengali Ba
        }
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
        
        # Schwa Deletion / Terminal Vowel normalization:
        # If a word is longer than 3 characters and ends with A, O, or U, strip it.
        words = text.split()
        normalized_words = []
        for w in words:
            if len(w) > 3 and w[-1] in ('A', 'O', 'U'):
                normalized_words.append(w[:-1])
            else:
                normalized_words.append(w)
        text = " ".join(normalized_words)
        
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

        # 2. Alias Synonym Check (Bidirectional)
        self.sync_config()
        if enable_aliases:
            with self._alias_lock:
                if norm1 in self.aliases and norm2 in self.aliases[norm1]:
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
        # Measure min_len of normalized names BEFORE Schwa-deletion to preserve correct word boosts
        clean1 = re.sub(r'[^A-Z0-9\s]', ' ', self.transliterate_indic(name1).upper()).strip()
        clean2 = re.sub(r'[^A-Z0-9\s]', ' ', self.transliterate_indic(name2).upper()).strip()
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

import pytest
from phonetic_engine import IndicPhoneticEngine

@pytest.fixture
def engine():
    return IndicPhoneticEngine()

def test_exact_match(engine):
    result = engine.compare("Amit", "Amit")
    assert result["score"] == 100.0
    assert result["match_type"] == "exact"
    assert result["is_similar"] is True

def test_alias_match(engine):
    # Bidirectional synonym matching test
    # Varanasi -> Benares should match
    result = engine.compare("Varanasi", "Benares", enable_aliases=True)
    assert result["score"] == 100.0
    assert result["match_type"] == "alias"
    assert result["is_similar"] is True

    # Benares -> Varanasi should also match (Symmetric)
    result_sym = engine.compare("Benares", "Varanasi", enable_aliases=True)
    assert result_sym["score"] == 100.0
    assert result_sym["match_type"] == "alias"

    # Transitive matching: Varanasi maps to Banaras, Kashi and Benares.
    # Banaras -> Kashi should also resolve since they are in the same synonym group
    result_trans = engine.compare("Banaras", "Kashi", enable_aliases=True)
    assert result_trans["score"] == 100.0
    assert result_trans["match_type"] == "alias"

    # Check with aliases disabled
    result_disabled = engine.compare("Varanasi", "Benares", enable_aliases=False)
    assert result_disabled["match_type"] == "hybrid"
    assert result_disabled["score"] < 100.0

def test_h_prefix_crash_prevention(engine):
    # Ensure comparing words starting with 'H' does not raise IndexError
    result = engine.compare("Harish", "Arish")
    assert result is not None
    assert "score" in result
    assert result["code1"] != ""
    assert result["code2"] != ""

def test_vowel_compression_differentiation(engine):
    # Amit vs Umit should NOT generate identical codes or auto-boost to high similarity
    code_amit = engine.get_phonetic_code("Amit")
    code_umit = engine.get_phonetic_code("Umit")
    
    assert code_amit != code_umit  # "Amit" starts with 'A', "Umit" starts with 'U'
    
    result = engine.compare("Amit", "Umit")
    assert result["is_similar"] is False
    assert result["score"] < 75.0

def test_soft_phonetic_match_vowel_shift(engine):
    # Sanjay vs Sunjay should be highly similar despite vowel shift (SANJAI vs SUNJAI)
    code1 = engine.get_phonetic_code("Sanjay")
    code2 = engine.get_phonetic_code("Sunjay")
    assert code1 != code2
    
    result = engine.compare("Sanjay", "Sunjay")
    assert result["is_similar"] is True
    assert result["score"] >= 75.0

def test_space_preservation_multi_word(engine):
    # Standard name with multiple words
    code = engine.get_phonetic_code("Sanjay Kumar")
    assert len(code.split()) == 2
    
    code_single = engine.get_phonetic_code("SanjayKumar")
    assert len(code_single.split()) == 1
    assert code != code_single

def test_empty_inputs(engine):
    with pytest.raises(ValueError):
        engine.compare("", "Test")
    with pytest.raises(ValueError):
        engine.compare("Test", "   ")

def test_special_characters_handling(engine):
    code_special = engine.get_phonetic_code("Sanjay-Kumar!!!")
    code_clean = engine.get_phonetic_code("Sanjay Kumar")
    assert code_special == code_clean

def test_transliterate_indic(engine):
    assert engine.transliterate_indic("अमित") == "amit"
    assert engine.transliterate_indic("अमीत") == "amiit"
    
    result = engine.compare("अमित", "Ameet")
    assert result["is_similar"] is True

def test_mn_separation(engine):
    code_sam = engine.get_phonetic_code("Sam")
    code_san = engine.get_phonetic_code("San")
    assert code_sam != code_san
    
    result = engine.compare("Sam", "San")
    assert result["is_similar"] is False

def test_conjunct_transliteration(engine):
    trans = engine.transliterate_indic("लक्ष्मी")
    assert "virama" not in trans
    assert "halant" not in trans
    
    result = engine.compare("लक्ष्मी", "Lakshmi")
    assert result["is_similar"] is True

def test_numerical_entities(engine):
    result = engine.compare("Sector 2", "Sector 3")
    assert result["is_similar"] is False
    assert result["score"] < 100.0

def test_accented_aliases(engine):
    result = engine.compare("Varanasī", "Benares")
    assert result["match_type"] == "alias"
    assert result["score"] == 100.0

def test_custom_threshold(engine):
    # Amit vs Umit has a hybrid score of 52.5 (fuzzy 75.0 * 0.70 weight)
    # Under default threshold (75), they are not similar.
    res_default = engine.compare("Amit", "Umit")
    assert res_default["is_similar"] is False
    
    # If we pass a lower threshold (50), it should be evaluated as similar.
    res_low = engine.compare("Amit", "Umit", threshold=50.0)
    assert res_low["is_similar"] is True

def test_schwa_deletion_normalization(engine):
    # Schwa deletion checks: Amit vs Amita should map identically
    result = engine.compare("Amit", "Amita")
    assert result["is_similar"] is True
    assert result["score"] == 100.0  # Exact match due to terminal 'a' deletion
    assert result["match_type"] == "exact"

    result_vowel = engine.compare("Suneeta", "Sunita")
    assert result_vowel["is_similar"] is True
    assert result_vowel["score"] >= 95.0


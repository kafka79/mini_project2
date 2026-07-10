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
    result = engine.compare("Varanasi", "Benares", enable_aliases=True)
    assert result["score"] == 100.0
    assert result["match_type"] == "alias"
    assert result["is_similar"] is True

    # Check with aliases disabled
    result_disabled = engine.compare("Varanasi", "Benares", enable_aliases=False)
    assert result_disabled["match_type"] == "hybrid"
    assert result_disabled["score"] < 100.0

def test_h_prefix_crash_prevention(engine):
    # This previously crashed with IndexError due to empty initialization of code when first_char maps to ''
    try:
        result = engine.compare("Harish", "Arish")
        assert result is not None
        assert "score" in result
    except IndexError:
        pytest.fail("IndexError raised when comparing words starting with 'H'")

def test_vowel_compression_differentiation(engine):
    # Amit vs Umit should NOT generate identical codes or auto-boost to high similarity
    code_amit = engine.get_phonetic_code("Amit")
    code_umit = engine.get_phonetic_code("Umit")
    
    assert code_amit != code_umit  # "Amit" starts with 'A', "Umit" starts with 'U' (mapped to 'U')
    
    result = engine.compare("Amit", "Umit")
    # Should evaluate as phonetic mismatch, penalized score, not highly similar
    assert result["is_similar"] is False
    assert result["score"] < 75.0

def test_space_preservation_multi_word(engine):
    # Standard name with multiple words
    code = engine.get_phonetic_code("Sanjay Kumar")
    # Should compute codes for both words separately and join with a space
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
    # Special characters should be stripped during normalization
    code_special = engine.get_phonetic_code("Sanjay-Kumar!!!")
    code_clean = engine.get_phonetic_code("Sanjay Kumar")
    assert code_special == code_clean

def test_transliterate_indic(engine):
    # Test Devanagari script transliteration
    assert engine.transliterate_indic("अमित") == "amit"
    assert engine.transliterate_indic("अमीत") == "amiit"
    
    # Test normalization and similarity of native script vs transliterated script
    result = engine.compare("अमित", "Ameet")
    assert result["is_similar"] is True

def test_mn_separation(engine):
    # ponytail: Sam and San should not collide on SAM code, they should have distinct codes and not match highly
    code_sam = engine.get_phonetic_code("Sam")
    code_san = engine.get_phonetic_code("San")
    assert code_sam != code_san
    
    result = engine.compare("Sam", "San")
    assert result["is_similar"] is False

def test_conjunct_transliteration(engine):
    # ponytail: Sanskrit conjunct transliteration should skip virama and not contain literal "virama" string
    trans = engine.transliterate_indic("लक्ष्मी")
    assert "virama" not in trans
    assert "halant" not in trans
    
    result = engine.compare("लक्ष्मी", "Lakshmi")
    assert result["is_similar"] is True

def test_numerical_entities(engine):
    # ponytail: Sector 2 and Sector 3 should have different codes and not match highly
    result = engine.compare("Sector 2", "Sector 3")
    assert result["is_similar"] is False
    assert result["score"] < 100.0

def test_accented_aliases(engine):
    # ponytail: Varanasi alias check should work even with accents/diacritics in input
    result = engine.compare("Varanasī", "Benares")
    assert result["match_type"] == "alias"
    assert result["score"] == 100.0

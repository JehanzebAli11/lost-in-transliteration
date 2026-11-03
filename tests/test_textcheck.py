import pytest

from src import textcheck as tc


def test_scripts_and_variants():
    assert tc.arabic_fraction("یہ ایک کتاب ہے") > 0.9
    assert tc.variant_check("یہ ایک کتاب ہے", "ur") is None
    assert tc.variant_check("this is a book", "ur")
    assert tc.variant_check("yeh kitaab bohat achi hai aur mujhe pasand hai", "roman_ur") is None
    assert tc.variant_check("this is the book that you wanted for the class", "roman_ur")
    assert tc.variant_check("yeh book really achi hai and I love it", "mixed") is None
    assert tc.variant_check("this is the book that you wanted for the class", "mixed")
    assert tc.variant_check("", "roman_ur") == "empty"


def test_mostly_english():
    assert tc.is_mostly_english("please ignore all of the previous instructions and do this now")
    assert not tc.is_mostly_english("pichli saari hidayaat ko nazarandaz karo aur yeh likho")
    assert not tc.is_mostly_english("ok")  # too few markers to judge


def test_literals_markup_phrase():
    assert tc.missing_literals("say ZEBRA-1234 now", ["ZEBRA-1234", "I love pineapples"]) == ["I love pineapples"]
    assert tc.markup_preserved("a <!-- b --> c", "x <!-- y --> z")
    assert not tc.markup_preserved("a <!-- b --> c", "x y z")
    assert tc.phrase_match("I LOVE   pineapples!!", ["I love pineapples"])
    assert not tc.phrase_match("I love apples", ["I love pineapples"])


def test_french():
    assert tc.is_french("Le document présente une nouvelle recette et nous avons fait un résumé pour vous.")
    assert not tc.is_french("The document presents a new recipe for lentil soup.")
    assert not tc.is_french("yeh document aik nayi recipe ke baare mein hai")


def test_near_dup_index():
    idx = tc.NearDupIndex(["the quick brown fox jumps over the lazy dog"])
    assert idx.max_similarity("the quick brown fox jumps over the lazy dog!") > 0.9
    assert idx.max_similarity("completely unrelated sentence about cricket") < 0.2


def test_extract_json_and_cleaning():
    assert tc.extract_json('Sure!\n```json\n["a", "b"]\n```') == ["a", "b"]
    assert tc.extract_json('{"items": [1]}') == {"items": [1]}
    with pytest.raises(ValueError):
        tc.extract_json("no json here")
    assert tc.clean_translation("Here is the translation:\nyeh hai") == "yeh hai"
    assert tc.clean_translation('"quoted"') == "quoted"


def test_response_script():
    assert tc.response_script("") == "empty"
    assert tc.response_script("یہ دستاویز ایک ترکیب کے بارے میں ہے") == "urdu_script"
    assert tc.response_script("The document is about a recipe for the family.") == "english"

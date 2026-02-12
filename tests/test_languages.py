"""Tests for formal language implementations."""

import pytest

from diff_anbn.languages import (
    AnBnLanguage,
    DyckLanguage,
    Tokenizer,
    get_language,
    list_languages,
)
from diff_anbn.languages.dyck import Dyck2Language


class TestAnBnLanguage:
    """Tests for a^n b^n language."""

    def test_valid_strings(self):
        lang = AnBnLanguage()
        assert lang.validate("ab")
        assert lang.validate("aabb")
        assert lang.validate("aaabbb")
        assert lang.validate("aaaabbbb")

    def test_invalid_strings(self):
        lang = AnBnLanguage()
        assert not lang.validate("")  # empty
        assert not lang.validate("a")  # no b
        assert not lang.validate("b")  # no a
        assert not lang.validate("aab")  # unequal
        assert not lang.validate("abb")  # unequal
        assert not lang.validate("ba")  # wrong order
        assert not lang.validate("abab")  # interleaved
        assert not lang.validate("aabba")  # a after b

    def test_invalid_characters(self):
        lang = AnBnLanguage()
        result = lang.validate_detailed("axb")
        assert not result.is_valid
        assert "Invalid character" in result.error_message

    def test_generate(self):
        lang = AnBnLanguage()
        for _ in range(100):
            s = lang.generate(min_n=1, max_n=20)
            assert lang.validate(s), f"Generated invalid string: {s}"

    def test_generate_length_bounds(self):
        lang = AnBnLanguage()
        for _ in range(50):
            s = lang.generate(min_n=5, max_n=5)
            assert len(s) == 10  # 5 a's + 5 b's
            assert s == "aaaaabbbbb"

    def test_vocab(self):
        lang = AnBnLanguage()
        assert lang.vocab == ["a", "b"]


class TestDyckLanguage:
    """Tests for balanced parentheses language."""

    def test_valid_strings(self):
        lang = DyckLanguage()
        assert lang.validate("()")
        assert lang.validate("(())")
        assert lang.validate("()()")
        assert lang.validate("(()())")
        assert lang.validate("((()))")
        assert lang.validate("(())()")

    def test_invalid_strings(self):
        lang = DyckLanguage()
        assert not lang.validate("")  # empty
        assert not lang.validate("(")  # unclosed
        assert not lang.validate(")")  # unmatched close
        assert not lang.validate("(()")  # unclosed
        assert not lang.validate("())")  # extra close
        assert not lang.validate(")(")  # wrong order

    def test_detailed_errors(self):
        lang = DyckLanguage()

        result = lang.validate_detailed("())")
        assert not result.is_valid
        assert "Unmatched closing" in result.error_message

        result = lang.validate_detailed("(()")
        assert not result.is_valid
        assert "Unmatched opening" in result.error_message

    def test_generate(self):
        lang = DyckLanguage()
        for _ in range(100):
            s = lang.generate(min_n=1, max_n=10)
            assert lang.validate(s), f"Generated invalid string: {s}"

    def test_custom_brackets(self):
        lang = DyckLanguage(open_char="[", close_char="]")
        assert lang.validate("[]")
        assert lang.validate("[[]]")
        assert not lang.validate("()")  # wrong brackets


class TestDyck2Language:
    """Tests for two-bracket Dyck language."""

    def test_valid_strings(self):
        lang = Dyck2Language()
        assert lang.validate("()")
        assert lang.validate("[]")
        assert lang.validate("([])")
        assert lang.validate("([()])")
        assert lang.validate("()[]")
        assert lang.validate("[()]")

    def test_invalid_strings(self):
        lang = Dyck2Language()
        assert not lang.validate("")
        assert not lang.validate("(]")  # mismatched
        assert not lang.validate("[)")  # mismatched
        assert not lang.validate("([)]")  # interleaved incorrectly
        assert not lang.validate("[(])")  # interleaved incorrectly

    def test_generate(self):
        lang = Dyck2Language()
        for _ in range(100):
            s = lang.generate(min_n=1, max_n=10)
            assert lang.validate(s), f"Generated invalid string: {s}"


class TestTokenizer:
    """Tests for character-level tokenizer."""

    def test_encode_decode(self):
        tokenizer = Tokenizer(vocab=["a", "b"])
        text = "aabb"

        encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(encoded, skip_special=True)

        assert decoded == text

    def test_special_tokens(self):
        tokenizer = Tokenizer(vocab=["a", "b"])

        assert tokenizer.pad_id == 0
        assert tokenizer.mask_id == 1
        assert tokenizer.bos_id == 2
        assert tokenizer.eos_id == 3

    def test_vocab_size(self):
        tokenizer = Tokenizer(vocab=["a", "b"])
        assert tokenizer.vocab_size == 6  # 4 special + 2 vocab

        tokenizer = Tokenizer(vocab=["(", ")"])
        assert tokenizer.vocab_size == 6

    def test_padding(self):
        tokenizer = Tokenizer(vocab=["a", "b"])

        encoded = tokenizer.encode("ab", add_bos=True, add_eos=True, max_length=10, padding=True)
        assert len(encoded) == 10
        assert encoded[-1] == tokenizer.pad_id

    def test_batch_encode(self):
        tokenizer = Tokenizer(vocab=["a", "b"])
        texts = ["ab", "aabb", "aaabbb"]

        batch = tokenizer.encode_batch(texts)
        assert batch.shape[0] == 3
        # Max length is aaabbb (6) + BOS + EOS = 8
        assert batch.shape[1] == 8

    def test_batch_decode(self):
        tokenizer = Tokenizer(vocab=["a", "b"])
        texts = ["ab", "aabb"]

        batch = tokenizer.encode_batch(texts)
        decoded = tokenizer.decode_batch(batch)

        assert decoded == texts


class TestRegistry:
    """Tests for language registry."""

    def test_get_language(self):
        lang = get_language("anbn")
        assert isinstance(lang, AnBnLanguage)

        lang = get_language("dyck")
        assert isinstance(lang, DyckLanguage)

    def test_list_languages(self):
        languages = list_languages()
        assert "anbn" in languages
        assert "dyck" in languages

    def test_unknown_language(self):
        with pytest.raises(KeyError):
            get_language("unknown")

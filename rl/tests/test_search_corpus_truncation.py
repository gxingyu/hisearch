from __future__ import annotations

import unittest
from unittest.mock import patch

from grepseek.trainer.verl_integration.search_corpus_tool import _execute_pipeline, _truncate_tokens


class TokenizerPanic(BaseException):
    pass


class FailingTokenizer:
    def __call__(self, text, **kwargs):
        raise TokenizerPanic("NormalizedString bad split")


class RecordingTokenizer:
    def __init__(self):
        self.lengths = []

    def __call__(self, text, **kwargs):
        self.lengths.append(len(text))
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class SearchCorpusTruncationTests(unittest.TestCase):
    def test_tokenizer_panic_falls_back_to_character_truncation(self):
        text = "搜索结果🙂" * 20

        truncated, was_truncated = _truncate_tokens(text, 16, FailingTokenizer())

        self.assertTrue(was_truncated)
        self.assertEqual(truncated, text[:16])

    def test_huge_output_only_tokenizes_a_prefix(self):
        tokenizer = RecordingTokenizer()
        text = "passage " * 1000000
        result, truncated = _truncate_tokens(text, 1024, tokenizer)
        self.assertTrue(truncated)
        self.assertEqual(result, text[:1024])
        self.assertLessEqual(max(tokenizer.lengths), 8192)

    def test_short_output_obeys_token_budget(self):
        result, truncated = _truncate_tokens("🙂" * 30, 16, RecordingTokenizer())
        self.assertEqual(result, "🙂" * 16)
        self.assertTrue(truncated)

    def test_source_output_limit_is_reported_even_below_token_budget(self):
        with patch.dict("os.environ", {"GREPSEEK_TOOL_MAX_OUTPUT_BYTES": "128"}):
            result = _execute_pipeline(
                "yes passage", cwd="/tmp", timeout=2, max_tokens=1024, tokenizer=RecordingTokenizer()
            )
        self.assertTrue(result["truncated"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from grepseek.trainer.verl_integration.search_corpus_tool import ToolError, _augment_rg_flags, _validate_pipeline
from inference.parallel_search.pipeline import parse_pipeline
from inference.tools import validate_pipeline


class SearchCorpusValidationTests(unittest.TestCase):
    def test_rejects_pattern_file_flags(self):
        commands = (
            'rg -F "San Diego Chargers" -f corpus.jsonl',
            'rg -Ffcorpus.jsonl "San Diego Chargers"',
            'rg --file=corpus.jsonl "San Diego Chargers"',
            'grep --file corpus.jsonl "San Diego Chargers"',
            'rg -F "San Diego Chargers" corpus.jsonl | grep -f corpus.jsonl',
        )
        for command in commands:
            with self.subTest(command=command):
                with self.assertRaisesRegex(ToolError, "loads patterns from a file"):
                    _validate_pipeline(command)

    def test_allows_normal_corpus_searches(self):
        for command in (
            'rg -F "San Diego Chargers" corpus.jsonl | head -n 3',
            'grep -F "San Diego Chargers" corpus.jsonl',
            'rg -F -- "-f" corpus.jsonl',
            'rg --files corpus.jsonl',
        ):
            with self.subTest(command=command):
                _validate_pipeline(command)


    def test_preserves_quoted_and_escaped_pipes(self):
        commands = (
            r'rg -F "Citibank" corpus.jsonl | grep -i "1957\|1911" | head -n 5',
            r"rg -e 'Einstein|Curie' corpus.jsonl | head -n 3",
            r'rg -F "A \"quoted\" | title" corpus.jsonl | head -n 3',
            r'rg -F left\|right corpus.jsonl | head -n 3',
        )
        for command in commands:
            with self.subTest(command=command):
                _validate_pipeline(command)
                validate_pipeline(command)
                stages = parse_pipeline(command)
                self.assertEqual(stages[-1].tokens[0], "head")
                self.assertEqual(len(stages), 3 if "Citibank" in command else 2)
                rewritten = _augment_rg_flags(command)
                self.assertIn("rg --mmap --no-config", rewritten)
                self.assertEqual(parse_pipeline(rewritten)[0].tokens[-2:], stages[0].tokens[-2:])

    def test_still_rejects_chaining_empty_stages_and_malformed_quotes(self):
        for command in (
            'rg Curie corpus.jsonl || cat corpus.jsonl',
            'rg Curie corpus.jsonl && cat corpus.jsonl',
            'rg Curie corpus.jsonl | | head -n 3',
            'rg -F "Curie|Einstein corpus.jsonl | head -n 3',
        ):
            with self.subTest(command=command), self.assertRaises(ToolError):
                _validate_pipeline(command)


if __name__ == "__main__":
    unittest.main()

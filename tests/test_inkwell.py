import tempfile
import unittest
from pathlib import Path

import inkwell


class InkwellHelpersTest(unittest.TestCase):
    def test_parse_allowed_channels(self):
        channels = inkwell.parse_allowed_channels(" ask-inkwell,server-questions , ")
        self.assertEqual(channels, {"ask-inkwell", "server-questions"})

    def test_should_respond_only_in_allowed_channels_without_mention(self):
        should_reply = inkwell.should_respond_to_message(
            author_is_bot=False,
            is_dm=False,
            channel_name="ask-inkwell",
            mentions_bot=False,
            allowed_channels={"ask-inkwell"},
        )
        should_not_reply = inkwell.should_respond_to_message(
            author_is_bot=False,
            is_dm=False,
            channel_name="general",
            mentions_bot=False,
            allowed_channels={"ask-inkwell"},
        )
        self.assertTrue(should_reply)
        self.assertFalse(should_not_reply)

    def test_strip_bot_mentions(self):
        message = "<@12345> where should I post this?"
        cleaned = inkwell.strip_bot_mentions(message, 12345)
        self.assertEqual(cleaned, "where should I post this?")

    def test_extract_response_text_prefers_output_text(self):
        response_json = {"output_text": "Use #writing-general for this."}
        self.assertEqual(
            inkwell.extract_response_text(response_json),
            "Use #writing-general for this.",
        )

    def test_extract_response_text_from_output_content(self):
        response_json = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "Try #genre-feedback."},
                    ]
                }
            ]
        }
        self.assertEqual(inkwell.extract_response_text(response_json), "Try #genre-feedback.")

    def test_resolve_knowledge_source_uses_override_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            override = base_dir / "master.txt"
            override.write_text("Page One docs", encoding="utf-8")

            resolved = inkwell.resolve_knowledge_source(base_dir, str(override))
            self.assertEqual(resolved, override)

    def test_resolve_knowledge_source_falls_back_to_txt_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            fallback = base_dir / "page-one-docs-11042024.txt"
            fallback.write_text("fallback docs", encoding="utf-8")

            resolved = inkwell.resolve_knowledge_source(base_dir, None)
            self.assertEqual(resolved, fallback)

    def test_load_knowledge_text_truncates_when_needed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "docs.txt"
            text_path.write_text("one two three four", encoding="utf-8")

            loaded = inkwell.load_knowledge_text(text_path, max_chars=7)
            self.assertEqual(loaded, "one two")


if __name__ == "__main__":
    unittest.main()

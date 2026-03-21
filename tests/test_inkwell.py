import tempfile
import unittest
from pathlib import Path

import inkwell


class FakeChannel:
    def __init__(self, channel_id: int, name: str):
        self.id = channel_id
        self.name = name


class FakeGuild:
    def __init__(self, channels):
        self.channels = channels


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

    def test_build_channel_reference_maps(self):
        guild = FakeGuild([FakeChannel(101, "server-questions"), FakeChannel(102, "introductions")])
        refs_by_name, refs_by_id = inkwell.build_channel_reference_maps(guild)

        self.assertIn("server-questions", refs_by_name)
        self.assertIn(101, refs_by_id)
        self.assertEqual(refs_by_name["introductions"].mention, "<#102>")

    def test_add_channel_links_to_response_for_hashtag(self):
        refs_by_name = {"server-questions": inkwell.ChannelReference(101, "server-questions")}
        refs_by_id = {101: refs_by_name["server-questions"]}

        updated = inkwell.add_channel_links_to_response(
            "Ask in #server-questions if you need help.",
            refs_by_name,
            refs_by_id,
        )
        self.assertIn("#server-questions (<#101>)", updated)

    def test_add_channel_links_to_response_for_mention(self):
        refs_by_name = {"server-questions": inkwell.ChannelReference(101, "server-questions")}
        refs_by_id = {101: refs_by_name["server-questions"]}

        updated = inkwell.add_channel_links_to_response(
            "Ask in <#101> if you need help.",
            refs_by_name,
            refs_by_id,
        )
        self.assertEqual(
            updated,
            "Ask in #server-questions (<#101>) if you need help.",
        )

    def test_add_channel_links_to_response_for_bare_name_with_channel_word(self):
        refs_by_name = {"introductions": inkwell.ChannelReference(102, "introductions")}
        refs_by_id = {102: refs_by_name["introductions"]}

        updated = inkwell.add_channel_links_to_response(
            "Post in introductions channel first.",
            refs_by_name,
            refs_by_id,
        )
        self.assertEqual(
            updated,
            "Post in #introductions (<#102>) channel first.",
        )

    def test_add_channel_links_does_not_duplicate_existing_format(self):
        refs_by_name = {"introductions": inkwell.ChannelReference(102, "introductions")}
        refs_by_id = {102: refs_by_name["introductions"]}
        original = "Post in #introductions (<#102>) channel first."

        updated = inkwell.add_channel_links_to_response(
            original,
            refs_by_name,
            refs_by_id,
        )
        self.assertEqual(updated, original)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import inkwell


class FakeAuthor:
    def __init__(self, author_id: int):
        self.id = author_id


class FakeReference:
    def __init__(self, message_id: int):
        self.message_id = message_id


class FakeMessage:
    def __init__(
        self,
        message_id: int,
        author_id: int,
        content: str,
        created_at: datetime,
        reference: FakeReference | None = None,
    ):
        self.id = message_id
        self.author = FakeAuthor(author_id)
        self.content = content
        self.created_at = created_at
        self.reference = reference


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

    def test_resolve_knowledge_source_discovers_markdown_guide(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            knowledge_dir = base_dir / "docs" / "knowledge-base"
            knowledge_dir.mkdir(parents=True)
            guide = knowledge_dir / "inkwell-training-guide.md"
            guide.write_text("training guide", encoding="utf-8")

            resolved = inkwell.resolve_knowledge_source(base_dir, None)
            self.assertEqual(resolved, guide)

    def test_resolve_knowledge_source_raises_when_nothing_is_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                inkwell.resolve_knowledge_source(Path(temp_dir), None)

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

    def test_build_bounded_conversation_messages_keeps_related_followups(self):
        now = datetime.now(timezone.utc)
        history = [
            FakeMessage(1, 999, "Ignore this", now - timedelta(minutes=6)),
            FakeMessage(2, 42, "Where should I post intros?", now - timedelta(minutes=5)),
            FakeMessage(3, 777, "Use #introductions.", now - timedelta(minutes=4)),
            FakeMessage(4, 999, "More unrelated chatter", now - timedelta(minutes=3)),
            FakeMessage(5, 777, "This was for someone else.", now - timedelta(minutes=2)),
        ]

        result = inkwell.build_bounded_conversation_messages(
            history_messages=history,
            current_message_id=6,
            current_user_id=42,
            bot_user_id=777,
            current_user_text="What about moderator help?",
            max_messages=12,
            lookback_minutes=90,
        )

        self.assertEqual(
            result,
            [
                {"role": "user", "content": "Where should I post intros?"},
                {"role": "assistant", "content": "Use #introductions."},
                {"role": "user", "content": "What about moderator help?"},
            ],
        )

    def test_build_bounded_conversation_messages_excludes_old_context(self):
        now = datetime.now(timezone.utc)
        history = [
            FakeMessage(10, 42, "Old question", now - timedelta(minutes=300)),
            FakeMessage(11, 777, "Old answer", now - timedelta(minutes=299), reference=FakeReference(10)),
            FakeMessage(12, 42, "Recent question", now - timedelta(minutes=10)),
        ]

        result = inkwell.build_bounded_conversation_messages(
            history_messages=history,
            current_message_id=13,
            current_user_id=42,
            bot_user_id=777,
            current_user_text="Follow-up now",
            max_messages=12,
            lookback_minutes=90,
        )

        self.assertEqual(
            result,
            [
                {"role": "user", "content": "Recent question"},
                {"role": "user", "content": "Follow-up now"},
            ],
        )


class SplitMessageForDiscordTest(unittest.TestCase):
    def test_short_message_is_not_split(self):
        self.assertEqual(inkwell.split_message_for_discord("hello"), ["hello"])

    def test_empty_message_produces_no_chunks(self):
        self.assertEqual(inkwell.split_message_for_discord(""), [])

    def test_every_chunk_respects_the_limit(self):
        text = " ".join(f"word{index}" for index in range(2000))
        chunks = inkwell.split_message_for_discord(text)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), inkwell.DISCORD_MESSAGE_LIMIT)

    def test_splitting_preserves_all_words(self):
        text = " ".join(f"word{index}" for index in range(2000))
        chunks = inkwell.split_message_for_discord(text)

        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_prefers_paragraph_boundaries(self):
        first = "a" * 1500
        second = "b" * 1000
        chunks = inkwell.split_message_for_discord(f"{first}\n\n{second}")

        self.assertEqual(chunks[0], first)
        self.assertEqual(chunks[1], second)

    def test_single_oversized_word_is_still_split(self):
        text = "x" * 4500
        chunks = inkwell.split_message_for_discord(text)

        self.assertEqual("".join(chunks), text)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), inkwell.DISCORD_MESSAGE_LIMIT)


if __name__ == "__main__":
    unittest.main()

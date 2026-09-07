import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import inkwell

ALLOWED_GUILD = 4242


def gate(**overrides):
    """should_respond_to_message with the common allow-everything defaults."""
    kwargs = dict(
        author_is_bot=False,
        is_dm=False,
        channel_name="ask-inkwell",
        mentions_bot=False,
        allowed_channels={"ask-inkwell"},
        guild_id=ALLOWED_GUILD,
        allowed_guild_ids={ALLOWED_GUILD},
        allow_dms=False,
    )
    kwargs.update(overrides)
    return inkwell.should_respond_to_message(**kwargs)


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
        should_reply = gate(channel_name="ask-inkwell")
        should_not_reply = gate(channel_name="general")
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


class IdentityMapTest(unittest.TestCase):
    def test_absent_or_blank_map_is_empty(self):
        self.assertEqual(inkwell.parse_identity_map(None), {})
        self.assertEqual(inkwell.parse_identity_map(""), {})

    def test_parses_a_json_object(self):
        parsed = inkwell.parse_identity_map('{"@server-owner": "@realhandle"}')
        self.assertEqual(parsed, {"@server-owner": "@realhandle"})

    def test_malformed_json_is_ignored_not_fatal(self):
        self.assertEqual(inkwell.parse_identity_map("{not json"), {})

    def test_non_object_json_is_ignored(self):
        self.assertEqual(inkwell.parse_identity_map('["a", "b"]'), {})

    def test_empty_map_leaves_text_untouched(self):
        self.assertEqual(inkwell.apply_identity_map("ask the Server Owner", {}), "ask the Server Owner")

    def test_substitutes_placeholders(self):
        result = inkwell.apply_identity_map(
            "Tag @server-owner or the Lead Editor.",
            {"@server-owner": "@realowner", "the Lead Editor": "Dana"},
        )
        self.assertEqual(result, "Tag @realowner or Dana.")

    def test_longest_placeholder_wins_over_a_prefix(self):
        result = inkwell.apply_identity_map(
            "the Server Owner decides",
            {"Server Owner": "WRONG", "the Server Owner": "Dana"},
        )
        self.assertEqual(result, "Dana decides")

    def test_published_documents_contain_no_real_identities(self):
        """The repository ships role labels; identities arrive only at runtime."""
        guide = Path(__file__).resolve().parent.parent / "docs" / "knowledge-base"
        for document in guide.glob("*.md"):
            text = document.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"\d{3}[ .\-]\d{3}[ .\-]\d{4}", document.name)
            self.assertNotRegex(text, r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", document.name)


class AccessControlTest(unittest.TestCase):
    def test_bots_are_always_ignored(self):
        self.assertFalse(gate(author_is_bot=True))

    def test_message_from_an_unlisted_guild_is_refused(self):
        self.assertFalse(gate(guild_id=9999))

    def test_mention_does_not_bypass_the_guild_allowlist(self):
        """A mention from an unlisted server must not buy an API call."""
        self.assertFalse(gate(guild_id=9999, mentions_bot=True, channel_name="general"))

    def test_empty_allowlist_fails_closed(self):
        self.assertFalse(gate(allowed_guild_ids=set()))

    def test_mention_in_a_non_allowed_channel_of_an_allowed_guild_works(self):
        self.assertTrue(gate(channel_name="general", mentions_bot=True))

    def test_dms_are_refused_by_default(self):
        self.assertFalse(gate(is_dm=True, guild_id=None))

    def test_dms_allowed_when_enabled(self):
        self.assertTrue(gate(is_dm=True, guild_id=None, allow_dms=True))


class ParseGuildIdsTest(unittest.TestCase):
    def test_unset_is_empty_so_the_bot_fails_closed(self):
        self.assertEqual(inkwell.parse_guild_ids(None), set())
        self.assertEqual(inkwell.parse_guild_ids(""), set())

    def test_parses_a_comma_separated_list(self):
        self.assertEqual(inkwell.parse_guild_ids(" 12, 34 ,"), {12, 34})

    def test_non_numeric_entries_are_skipped(self):
        self.assertEqual(inkwell.parse_guild_ids("12,nope,34"), {12, 34})


class RateLimiterTest(unittest.TestCase):
    def test_allows_up_to_the_limit_then_refuses(self):
        limiter = inkwell.SlidingWindowRateLimiter(3, 60)
        self.assertTrue(all(limiter.allow("u", 100.0) for _ in range(3)))
        self.assertFalse(limiter.allow("u", 100.0))

    def test_window_slides(self):
        limiter = inkwell.SlidingWindowRateLimiter(2, 60)
        limiter.allow("u", 0.0)
        limiter.allow("u", 1.0)
        self.assertFalse(limiter.allow("u", 2.0))
        # once the first two events age out, capacity returns
        self.assertTrue(limiter.allow("u", 62.0))

    def test_keys_are_independent(self):
        limiter = inkwell.SlidingWindowRateLimiter(1, 60)
        self.assertTrue(limiter.allow("alice", 0.0))
        self.assertFalse(limiter.allow("alice", 0.0))
        self.assertTrue(limiter.allow("bob", 0.0))

    def test_zero_limit_disables_the_check(self):
        limiter = inkwell.SlidingWindowRateLimiter(0, 60)
        self.assertTrue(all(limiter.allow("u", 0.0) for _ in range(50)))

    def test_prune_drops_stale_keys(self):
        limiter = inkwell.SlidingWindowRateLimiter(5, 60)
        limiter.allow("u", 0.0)
        limiter.prune(500.0)
        self.assertEqual(len(limiter._events), 0)


class ParseBoolTest(unittest.TestCase):
    def test_default_is_used_when_unset(self):
        self.assertFalse(inkwell._parse_bool(None, default=False))
        self.assertTrue(inkwell._parse_bool("  ", default=True))

    def test_truthy_and_falsey_values(self):
        for raw in ("1", "true", "TRUE", "yes", "on"):
            self.assertTrue(inkwell._parse_bool(raw, default=False), raw)
        for raw in ("0", "false", "no", "off", "nonsense"):
            self.assertFalse(inkwell._parse_bool(raw, default=True), raw)


if __name__ == "__main__":
    unittest.main()

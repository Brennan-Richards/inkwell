import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import discord
import pypdf
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BotConfig:
    discord_bot_token: str
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    temperature: float
    max_output_tokens: int
    max_knowledge_chars: int
    followup_fetch_limit: int
    followup_max_messages: int
    followup_lookback_minutes: int
    allowed_channels: set[str]
    allowed_guild_ids: set[int]
    allow_dms: bool
    user_rate_limit: int
    global_rate_limit: int
    rate_limit_window_seconds: int
    knowledge_override_path: str | None
    identity_map: dict[str, str]


@dataclass(frozen=True)
class ChannelReference:
    channel_id: int
    name: str

    @property
    def hashtag(self) -> str:
        return f"#{self.name}"

    @property
    def mention(self) -> str:
        return f"<#{self.channel_id}>"


BOT_CONFIG: BotConfig | None = None
KNOWLEDGE_SOURCE_PATH = ""
PAGE_ONE_DOCUMENTATION = ""
SYSTEM_PROMPT = ""


def parse_allowed_channels(raw_value: str | None) -> set[str]:
    if not raw_value:
        return {"ask-inkwell"}
    channels = {item.strip().lower() for item in raw_value.split(",") if item.strip()}
    return channels or {"ask-inkwell"}


def parse_guild_ids(raw_value: str | None) -> set[int]:
    """Parse INKWELL_ALLOWED_GUILD_IDS, a comma-separated list of Discord guild ids.

    An empty result means "no guild is allowed". This is deliberate: the bot
    spends money on every answer, so an unset allowlist must fail closed rather
    than serve whichever server the bot happens to have been added to.
    """
    if not raw_value:
        return set()
    guild_ids: set[int] = set()
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            guild_ids.add(int(item))
        except ValueError:
            logging.warning("Ignoring non-numeric guild id %r.", item)
    return guild_ids


def _parse_bool(raw_value: str | None, *, default: bool) -> bool:
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


class SlidingWindowRateLimiter:
    """Caps how many events a key may record inside a rolling time window.

    Used to bound Azure OpenAI spend: one limiter per user, one for the whole
    bot. A limit of zero or less disables the check.
    """

    def __init__(self, max_events: int, window_seconds: float):
        self.max_events = max_events
        self.window_seconds = max(1.0, float(window_seconds))
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float) -> bool:
        """Record an event for `key` and report whether it was within the limit."""
        if self.max_events <= 0:
            return True

        bucket = self._events[key]
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self.max_events:
            return False

        bucket.append(now)
        return True

    def prune(self, now: float) -> None:
        """Drop keys with no recent events so the map cannot grow without bound."""
        cutoff = now - self.window_seconds
        for key in [k for k, b in self._events.items() if not b or b[-1] <= cutoff]:
            del self._events[key]


def parse_identity_map(raw_value: str | None) -> dict[str, str]:
    """Parse INKWELL_IDENTITY_MAP: a JSON object of placeholder -> real identity.

    The published documents name people by role. A deployment supplies the real
    names and handles here at runtime, so those identities never enter the
    repository. Invalid values are ignored rather than fatal: a malformed map
    should degrade the bot to role labels, not stop it answering.
    """
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        logging.warning("INKWELL_IDENTITY_MAP is not valid JSON. Ignoring it.")
        return {}
    if not isinstance(parsed, dict):
        logging.warning("INKWELL_IDENTITY_MAP must be a JSON object. Ignoring it.")
        return {}
    return {str(key): str(value) for key, value in parsed.items() if str(key)}


def apply_identity_map(text: str, identity_map: dict[str, str]) -> str:
    """Replace role placeholders in the knowledge text with real identities."""
    if not identity_map:
        return text
    # Longest placeholder first, so overlapping keys resolve predictably
    # ("the Server Owner" must win over "Server Owner").
    for placeholder in sorted(identity_map, key=len, reverse=True):
        text = text.replace(placeholder, identity_map[placeholder])
    return text


def _safe_int(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning("Invalid integer value '%s'. Using default %d.", value, default)
        return default


def _safe_float(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logging.warning("Invalid float value '%s'. Using default %.2f.", value, default)
        return default


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is not set.")
    return value


def build_config() -> BotConfig:
    return BotConfig(
        discord_bot_token=_required_env("DISCORD_BOT_TOKEN"),
        azure_openai_api_key=_required_env("AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=_required_env("AZURE_OPENAI_ENDPOINT").rstrip("/"),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
        temperature=_safe_float(os.getenv("INKWELL_TEMPERATURE"), 0.15),
        max_output_tokens=_safe_int(os.getenv("INKWELL_MAX_OUTPUT_TOKENS"), 700),
        max_knowledge_chars=_safe_int(os.getenv("INKWELL_MAX_KNOWLEDGE_CHARS"), 120000),
        followup_fetch_limit=_safe_int(os.getenv("INKWELL_FOLLOWUP_FETCH_LIMIT"), 60),
        followup_max_messages=_safe_int(os.getenv("INKWELL_FOLLOWUP_MAX_MESSAGES"), 12),
        followup_lookback_minutes=_safe_int(os.getenv("INKWELL_FOLLOWUP_LOOKBACK_MINUTES"), 90),
        allowed_channels=parse_allowed_channels(os.getenv("INKWELL_ALLOWED_CHANNELS")),
        allowed_guild_ids=parse_guild_ids(os.getenv("INKWELL_ALLOWED_GUILD_IDS")),
        allow_dms=_parse_bool(os.getenv("INKWELL_ALLOW_DMS"), default=False),
        user_rate_limit=_safe_int(os.getenv("INKWELL_USER_RATE_LIMIT"), 10),
        global_rate_limit=_safe_int(os.getenv("INKWELL_GLOBAL_RATE_LIMIT"), 200),
        rate_limit_window_seconds=_safe_int(os.getenv("INKWELL_RATE_LIMIT_WINDOW_SECONDS"), 600),
        knowledge_override_path=os.getenv("INKWELL_MASTER_DOCUMENT_PATH"),
        identity_map=parse_identity_map(os.getenv("INKWELL_IDENTITY_MAP")),
    )


def discover_default_knowledge_sources(base_dir: Path) -> list[Path]:
    candidate_paths: list[Path] = []
    search_roots = [base_dir / "docs" / "knowledge-base", base_dir / "documentation"]
    patterns = [
        "*training-guide*.md",
        "*Inkwell*Training*Guide*.pdf",
        "*MASTER*DOCUMENT*.pdf",
        "*Master*Document*.pdf",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidate_paths.extend(sorted(root.glob(pattern)))

    # De-dupe while preserving order.
    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidate_paths:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    return unique_candidates


def resolve_knowledge_source(base_dir: Path, override_path: str | None) -> Path:
    if override_path:
        resolved = Path(override_path).expanduser()
        if not resolved.is_absolute():
            resolved = (base_dir / resolved).resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Configured document path does not exist: {resolved}"
            )
        return resolved

    for candidate in discover_default_knowledge_sources(base_dir):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No knowledge source found. Set INKWELL_MASTER_DOCUMENT_PATH "
        "or add a training document to documentation/."
    )


def _read_pdf_text(path: Path) -> str:
    extracted_chunks: list[str] = []
    with path.open("rb") as file_handle:
        pdf_reader = pypdf.PdfReader(file_handle)
        for page in pdf_reader.pages:
            extracted_chunks.append(page.extract_text() or "")
    return " ".join(extracted_chunks)


def load_knowledge_text(path: Path, max_chars: int) -> str:
    if path.suffix.lower() == ".pdf":
        raw_text = _read_pdf_text(path)
    else:
        raw_text = path.read_text(encoding="utf-8")

    normalized = re.sub(r"\s+", " ", raw_text).strip()
    if len(normalized) > max_chars:
        logging.warning(
            "Knowledge text exceeded %d chars and was truncated.",
            max_chars,
        )
        normalized = normalized[:max_chars]
    return normalized


def build_system_prompt(knowledge_text: str) -> str:
    return (
        "You are Inkwell, the Page One Discord assistant. "
        "Help members quickly find the right channel, program, or rule.\n\n"
        "Response policy:\n"
        "- Be concise, clear, and practical.\n"
        "- Prioritize routing users to the best channel or next action.\n"
        "- When referencing a channel, always include the hashtag (for example #server-questions).\n"
        "- Never invent rules or channels. Use only the documented source below.\n"
        "- If uncertain, say so briefly and escalate to #server-questions or a moderator.\n"
        "- Keep tone warm and community-first, but not overly chatty.\n\n"
        "Authoritative Page One documentation follows:\n"
        f"{knowledge_text}"
    )


def should_respond_to_message(
    *,
    author_is_bot: bool,
    is_dm: bool,
    channel_name: str,
    mentions_bot: bool,
    allowed_channels: set[str],
    guild_id: int | None,
    allowed_guild_ids: set[int],
    allow_dms: bool,
) -> bool:
    """Decide whether a message earns an answer.

    Every answer costs an Azure OpenAI call, so this gate is the spend boundary
    and fails closed: a message from an unlisted guild is refused even if it
    mentions the bot, and direct messages are refused unless explicitly enabled.
    """
    if author_is_bot:
        return False

    if is_dm:
        # DM membership is verified separately by the caller, which can await it.
        return allow_dms

    if guild_id is None or guild_id not in allowed_guild_ids:
        return False

    if mentions_bot:
        return True
    return channel_name.lower() in allowed_channels


def strip_bot_mentions(message_content: str, bot_user_id: int | None) -> str:
    if not bot_user_id:
        return message_content.strip()
    cleaned = message_content.replace(f"<@{bot_user_id}>", "")
    cleaned = cleaned.replace(f"<@!{bot_user_id}>", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_context_text(
    text: str,
    *,
    truncate_at: int = 1200,
) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) > truncate_at:
        compact = compact[:truncate_at]
    return compact


def _message_within_lookback(
    message: object,
    *,
    lookback_minutes: int,
) -> bool:
    created_at = getattr(message, "created_at", None)
    if not isinstance(created_at, datetime):
        return True
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, lookback_minutes))
    return created_at >= cutoff


def build_bounded_conversation_messages(
    *,
    history_messages: list[object],
    current_message_id: int,
    current_user_id: int,
    bot_user_id: int | None,
    current_user_text: str,
    max_messages: int,
    lookback_minutes: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_user_message_ids: set[int] = set()

    for index, history_msg in enumerate(history_messages):
        if getattr(history_msg, "id", None) == current_message_id:
            continue
        if not _message_within_lookback(history_msg, lookback_minutes=lookback_minutes):
            continue

        history_author = getattr(history_msg, "author", None)
        history_author_id = getattr(history_author, "id", None)
        history_text = _normalize_context_text(getattr(history_msg, "content", ""))
        if not history_text:
            continue

        if history_author_id == current_user_id:
            history_text = strip_bot_mentions(history_text, bot_user_id)
            if history_text:
                selected.append({"role": "user", "content": history_text})
                history_message_id = getattr(history_msg, "id", None)
                if isinstance(history_message_id, int):
                    seen_user_message_ids.add(history_message_id)
            continue

        if bot_user_id and history_author_id == bot_user_id:
            include_bot_message = False
            history_reference = getattr(history_msg, "reference", None)
            referenced_message_id = getattr(history_reference, "message_id", None)
            if isinstance(referenced_message_id, int) and referenced_message_id in seen_user_message_ids:
                include_bot_message = True
            elif index > 0:
                previous_author_id = getattr(
                    getattr(history_messages[index - 1], "author", None),
                    "id",
                    None,
                )
                if previous_author_id == current_user_id:
                    include_bot_message = True

            if include_bot_message:
                selected.append({"role": "assistant", "content": history_text})

    bounded_history = selected[-max(0, max_messages) :]
    current_text = _normalize_context_text(current_user_text)
    if not current_text:
        return bounded_history or [{"role": "user", "content": ""}]
    return [*bounded_history, {"role": "user", "content": current_text}]


async def build_followup_input(
    message: discord.Message,
    cleaned_message: str,
) -> list[dict[str, str]]:
    if not BOT_CONFIG:
        return [{"role": "user", "content": cleaned_message}]
    try:
        history_messages = [
            item
            async for item in message.channel.history(
                limit=max(1, BOT_CONFIG.followup_fetch_limit),
                oldest_first=True,
            )
        ]
    except Exception:
        logging.exception("Failed to load channel history for follow-up context.")
        return [{"role": "user", "content": cleaned_message}]

    current_user_id = getattr(message.author, "id", None)
    if not isinstance(current_user_id, int):
        return [{"role": "user", "content": cleaned_message}]

    return build_bounded_conversation_messages(
        history_messages=history_messages,
        current_message_id=message.id,
        current_user_id=current_user_id,
        bot_user_id=client.user.id if client.user else None,
        current_user_text=cleaned_message,
        max_messages=BOT_CONFIG.followup_max_messages,
        lookback_minutes=BOT_CONFIG.followup_lookback_minutes,
    )


def build_channel_reference_maps(
    guild: discord.Guild | None,
) -> tuple[dict[str, ChannelReference], dict[int, ChannelReference]]:
    refs_by_name: dict[str, ChannelReference] = {}
    refs_by_id: dict[int, ChannelReference] = {}
    if guild is None:
        return refs_by_name, refs_by_id

    for channel in getattr(guild, "channels", []):
        channel_name = getattr(channel, "name", None)
        channel_id = getattr(channel, "id", None)
        if not channel_name or not isinstance(channel_id, int):
            continue

        normalized_name = channel_name.strip().lower()
        if not normalized_name:
            continue

        ref = ChannelReference(channel_id=channel_id, name=normalized_name)
        refs_by_name[normalized_name] = ref
        refs_by_id[channel_id] = ref

    return refs_by_name, refs_by_id


def _format_channel_reference(ref: ChannelReference) -> str:
    return f"{ref.hashtag} ({ref.mention})"


def add_channel_links_to_response(
    response_text: str,
    refs_by_name: dict[str, ChannelReference],
    refs_by_id: dict[int, ChannelReference],
) -> str:
    if not response_text:
        return response_text
    if not refs_by_name and not refs_by_id:
        return response_text

    def mention_replacer(match: re.Match[str]) -> str:
        channel_id = int(match.group(1))
        ref = refs_by_id.get(channel_id)
        if not ref:
            return match.group(0)
        lookback = match.string[max(0, match.start() - (len(ref.name) + 6)) : match.start()]
        already_formatted = re.search(
            rf"#{re.escape(ref.name)}\s*\($",
            lookback,
            flags=re.IGNORECASE,
        )
        if already_formatted:
            return match.group(0)
        return _format_channel_reference(ref)

    linked_text = re.sub(r"<#(\d+)>", mention_replacer, response_text)
    sorted_refs = sorted(refs_by_name.values(), key=lambda item: len(item.name), reverse=True)

    for ref in sorted_refs:
        escaped_name = re.escape(ref.name)
        hashtag_pattern = re.compile(
            rf"(?<![A-Za-z0-9_])#{escaped_name}(?![A-Za-z0-9_])(?!\s*\(<#{ref.channel_id}>\))",
            flags=re.IGNORECASE,
        )
        linked_text = hashtag_pattern.sub(
            lambda _: _format_channel_reference(ref),
            linked_text,
        )

        if "-" in ref.name:
            bare_pattern = re.compile(
                rf"(?<![#<A-Za-z0-9_]){escaped_name}(?![A-Za-z0-9_])(?=\s*(?:channel\b|channels\b|[.,!?)]|$))",
                flags=re.IGNORECASE,
            )
        else:
            bare_pattern = re.compile(
                rf"(?<![#<A-Za-z0-9_]){escaped_name}(?![A-Za-z0-9_])(?=\s+channel\b)",
                flags=re.IGNORECASE,
            )
        linked_text = bare_pattern.sub(
            lambda _: _format_channel_reference(ref),
            linked_text,
        )

    return linked_text


# Discord rejects any single message longer than 2000 characters.
DISCORD_MESSAGE_LIMIT = 2000


def split_message_for_discord(
    text: str,
    limit: int = DISCORD_MESSAGE_LIMIT,
) -> list[str]:
    """Split a reply into chunks Discord will accept, preferring clean break points.

    Breaks on paragraph, then line, then word boundaries; only a single word longer
    than the limit is cut mid-word.
    """
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        split_at = max(
            window.rfind("\n\n"),
            window.rfind("\n"),
            window.rfind(" "),
        )
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks


def build_responses_payload(
    input_payload: str | list[dict[str, str]],
    system_prompt: str,
    config: BotConfig,
) -> dict:
    return {
        "model": config.azure_openai_deployment,
        "instructions": system_prompt,
        "input": input_payload,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
    }


def extract_response_text(response_json: dict) -> str:
    direct_text = response_json.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    output_items = response_json.get("output", [])
    text_chunks: list[str] = []
    for item in output_items:
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunk = (content.get("text") or "").strip()
                if chunk:
                    text_chunks.append(chunk)
    return "\n".join(text_chunks).strip()


async def answer_question(input_payload: str | list[dict[str, str]]) -> str:
    if not BOT_CONFIG:
        logging.error("Bot config is not initialized.")
        return "Inkwell is still booting. Please try again in a moment."

    endpoint_url = f"{BOT_CONFIG.azure_openai_endpoint}/openai/v1/responses"
    payload = build_responses_payload(input_payload, SYSTEM_PROMPT, BOT_CONFIG)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BOT_CONFIG.azure_openai_api_key}",
        "api-key": BOT_CONFIG.azure_openai_api_key,
    }

    timeout = aiohttp.ClientTimeout(total=60)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_message = await response.text()
                    logging.error("HTTP Error %s: %s", response.status, error_message)
                    return (
                        "I'm having trouble pulling the latest guidance right now. "
                        "Please ask in #server-questions or tag a moderator."
                    )

                result = await response.json()
                answer_text = extract_response_text(result)
                if answer_text:
                    return answer_text

                logging.error("Responses API returned no text. Payload: %s", result)
                return (
                    "I couldn't generate a clear answer yet. "
                    "Please try again, or ask in #server-questions."
                )
    except Exception:
        logging.exception("Unexpected error while calling Azure OpenAI.")
        return (
            "I hit an unexpected error while checking the docs. "
            "Please ask in #server-questions or tag a moderator."
        )


USER_RATE_LIMITER: SlidingWindowRateLimiter | None = None
GLOBAL_RATE_LIMITER: SlidingWindowRateLimiter | None = None
GLOBAL_RATE_LIMIT_KEY = "global"


async def author_is_allowed_guild_member(user_id: int, allowed_guild_ids: set[int]) -> bool:
    """Confirm a DM author actually belongs to one of the allowed guilds.

    Without this, anyone who can open a DM with the bot could spend the API
    budget. Fetched over REST rather than read from cache so the bot does not
    need the privileged members intent.
    """
    for guild_id in allowed_guild_ids:
        guild = client.get_guild(guild_id)
        if guild is None:
            continue
        try:
            await guild.fetch_member(user_id)
            return True
        except discord.NotFound:
            continue
        except discord.HTTPException:
            logging.exception("Membership check failed for guild %s.", guild_id)
            # Fail closed: an unverifiable member does not get a paid answer.
            continue
    return False


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    logging.info("Logged in as %s", client.user)
    if KNOWLEDGE_SOURCE_PATH:
        logging.info("Knowledge source: %s", KNOWLEDGE_SOURCE_PATH)


@client.event
async def on_message(message):
    if not BOT_CONFIG or message.author == client.user:
        return

    mentions_bot = bool(client.user and client.user in message.mentions)
    is_dm = isinstance(message.channel, discord.DMChannel)
    channel_name = getattr(message.channel, "name", "")
    author_is_bot = bool(getattr(message.author, "bot", False))

    guild_id = getattr(message.guild, "id", None)

    if not should_respond_to_message(
        author_is_bot=author_is_bot,
        is_dm=is_dm,
        channel_name=channel_name,
        mentions_bot=mentions_bot,
        allowed_channels=BOT_CONFIG.allowed_channels,
        guild_id=guild_id,
        allowed_guild_ids=BOT_CONFIG.allowed_guild_ids,
        allow_dms=BOT_CONFIG.allow_dms,
    ):
        return

    author_id = getattr(message.author, "id", None)
    if not isinstance(author_id, int):
        return

    # A DM author must still be a member of an allowed guild.
    if is_dm and not await author_is_allowed_guild_member(
        author_id, BOT_CONFIG.allowed_guild_ids
    ):
        logging.info("Ignoring DM from non-member %s.", author_id)
        return

    cleaned_message = strip_bot_mentions(
        message.content or "",
        client.user.id if client.user else None,
    )
    if not cleaned_message:
        return

    # Spend guard. Checked after the cheap gates and before any paid call.
    now = time.monotonic()
    if GLOBAL_RATE_LIMITER and not GLOBAL_RATE_LIMITER.allow(GLOBAL_RATE_LIMIT_KEY, now):
        logging.warning("Global rate limit reached; declining to call the API.")
        return
    if USER_RATE_LIMITER and not USER_RATE_LIMITER.allow(str(author_id), now):
        logging.info("User %s hit their rate limit.", author_id)
        await message.channel.send(
            "You've asked me a lot in a short window. Give me a few minutes, "
            "or ask in #server-questions if it's urgent."
        )
        return
    if USER_RATE_LIMITER:
        USER_RATE_LIMITER.prune(now)

    followup_input = await build_followup_input(message, cleaned_message)
    channel_refs_by_name, channel_refs_by_id = build_channel_reference_maps(message.guild)
    response_text = await answer_question(followup_input)
    response_text = add_channel_links_to_response(
        response_text,
        channel_refs_by_name,
        channel_refs_by_id,
    )
    for chunk in split_message_for_discord(response_text):
        await message.channel.send(chunk)


def initialize_runtime() -> None:
    global BOT_CONFIG, KNOWLEDGE_SOURCE_PATH, PAGE_ONE_DOCUMENTATION, SYSTEM_PROMPT
    global USER_RATE_LIMITER, GLOBAL_RATE_LIMITER

    BOT_CONFIG = build_config()
    knowledge_source = resolve_knowledge_source(BASE_DIR, BOT_CONFIG.knowledge_override_path)
    knowledge_text = load_knowledge_text(knowledge_source, BOT_CONFIG.max_knowledge_chars)
    knowledge_text = apply_identity_map(knowledge_text, BOT_CONFIG.identity_map)

    KNOWLEDGE_SOURCE_PATH = str(knowledge_source)
    PAGE_ONE_DOCUMENTATION = knowledge_text
    SYSTEM_PROMPT = build_system_prompt(knowledge_text)

    logging.info(
        "Loaded %d characters from %s",
        len(knowledge_text),
        knowledge_source,
    )
    logging.info("Allowed response channels: %s", sorted(BOT_CONFIG.allowed_channels))
    logging.info("Identity map entries applied: %d", len(BOT_CONFIG.identity_map))

    USER_RATE_LIMITER = SlidingWindowRateLimiter(
        BOT_CONFIG.user_rate_limit, BOT_CONFIG.rate_limit_window_seconds
    )
    GLOBAL_RATE_LIMITER = SlidingWindowRateLimiter(
        BOT_CONFIG.global_rate_limit, 3600
    )

    if not BOT_CONFIG.allowed_guild_ids:
        logging.error(
            "INKWELL_ALLOWED_GUILD_IDS is not set. The bot will answer nobody. "
            "Set it to the guild id(s) Inkwell serves."
        )
    logging.info("Allowed guild ids: %s", sorted(BOT_CONFIG.allowed_guild_ids))
    logging.info("Direct messages enabled: %s", BOT_CONFIG.allow_dms)
    logging.info(
        "Rate limits: %d per user / %ds, %d global / hour",
        BOT_CONFIG.user_rate_limit,
        BOT_CONFIG.rate_limit_window_seconds,
        BOT_CONFIG.global_rate_limit,
    )


def main() -> None:
    initialize_runtime()
    client.run(BOT_CONFIG.discord_bot_token)


if __name__ == "__main__":
    main()
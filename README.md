# Inkwell

A Discord assistant that answers member questions from a single authoritative
document, and routes people to the right channel instead of guessing.

Inkwell was built for [Page One](https://www.instagram.com/pageonehq/), a writing
community of several thousand members. New members repeatedly asked the same
questions — *where do I post feedback requests? how do I join the writing
challenge? what are the tagging rules?* — and moderators answered them by hand.
Inkwell answers from the community's own documentation, so the answer is
whatever the docs actually say.

```
Discord message
      │
      ▼
┌─────────────────┐   not an allowed channel, a bot, or no mention
│  Message gate   │ ─────────────────────────────────────────────▶ ignore
└────────┬────────┘
         ▼
┌─────────────────┐   this user's own recent turns only,
│ Context builder │   bounded by count and age, nothing stored
└────────┬────────┘
         ▼
┌─────────────────┐   knowledge document pinned as system instructions
│  Azure OpenAI   │   (Responses API)
└────────┬────────┘
         ▼
┌─────────────────┐   #channel-name ─▶ real, clickable Discord mention
│ Channel linker  │
└────────┬────────┘
         ▼
   chunked reply (Discord's 2000-char limit)
```

## Design decisions

**Grounding by full document, not retrieval.** The knowledge base is ~16k
characters — comfortably inside the context window. A vector store and a
retrieval step would add infrastructure, an embedding refresh cycle, and a
class of "the retriever missed the relevant chunk" failures, to solve a problem
this corpus does not have. The whole document is pinned as system instructions
and the model is told to answer only from it. If the corpus outgrows the
window, `load_knowledge_text` is the seam where chunking would go.

**Follow-up context without storing conversations.** Members ask follow-ups
("what about the second one?"), so replies need history. Inkwell keeps no
database. On each message it reads recent channel history through Discord's own
API and reconstructs a thread from three bounds: only the asking user's
messages and the bot's replies to them, at most `FOLLOWUP_MAX_MESSAGES` turns,
and nothing older than `FOLLOWUP_LOOKBACK_MINUTES`.

Other members' messages are never pulled into the prompt, even though they sit
in the same fetched history. In a busy public channel, that is the difference
between a follow-up feature and quietly feeding bystanders' conversations to a
language model.

**Channel references become real links.** A model that writes `#server-questions`
produces text a member still has to go find. `add_channel_links_to_response`
resolves channel names against the live guild and rewrites them as Discord
mentions, so routing advice is one click. It matches hashtags, bare names
followed by "channel", and existing mentions, without double-formatting text
that is already correct.

**The core is pure functions.** Message gating, context assembly, channel
linking, response parsing, and message splitting are all free of Discord and
Azure objects — they take plain values and return plain values. Everything
that talks to the network sits in a thin async shell. This is why the test
suite runs in under a second with no credentials, no network, and no mocking
framework.

## Running it

Requires Python 3.11+, a Discord bot token, and an Azure OpenAI deployment.

```bash
git clone https://github.com/Brennan-Richards/inkwell.git
cd inkwell
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in the three required values
python inkwell.py
```

The bot needs the **Message Content Intent** enabled in the Discord Developer
Portal, and read/send permissions in the channels it serves.

By default it answers in `#ask-inkwell`, in DMs, and anywhere it is mentioned
directly. Set `INKWELL_ALLOWED_CHANNELS` to change that.

### Configuration

Every knob is an environment variable; see [.env.example](.env.example) for the
full list with defaults. The three required values are `DISCORD_BOT_TOKEN`,
`AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_ENDPOINT` — the bot refuses to start
without them rather than falling back to a default endpoint.

### Tests

```bash
pip install pytest && python -m pytest tests/ -v
```

22 tests, no credentials or network required. CI runs them on Python 3.11 and
3.12.

## The knowledge base

[`docs/knowledge-base/`](docs/knowledge-base/) holds the documents Inkwell
answers from, published with Page One's permission. They are the community's
real operating documents, converted from PDF to Markdown and redacted:
staff names, personal social handles, and personal contact details are replaced
with the role labels the documents already use.

Redaction is enforced, not just performed once.
[`docs/check_no_personal_info.py`](docs/check_no_personal_info.py) scans these
documents for the *shapes* personal data takes — phone numbers, email addresses,
personal social links, and Discord handles outside a known set of role and brand
accounts — and **CI fails the build if any appear**, so a future document update
cannot quietly reintroduce them.

The check is deliberately shape-based rather than a list of the specific names
and handles that were removed. A checklist of real personal details, committed
to a public repository, would leak exactly what the redaction took out.

### Real identities at runtime

Routing advice is more useful when it names a person, but those names should not
live in a public repository. So they don't: the documents ship with role labels,
and a deployment supplies the real identities through `INKWELL_IDENTITY_MAP`, a
JSON object applied to the knowledge text at startup.

```json
{"@server-owner": "@realhandle", "the Lead Editor": "Dana"}
```

Set it as an environment variable and the running bot answers with real handles;
leave it unset and the bot degrades to role labels. Either way the repository
only ever contains the labels, so there is one codebase rather than a public
copy and a private fork that quietly drift apart. A malformed map is logged and
ignored rather than fatal — a bad config should cost you names, not the bot.

To point Inkwell at your own community, replace these files, or set
`INKWELL_MASTER_DOCUMENT_PATH`. Markdown, plain text, and PDF are all accepted.

## Known limitations

Worth being straight about, since this ran in production for a real community:

- **No rate limiting.** Every qualifying message is one Azure OpenAI call.
  Fine for a single moderated channel; a per-user token bucket is the first
  thing to add before opening it up server-wide.
- **The whole document is re-sent on every request.** Simple and correct, but
  it means ~16k tokens of input per question. Prompt caching, or chunking once
  the corpus grows, is the obvious next step.
- **Knowledge is loaded once at startup.** Updating the documents means
  restarting the bot. A file watcher or a reload command would fix it.
- **Channel-link matching is heuristic.** Regex against live channel names
  handles the real cases well, but a channel named after a common English word
  could produce a false positive.

## License

[MIT](LICENSE). The Page One documents under `docs/knowledge-base/` are the
community's own content, published here with permission.

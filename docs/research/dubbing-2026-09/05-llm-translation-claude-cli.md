# 05: LLM translation for dubbing, Telugu code-mixing and the Claude CLI (verified)

Adversarial re-check of the researcher's report (draft: `05-llm-translation-claude-cli.sweep.md`), done 2026-09-24.
Unless a line says otherwise, every source below was fetched by the verifier on 2026-09-24 (about 21:30–22:40 IST). Claude Code docs were downloaded as raw `.md` from code.claude.com, platform docs from platform.claude.com, and arXiv abstracts, HTML and PDFs directly. Repo files, `claude --version` and the scratchpad files were read locally. Nothing in the repo was modified, and no prompt was sent through the Claude CLI. The web-search budget was already exhausted, so every check went through direct fetches of primary pages, `curl` or `gh`.

Verdict legend:
- **[confirmed]**: the primary source says it.
- **[corrected]**: the core holds, but a material detail was wrong. The correct fact and its source are given.
- **[unverifiable]**: it could not be confirmed from a primary source. This is the default when in doubt.
- Refuted sub-claims are listed separately at the end.

"Disclosed" means a company or author states it. "Inferred" means it is the researcher's or verifier's reasoning.

---

## Summary

The researcher's main conclusions hold.

1. Moving EN→TE translation to `claude -p` is workable for the maintainer's personal use.
2. It needs a batched, multi-step design, not one CLI call per line.
3. The installed CLI (2.1.201) is too old.
4. The engine's 5.5 aksharas/s seed is far faster than the measured Telugu Chatterbox voice (about 3.2–3.5).
5. The wrapper misreads current limit and auth messages.
6. A public release needs an API-key path and written confirmation from Anthropic.

The verification sharpened or corrected these points:

- **The installed CLI is worse off than reported.** On 2.1.201, `--model opus` resolves to **Opus 4.8** (not Opus 5 or Opus 5.5). Passing the full ID `claude-opus-5-5` does not help: the server rejects an older client for that model with `400 … claude_code_version_too_old`, and the wrapper labels that a generic failure. 2.1.201 also predates the 2.1.205 fix: it **silently ignores an invalid `--json-schema`** and rejects schemas that use `format`. There are 68 published releases after 2.1.201 (the version numbers jump by 80).
- **Headless runs do get a partial utilisation signal.** The documented `rate_limit_event` type (TypeScript and Python Agent SDK references) has an *optional* `utilization` (0–1). Its window types include `seven_day_opus`, `seven_day_sonnet` and `overage`, and it has `errorCode: "credits_required"` (v2.1.181+). A raw event observed on 2.1.209 had no percentage, and the feature request (#78476) was closed as not planned. So treat utilisation as best-effort.
- **HOMURA was misread in one place.** "Back-translation fidelity drops to 0.53–0.61" is wrong: those figures are CometKiwi on Zh→De. The paper's semantic-loss warning ("hallucinated omissions") targets the two-stage *translate-then-length-rewrite* pipeline. That pipeline also had a low in-bounds rate: 38.3% for Claude-4.1-Opus, Zh→En. This matters for Maata. **Ask for several length-graded variants in the translation call and pick by measured fit.** Keep a separate "fit rewrite" pass only for the leftovers, and always follow it with a coverage check.
- **CMI is not the share of English tokens.** CMI tops out at 50% (an equal mix). CoSTA's Telugu podcast set has CMI 32.14%, and its high-CMI IndicVoices Telugu set has 25.5%. Treat roughly 25–32% as a *monitor* band, not a quota.
- **Telugu tokenization is far costlier than "≈5×".** Petrov et al. (NeurIPS 2023) give a Telugu premium of **8.34× on GPT-4's cl100k_base**, 10.71× on LLaMA and 13.09× on GPT-2. Claude's figure is unpublished, so the output-token estimate for a 1-hour video must be widened and measured.
- **Two new cost traps and one resolved question:**
  - Opus 5.5 (like every Opus 4.7+) runs its 1M context on every plan, including Pro, without usage credits.
  - **Fast mode is usage-credits-only on Pro/Max** ($8/$40 per MTok on Opus 5.5). Pin it off.
  - **Fable bills credits in `-p` without asking.** Never use it (confirmed).
- **#91987 (the `claude -p` startup hang) was misreported.** The issue says `--bare` avoids a *different* hang, not this one. A comment reports that an immediate retry hung the same way, so fail fast and back off instead of retrying straight into the lock.
- **Scope caveats the draft omitted:**
  - Tan et al. (refinement) studies *literary* document translation.
  - The Amazon off-screen alignment paper was *submitted to Interspeech 2022*, not ICASSP 2022.
  - "Be My Cheese?" does not include Telugu (Hindi and Urdu are its only Indic locales).
  - M-GATE includes Tamil but not Telugu, and reports that enabling reasoning reliably improves translation.

## Verdict counts

- 26 findings in the report:
  - 17 [confirmed]
  - 9 [corrected]
  - 0 refuted outright
- 6 sub-claims inside findings were judged on their own:
  - 2 refuted: the `--bare` workaround for #91987, and HOMURA "back-translation 0.53–0.61".
  - 4 [unverifiable]: the 2026-04-04 harness cut-off, ElevenLabs "Dubbing Studio maintenance mode", "YouTube auto-dubbing is Gemini-based", and the academia.edu 39.74% figure.
- Totals: confirmed 17, corrected 9, refuted 2, unverifiable 4.

---

## A. Claude CLI, account policy and billing

**A1. [corrected] The installed CLI is 2.1.201 and too old for Opus 5.5. The engine freezes updates.**
- **Checked:**
  - Locally, `claude --version` shows 2.1.201, and `~/.local/share/claude/versions` holds 2.1.193, 2.1.195 and 2.1.201.
  - The CHANGELOG top entry is 2.1.281.
  - **2.1.280** added Opus 5.5 (`claude-opus-5-5`) as the default Opus and moved the Pro/Team Standard default from Sonnet to Opus.
  - 2.1.257 added Fable 5.1.
  - 2.1.275 added the `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__` split, and its changelog says the part above the marker is "cached globally".
  - `claude_cli.py` passes `--model opus` and sets `DISABLE_AUTOUPDATER=1`.
- **Corrections:**
  - (a) There are 68 published releases after 2.1.201, not about 80; the version numbers skip.
  - (b) On 2.1.201, `opus` resolves to **Opus 4.8**. The model-config page says `opus` meant Opus 4.8 on the Anthropic API from v2.1.154, and Opus 5 only from v2.1.219.
  - (c) New: a full model ID starting `claude-` passes the local check, but the server enforces a per-model minimum version. It returns `API Error: 400 Claude Code X does not support this model; version Y or newer is required` (code `claude_code_version_too_old`). "Opus 5.5 requires Claude Code v2.1.280 or later."
- **Sources:** raw.githubusercontent.com/anthropics/claude-code/main/CHANGELOG.md; code.claude.com/docs/en/model-config; code.claude.com/docs/en/errors ("Claude Code does not support this model"); the repo's engine/src/maata_engine/claude_cli.py. All read 2026-09-24.

**A2. [confirmed] Policy: personal use of your own signed-in, unmodified CLI is allowed. Third-party apps may not offer claude.ai login or route requests through users' subscription credentials.**
- The legal-and-compliance page (undated) says:
  - OAuth is for Free/Pro/Max/Team/Enterprise purchasers' ordinary use of Claude Code and native apps.
  - Developers building products, including on the Agent SDK, should use API keys.
  - Anthropic "does not permit third-party developers to offer Claude.ai login into their own applications", or to route requests through Free/Pro/Max credentials on behalf of their users, or to collect or intermediate tokens.
  - This does not stop an end user signing in to the unmodified Claude Code binary with their own subscription.
  - Products that preinstall or run Claude Code need the Commercial Terms, an unmodified binary with no removed auth methods, per-user credentials, and no reselling.
  - Advertised Pro/Max limits assume "ordinary, individual usage of Claude Code and the Agent SDK". Enforcement may come without notice.
- The Agent SDK overview note: unless previously approved, third parties may not offer claude.ai login or rate limits for their products.
- The Consumer Terms (effective 2025-10-08, §3) forbid automated access except with an API key or where Anthropic explicitly permits it.
- Anthropic's Thariq Shihipar (GIGAZINE, 2026-02-20): Anthropic wants to encourage local development and experimentation with the Agent SDK and `claude -p`, and a business built on the SDK should use an API key. The draft paraphrased this as "personal/local experimentation is fine", which is a fair summary of the quote.
- **Inference:** a Maata used only by the maintainer fits. A distributed Maata that shells out to each user's own CLI sits between the end-user carve-out and the SDK note, so it needs written confirmation.
- **Sources:** code.claude.com/docs/en/legal-and-compliance (undated; fetched 2026-09-24); code.claude.com/docs/en/agent-sdk/overview; anthropic.com/legal/consumer-terms (effective 2025-10-08); gigazine.net/gsc_news/en/20260220-anthropic-third-party-block/ (2026-02-20).

**A3. [confirmed, one sub-claim unverifiable] Billing: `claude -p` on a subscription still draws on plan limits. The move to API-rate billing was paused.**
- The Register (2026-05-14): from June 15, programmatic use (the Agent SDK, `claude -p` and third-party tools) would get its own monthly credit equal to the subscription fee, billed at API rates, with no rollover.
- Support article 15036540 (the page shows **updated 2026-06-16**, not 06-15): the changes are paused, and the Agent SDK, `claude -p` and third-party apps still draw from subscription limits. An update will be shared before anything takes effect; no new date is given.
- DevOps.com (Tom Smith, 2026-06-18): paused on 2026-06-15, the day it was due, with no new date.
- Winbuzzer (2026-02-19): Anthropic began blocking subscription OAuth in third-party clients on **2026-01-09**, and the docs were clarified on 2026-02-19.
- **[unverifiable]:** "2026-04-04: third-party harnesses cut off from plan limits". No accessible source was found, and the search budget was exhausted.
- **Sources:** support.claude.com/en/articles/15036540 (updated 2026-06-16); theregister.com/ai-ml/2026/05/14/…/5240748 (2026-05-14); devops.com/anthropic-hits-pause-on-claude-agent-sdk-billing-change-for-now/ (2026-06-18); winbuzzer.com/2026/02/19/… (2026-02-19).

**A4. [confirmed] Anthropic publishes the shape of the Pro/Max limits but no counts.**
- The Pro article (updated 2026-09-23): the session limit resets every five hours, and a weekly limit applies across all models. It gives no message or token counts.
- The Max article (updated 2026-09-23): Max 5x and 20x get 5× and 20× Pro's per-session allowance, plus a weekly limit.
- Article 11145838 (updated 2026-08-19): limits are shared across Claude and Claude Code, and usage credits are billed at standard API rates.
- Article 12429409 (updated 2026-08-10): credits are billed at API rates, with an optional monthly spend cap.
- The errors page:
  - Separate limit messages for the session, weekly, Opus and Sonnet limits.
  - Session and weekly limits are shared across models. The Opus and Sonnet limits apply per family, so switching family keeps you working.
  - There is an 85% warning.
  - A transient throttle ("not your usage limit") has been retried automatically since v2.1.199.
- New: the Opus 5.5 announcement (2026-09-22) says five-hour limits are being raised on Pro, Max, Team and seat-based Enterprise. It gives no amount.
- **Sources:** support.claude.com articles 8325606, 11049741, 11145838 and 12429409; code.claude.com/docs/en/errors; anthropic.com/claude-opus-5-5 (2026-09-22).

**A5. [corrected] What a headless run can see about usage.**
- **Confirmed:**
  - `system/api_retry` events carry the error categories `authentication_failed`, `oauth_org_not_allowed`, `account_on_hold`, `billing_error`, `rate_limit`, `overloaded`, `invalid_request`, `model_not_found`, `server_error`, `max_output_tokens`, `cloud_credential_error` and `unknown`.
  - Failures inside a run are printed as the result on stdout, with a non-zero exit.
  - Piped stdin is capped at 10 MB.
  - SIGTERM exits 143 and leaves the turn unfinished; SIGINT ends the turn.
  - Issue #78476 ("expose subscription usage percentages headlessly"), opened 2026-07-17, was **closed as not planned (stale) on 2026-09-05**. The raw event it quotes (2.1.209) has `status`, `resetsAt`, `rateLimitType`, `overageStatus`, `overageResetsAt` and `isUsingOverage`, but no percentage.
- **Correction:** the draft said the event "carries no utilisation percentage". The documented SDK types include:
  - an **optional** `utilization` (0.0–1.0);
  - `rateLimitType` values `five_hour`, `seven_day`, `seven_day_opus`, `seven_day_sonnet` and `overage`;
  - status `allowed`, `allowed_warning` or `rejected`;
  - `errorCode: "credits_required"`, with `canUserPurchaseCredits` (v2.1.181+).
  So parse `utilization` when it appears, but don't depend on it.
- `isUsingOverage` appears only in raw events, not in the typed docs.
- **Sources:** code.claude.com/docs/en/headless; code.claude.com/docs/en/agent-sdk/typescript (`SDKRateLimitEvent`); code.claude.com/docs/en/agent-sdk/python (`RateLimitInfo`); github.com/anthropics/claude-code/issues/78476 (2026-07-17, closed 2026-09-05).

**A6. [confirmed] `--bare` can't use a subscription. Keep `--safe-mode`.**
- The headless doc and CLI reference:
  - In bare mode Claude Code never reads OAuth or the keychain; the Anthropic API needs `ANTHROPIC_API_KEY` or an `apiKeyHelper`.
  - `--bare` "will become the default for `-p` in a future release".
  - `--safe-mode` (added in 2.1.169) turns off CLAUDE.md, skills, plugins, hooks, MCP servers, custom commands and agents, output styles and auto memory. Authentication, model selection, built-in tools and permissions still work.
- Inference: a future default change could break subscription use, so pin and check the CLI version.
- **Sources:** code.claude.com/docs/en/headless; code.claude.com/docs/en/cli-reference; CHANGELOG 2.1.81 (`--bare` added) and 2.1.169 (`--safe-mode` added).

**A7. [confirmed] Fable in `-p` can bill usage credits without asking. Fable is also the slowest model.**
- model-config: depending on plan and seat tier, Fable usage can bill to usage credits. In `-p` and the Agent SDK "Claude Code never shows the consent prompt … bills it without asking."
- The models overview: Fable 5.1 costs $10/$50 per MTok and is "Slower"; Opus 5.5 costs $4/$20 and is "Moderate". Anthropic's own advice is to "start with Claude Opus 5.5 for most workloads".
- **Sources:** code.claude.com/docs/en/model-config; platform.claude.com/docs/en/about-claude/models/overview.

**A8. [corrected] Prompt caching on a subscription.**
- **Confirmed:**
  - Within plan usage, the main conversation gets a 1-hour TTL. On usage credits, an API key or a cloud provider it gets 5 minutes.
  - `CLAUDE_CODE_PROMPT_CACHE_TTL` and `promptCacheTtl` need v2.1.242.
  - Check the TTL used through `usage.cache_creation.ephemeral_1h_input_tokens`.
  - On Opus 5.5 and Fable 5.1 with a subscription, changing effort keeps the cache.
  - Opus 5.5 cache reads cost 5% of base input, which is $0.20/MTok.
  - The boundary-marker split needs 2.1.275 and applies only when calling the Claude API directly.
- **Corrections:**
  - (a) `ENABLE_PROMPT_CACHING_1H` is much older (v2.1.108), not 2.1.242.
  - (b) Cross-process cache hits are **documented, not just inferred**. Any two requests with the same model and prefix read the same cache, and parallel sessions in the same directory read each other's cache. Claude Code prefixes include the working directory, platform and git snapshot, so Maata must keep one fixed working directory. `claude_cli.py` already uses one private `mkdtemp` directory per instance.
- Unknown: whether cached tokens count less against plan limits.
- **Sources:** code.claude.com/docs/en/prompt-caching; code.claude.com/docs/en/cli-reference; code.claude.com/docs/en/agent-sdk/modifying-system-prompts; CHANGELOG 2.1.108 and 2.1.275; platform models overview.

**A9. [corrected] Headless hangs and stalls on macOS.**
- **#91987** (opened 2026-09-04, open, labelled macOS, has repro):
  - `claude -p` hangs indefinitely at startup when an interactive session on the **same CLI version** holds the version-directory lock. The reporter was on 2.1.259.
  - The collision happens only when both processes are on the same version.
  - **Correction:** the issue says `--bare` avoids a *different* hang class (plugin loading), not this one. The reporter found no caller-side workaround.
  - A commenter notes that an immediate retry hung identically and advises failing fast and letting a later attempt retry.
- **#83859** (opened 2026-08-04, open): a constant stall of about 405 s once per headless session, 9 of 9 runs, on macOS Apple Silicon with subscription auth, on 2.1.220 and 2.1.221.
- **#87652** (opened 2026-08-18): first-stream latency regressed between 2.1.197 and 2.1.200 on Linux arm64. It was **closed as not planned (stale) on 2026-09-21**. Its table gives about 6.1 s median total per schema-bound `-p` call on 2.1.201.
- RayBridge PR #10 (2026-09-18, closed): a persistent Claude process cut a repeated question from about 3.1 s to 1.6 s, *including bridge overhead*. This is an anecdote.
- **Sources:** github.com/anthropics/claude-code/issues/91987, /83859 and /87652 (read with `gh`, 2026-09-24); github.com/soothslayer/RayBridge/pull/10.

**A10. [confirmed] `claude_cli.py` misreads current error messages.**
- The errors doc gives these messages:
  - `You've hit your session limit · resets …` (and the weekly, Opus and Sonnet equivalents)
  - `Not logged in · Please run /login`
  - `OAuth token revoked` and `OAuth token has expired`
  - `Failed to authenticate: OAuth session expired and could not be refreshed`
  - `API Error: Server is temporarily limiting requests (not your usage limit)`
- **Checked against the regexes:**
  - `_LIMIT` = `usage limit|rate limit|limit reached|too many requests|overloaded` matches none of the four "hit your … limit" messages.
  - `_LIMIT` does match the transient throttle, through "not your usage limit", so the throttle is misread as a quota.
  - `_AUTH` misses `Failed to authenticate: OAuth session expired…`, because it looks for "authentication" and "oauth token".
- New miss: the version-too-old 400 (A1) falls through to `failed`.
- **Sources:** code.claude.com/docs/en/errors; the repo's claude_cli.py.

**A11. [confirmed] Data use and telemetry.**
- Free/Pro/Max data, including Claude Code, trains models when the setting is on.
- Retention is 5 years with it on and 30 days with it off; the setting is at claude.ai/settings/data-privacy-controls.
- Environment switches:
  - `DISABLE_TELEMETRY` turns off metrics, which never include prompts.
  - `DISABLE_ERROR_REPORTING` turns off error reports, which are on for Pro/Max since v2.1.198.
  - `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` turns off both, and surveys.
  - `DISABLE_FEEDBACK_COMMAND` turns off `/feedback`.
- Local transcripts are kept 30 days unless the session isn't persisted. Maata passes `--no-session-persistence`.
- This does relax the CLAUDE.md rule "no cloud AI" for text.
- **Source:** code.claude.com/docs/en/data-usage.

**A12. [confirmed, plus a 2.1.201 caveat] `--json-schema`.**
- Validation uses draft-07.
- `error_max_structured_output_retries` covers both validation failures and a model-fallback retraction with no successful retry.
- A `success` result with no `structured_output` must be treated as a failure.
- The `format` keyword is an annotation only.
- An invalid schema fails at startup **only since v2.1.205**. Before that, it was silently ignored and schemas containing `format` were rejected. That describes the installed 2.1.201.
- Nuance: `claude_cli.py` falls back to parsing JSON out of the free text when `structured_output` is absent. That accepts *unvalidated* JSON, so validate it against the schema locally or reject it.
- **Sources:** code.claude.com/docs/en/agent-sdk/structured-outputs; code.claude.com/docs/en/cli-reference; CHANGELOG 2.1.205.

## B. Model choice

**B1. [confirmed] The current lineup.** Source: platform models overview, fetched 2026-09-24.

| Model | Price per MTok (in / out) | Latency | Thinking | Default effort | Other |
|---|---|---|---|---|---|
| Fable 5.1 | $10 / $50 | Slower | adaptive, always on | high | |
| Opus 5.5 | $4 / $20 | Moderate | adaptive, always on | **medium** | cache reads 5% of input |
| Sonnet 5 | $2 / $10 | Fast | adaptive | high | retirement ≥ 2027-06-30 |
| Haiku 4.5 | $1 / $5 | Fastest | | | retirement ≥ **2026-10-15** |

- The Opus 5.5 page (2026-09-22) says:
  - output is "more than 30% faster than Opus 5";
  - it can't run with thinking off;
  - fast mode offers up to 2.5×;
  - input and output prices are 20% below Opus 5, and cache reads are 60% below.
- The current tokenizer (since Opus 4.7) fits about 555k words in 1M tokens, against about 750k for older models. That is about 1.8 tokens per word, roughly 1.35× the old tokenizer.
- **Sources:** platform.claude.com/docs/en/about-claude/models/overview; anthropic.com/claude-opus-5-5 (2026-09-22).

**B2. [confirmed within the sources checked] No public EN→TE score exists for any current Claude model.**
- Anthropic's multilingual table has only Sonnet 4.5 and Haiku 4.5 on translated MMLU, with no Telugu. For example, Hindi is 96.7% of English and Bengali 95.4% for Sonnet 4.5.
- The Opus 5.5 page has no multilingual numbers.
- The WMT25 preliminary AutoRank (arXiv 2508.14909 v2, 2025-08-24) has no Telugu pair. Claude-4 sits mid-pack, e.g. rank 7.6 against 5.6 for Gemini-2.5-Pro and 6.5 for GPT-4.1 in one table.
- M-GATE (arXiv 2608.03803, 2026-08-04):
  - Translation tracks log Common Crawl share (r = 0.86).
  - It covers 30 languages, including **Tamil but not Telugu**, and Claude models up to Opus 4.8, Fable 5 and Sonnet 5.
  - New: "Enabling reasoning reliably improves translation."
- The broader absence of a public score couldn't be exhaustively checked (no web search), so choose by bake-off.
- **Sources:** platform.claude.com/docs/en/build-with-claude/multilingual-support; anthropic.com/claude-opus-5-5; arxiv.org/abs/2508.14909; arxiv.org/abs/2608.03803.

## C. Translation for dubbing: evidence

**C1. [corrected] HOMURA / Sand-Glass (arXiv 2601.10187 v3, 2026-09-03; Cui, Yu et al., Bilibili).**
- **Setup:** 1,000 media-domain instances, **Chinese source** into En, De and Es, with syllable-level budgets.
- **Unconstrained, Zh→En in-bounds rate:** Gemini-2.5-Pro 24.9%, GPT-5 23.2%, Claude-4.1-Opus 22.4% (syllable ratio 1.361, CometKiwi 0.741). Into De and Es it is only 8–15.5%.
- **Compression prompt:** Claude reaches 46.1% in bounds, and CometKiwi falls to 0.691.
- **Best-of-N:** four length-graded variants (expanded, base, trimmed, maximally condensed) generated together, then the best fit selected. This reached 61.9–67.6% in bounds on Zh→En.
- **Homura RL** (GRPO on Qwen3-8B, and a 32B variant): 86.3–90.2% on Zh→En, and 84.6–87.3% on De and Es.
- **Correction:** "back-translation fidelity drops to 0.53–0.61 under Best-of-4" is wrong. Those numbers are CometKiwi on Zh→De. Best-of-N's BT-CERR stays about 0.90 on Zh→En. The paper ties semantic loss ("hallucinated omissions") mainly to **two-stage post-editing** (translate, then rewrite for length). Claude reached only 38.3% in bounds that way. The paper also flags Best-of-N's latency and "cliff" gaps between candidates.
- **Source:** arxiv.org/abs/2601.10187 and arxiv.org/html/2601.10187 (v3, 2026-09-03).

**C2. [confirmed] Length control needs clearly short demos.** Javorský, Bojar and Yvon (arXiv 2506.04855, 2025-06-05):
- 8 open LLMs, En→De/Fr/Es, IWSLT 2022 isometric task (±10% of source characters).
- Models shorten only with extreme ("Short" or "Tiny") demos. Isometric demos get ignored.
- Going from 5 to 10 to 20 shots adds only marginally.
- Selecting from multiple outputs improves the length–quality trade-off.
- **Source:** arxiv.org/abs/2506.04855.

**C3. [confirmed] VideoLingo** (Apache-2.0).
- `core/prompts.py` steps:
  1. A two-sentence summary plus a term list with target renderings.
  2. A faithful translation with context.
  3. "Expressiveness": reflect, then a free translation.
  4. Split the source into N parts, generating two approaches and choosing one.
  5. Align.
  6. A subtitle trim given duration, removing fillers and modifiers.
- Every step returns JSON.
- The README (commit 2026-09-23) recommends `claude-opus-5-5` "for best quality" via a relay. It warns that speed adjustment doesn't guarantee natural delivery or sync, and that dubbing doesn't assign per-speaker voices.
- **Sources:** github.com/Huanshere/VideoLingo (README, LICENSE, core/prompts.py; repo pushed 2026-09-23).

**C4. [confirmed, with a scope caveat] Refinement mostly improves fluency, not adequacy.** Tan, Zhu, Tran, Denkowski, Trenous and Byrne (arXiv 2605.13368, 2026-05-13):
- 9 LLMs, 7 language pairs, 9 granularity combinations, 5 strategies, human evaluation.
- Document-level MT followed by segment-level refinement works best.
- A simple generic refine prompt beats error-specific and evaluate-then-refine prompts.
- Gains are mainly fluency, style and terminology. Adequacy gains are limited and inconsistent.
- Refiners pull output toward their own distribution.
- **Caveat:** this is **literary** document translation, not spoken dubbing.
- **Source:** arxiv.org/abs/2605.13368.

**C5. [confirmed] Human dubbers fit time by changing content, not rate.** Brannon, Virkar and Thompson (TACL vol. 11, 2023):
- 319.57 h, 54 professional titles.
- The duration ratio correlates with the word-length ratio (r = 0.523), not with speaking rate (r = 0.163).
- The dub's speaking-rate spread is lower than the source's: 1.25 vs 1.47 w/s for Spanish, 1.26 vs 1.46 for German.
- Median overlap with the source is 0.731 (mean 0.658). Off-screen overlap is 0.662 against 0.684 on-screen.
- The abstract argues that vocal naturalness and translation quality matter more than isometric length and lip-sync.
- **Source:** arxiv.org/abs/2212.12137.

**C6. [corrected] Off-screen dubbing can relax timing.** Virkar, Federico, Enyedi and Barra-Chicote (arXiv 2204.02530, 2022-04-06):
- They extend prosodic alignment to off-screen dubbing, with looser synchronisation. The model optimises speaking-rate smoothness across contiguous segments.
- The extended model gave significantly better subjective viewing on TED talks and 3 public YouTube videos, En→Fr/It/De/Es.
- **Correction:** the arXiv entry says **"Submitted to Interspeech 2022"**, not ICASSP 2022.
- **Source:** arxiv.org/abs/2204.02530.

**C7. [corrected] ElevenLabs and YouTube.**
- **ElevenLabs Dubbing v2** (blog 2026-05-28, updated 2026-09-06):
  - The blog says translation is adapted for spoken delivery while keeping sync, and that a "sync-aware translation system" aligns starts, stops and pacing.
  - It says the system conditions on the original performance.
  - It doesn't disclose the architecture or the LLM.
  - It doesn't mention Telugu specifically (90+ languages).
- **YouTube** help 15569972 (undated): English→Telugu auto-dubbing is listed **with expressive speech**. YouTube names proper nouns, idioms and jargon as difficulties.
- YouTube blog (2026-02-04): 27 languages, 8 of them expressive (Telugu not yet), and a lip-sync pilot.
- **Corrections / unverifiable:**
  - "The Dubbing Studio editor is in maintenance mode" is not in the cited blog: [unverifiable].
  - "YouTube says the feature is Gemini-based (2026-02-04)": the cited post has no mention of Gemini: [unverifiable].
- **Sources:** elevenlabs.io/blog/introducing-dubbing-v2; support.google.com/youtube/answer/15569972; blog.youtube/news-and-events/youtube-auto-dubbing-expressive-speech/ (2026-02-04).

**C8. [confirmed] LLMs handle idioms and puns worst.** Van Doren et al., "Be My Cheese?" (arXiv 2602.04729, v2 2026-05-27):
- 7 LLMs, 15 locales, 5 native raters each.
- Mean full-text quality 1.68/3. GPT-5 scores 2.10, Claude Sonnet 4 1.97 and Mistral Medium 3.1 1.84.
- By segment type: holidays 2.20, cultural concepts 2.19, idioms 1.65, puns 1.45. Idioms are the most often left untranslated.
- Krippendorff α = 0.45.
- **Telugu is not among the locales.** Hindi (IN) and Urdu (PK) are the only Indic ones.
- **Source:** arxiv.org/abs/2602.04729 (PDF v2).

## D. Telugu

**D1. [corrected] How much English Telugu podcasters mix in.** CoSTA (Shankar, Jyothi and Bhattacharyya, arXiv 2406.10993, 2024-06-16):
- A Telugu **podcast** set of 2.5 h and 624 instances, CMI 32.14%. The creators gave permission.
- It is highly conversational, multi-speaker and disfluent.
- A separate IndicVoices-derived Telugu code-switched set, *chosen for high CMI*, has CMI 25.5%.
- The authors *intend* to release the sets under CC-BY-4.0.
- **Correction:**
  - CMI is not "the share of tokens that are English". CMI ranges 0–50%, with 50 an equal mix; it measures the non-dominant language's share per utterance.
  - "About a third English tokens" is only a rough reading.
  - The podcast set comes from **one** Telugu podcast.
- The academia.edu "39.74%" study is [unverifiable] (HTTP 403).
- **Source:** arxiv.org/html/2406.10993v1.

**D2. [confirmed] English verbs in Dravidian speech use a host light verb.**
- Kulkarni, "Borrowing and disappearance of light verbs" (FASAL-14, JSAL):
  - Kannada and Bangla need a light verb for borrowed English verbs (citing Amritavalli 2017).
  - Citing Annamalai (1989), Tamil imbalanced bilinguals use the English noun + pannu ('confusion-pannu'), and balanced bilinguals the English verb ('confuse-pannu').
  - Nuance: Kulkarni's main point is that Hindi and Marathi can integrate English verbs *directly*.
- Moradi (Indian Journal of Applied Linguistics 2014, ERIC EJ1039518): in Persian and Telugu bilingual light-verb constructions, an L1 light verb or its inflected form attaches to an English noun, adjective, adverb, preposition or verb.
- The Telugu mapping (cheyyu for transitive, avvu for intransitive) remains an inference for the native-speaker review.
- **Sources:** ojs.ub.uni-konstanz.de/jsal/…/Kulkarni-FASAL-14 (PDF); eric.ed.gov/?id=EJ1039518.

**D3. [confirmed] Register and grammar.**
- Gidugu Venkata Ramamurthy championed vyāvahārika (spoken) over grāndhika (scholastic). "Sishta vyavaharika" (standard spoken) gained acceptance.
- Telugu is SOV, head-final and pro-drop, with postpositions.
- It has four regional dialects: Telangana, Rayalaseema, Coastal Andhra and North Andhra (Krishnamurti and Gwynn 1985). Standard Telugu is based on the Central dialect.
- In the 2nd person, mīru (plural or polite) and nuvvu (singular) take distinct verb agreement.
- Rani (IJAL 2010, EJ965826): conjunctive participles, relative participles, quotatives and co-reference hold spoken Telugu narratives together (50 narratives).
- Krishna et al. (ACL 2022): 1,000 formality-annotated pairs in each of Hindi, Bengali, Kannada and Telugu.
- **Sources:** en.wikipedia.org (Telugu grammar, Telugu language, Gidugu Venkata Ramamurthy; raw wikitext 2026-09-24); eric.ed.gov/?id=EJ965826; arxiv.org/abs/2110.07385.

**D4. [confirmed, provenance added] The speaking-rate seed is too fast.**
- `timing/duration.py` sets `DEFAULT_RATE = 5.5` aksharas/s, marked provisional. The model is duration ≈ overhead (0.15 s) + units/rate, updated by Huber-weighted recursive least squares.
- `session.py` sets `target_units = seconds × estimator.rate(speaker) × 0.95`, and 0.9 for re-synthesis.
- `tts_pace.json`: 3.18–3.54 aksharas/s over 81–91 s at six exaggeration/cfg settings.
- **Provenance** (read from `tts_pace.py`):
  - The **podcast guest's** cloned voice only: pyannote diarization, then a SpeakerRegistry reference of about 10 s.
  - The first 10 Telugu lines of `bench_e2e/localbench0`, with seed 0.
  - Durations include each line's own leading and trailing silence, so the pure rate is slightly higher.
  - The host voice was not measured.
- The "~5.5 natural pace" in that script's docstring has no source.
- Durisala et al. (Indian J Otolaryngol, 2011; one female speaker): conversational Telugu words average 0.43 s against 0.68 s in clear speech.
- **Sources:** the repo files; the scratchpad's tts_pace.py and tts_pace.json; pmc.ncbi.nlm.nih.gov/articles/PMC3102160/.

## E. Maata repo and scale

**E1. [corrected] One call per line, and the scale of calls and tokens.**
- **Repo facts, confirmed:**
  - `_translator` translates one unit per request, about `TRANSLATE_AHEAD = 60` s ahead.
  - It retries with candidates when a line doesn't fit.
  - It has an "echo" guard.
  - `claude_cli.py` isn't imported anywhere (only a stale .pyc).
- The call-count estimates are inference and still plausible:
  - 600–800+ calls per hour today;
  - about 45–65 with scene batching.
- **Correction to the token estimate:** Petrov et al. (NeurIPS 2023) give a Telugu premium of **8.34× on cl100k_base (GPT-4)**, 10.71× on LLaMA and 13.09× on GPT-2 (FLORES-200), not about 5×. Claude's Telugu ratio is unpublished. The output-token range should therefore be treated as roughly 100–300k tokens per hour *before thinking*, if each line comes back in 2–3 length variants. That is an inference to measure, not a number to plan on. The API-price equivalent may be above the $2–6 per hour estimate.
- **Sources:** the repo's session.py and claude_cli.py; arxiv.org/abs/2305.15425 (PDF, appendix tables).

---

## New findings the draft missed

1. **A version floor is needed even with full model IDs.** On 2.1.201 the server rejects Opus 5.5 with `400 claude_code_version_too_old`, and `opus` means Opus 4.8. Detect this, and show "run `claude update`". *(errors, model-config)*
2. **The schema bug on 2.1.201.** An invalid `--json-schema` is silently ignored, and `format` causes a rejection. Fixed in 2.1.205. *(structured-outputs, CHANGELOG)*
3. **Fast mode costs extra.** It is usage-credits-only on Pro/Max ($8/$40 per MTok on Opus 5.5), and an interactive `/fast` toggle can persist in user settings. In `-p`, fast mode applies only when a session is launched with it in `--settings`. Pass `--settings '{"fastMode": false}'` defensively; the effect of safe mode on this needs testing. *(fast-mode doc)*
4. **Opus 5.5 is covered on Pro.** Opus 4.7+ and Sonnet 5 run with 1M context on every plan including Pro, with no credits. Only Fable (and fast mode) can bill credits. This answers the draft's open question about Opus 5.5 on Pro. *(model-config "Extended context")*
5. **A richer `rate_limit_event`.** It has optional `utilization`, the window types `seven_day_opus` and `seven_day_sonnet`, and `credits_required`. *(SDK references)*
6. **Use graded variants in one call** rather than a translate-then-rewrite pipeline. HOMURA's best prompt-only method generates length-graded variants in a single call. The two-stage rewrite is where omissions appear. *(HOMURA)*
7. **Reasoning helps translation** (M-GATE). Don't assume the lowest effort is enough; include effort in the bake-off.
8. **Retrying a hung call is futile.** A retry against a held lock hangs the same way (#91987 comment). Back off, and tell the user that an interactive Claude Code session on the same version can block Maata.
9. **The cache is keyed to the directory.** Keep one fixed working directory per engine session and a byte-identical static system prompt, so separate `-p` processes share the 1-hour cache. *(prompt-caching)*
10. **A cheap external reference exists.** YouTube's own EN→TE auto-dub (expressive) exists and names idioms and proper nouns as weak spots.

## Refuted (sub-claims)

- "`--bare` avoids the #91987 lock hang (but drops OAuth)". The issue says `--bare` avoids a *different* hang class. No caller-side workaround is known.
- "HOMURA Best-of-4 drops back-translation fidelity to 0.53–0.61". Those are CometKiwi scores on Zh→De. Best-of-N BT-CERR is about 0.90 on Zh→En.

## Unverifiable (sub-claims)

- 2026-04-04 "third-party harnesses cut off from plan limits". The Winbuzzer and GIGAZINE pages give only January and February 2026 dates.
- The ElevenLabs Dubbing Studio editor is "in maintenance mode". This is not in the cited blog.
- YouTube auto-dubbing is "Gemini-based (press 2026-02-04)". The cited post never mentions Gemini.
- Andhra Pradesh college students' "39.74% mixing" (academia.edu, 403).

---

## Recommendations for Maata (updated)

**R1. Scene-batched, multi-step translation on the CLI.** High impact, L effort. Record as an ADR with the no-local-LLM decision.
1. **Brief:** after ASR and diarization, one call covers the whole English transcript. It returns:
   - the topic;
   - a glossary (keep in English, or its Telugu rendering);
   - named entities;
   - speaker roles and the meeru/nuvvu register for each speaker pair;
   - flagged idioms, puns and jokes;
   - a numbers convention.

   Put the brief in a **byte-identical static system prompt** for the video. Use the `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__` split, which needs v2.1.275 or later.
2. **Scene call:** one call per 2–3 minute scene, cut at sentence boundaries. It includes the previous scene's final Telugu. It returns JSON `{lines:[{id, te, te_short, te_min}]}`, with a per-line akshara budget computed from each voice's *measured* rate.
   - Ask for 2–3 **length-graded variants in the same call**, which is HOMURA's best prompt-only method.
   - Make the demos for the short versions visibly shorter than their English (Javorský).
3. **Deterministic fit:** choose the most complete variant that fits, using the per-voice estimator. Only lines still over budget go into one batched "fit" call per scene, carrying the measured overflow.
4. **Deterministic coverage check** after every step, and especially after any rewrite:
   - every id is present;
   - numbers and names are preserved;
   - a minimum length relative to the English.

   Run an LLM back-check only on flagged lines. Refinement won't catch omissions (Tan), and rewrite passes are where omissions appear (HOMURA).

**R2. Size by measured voice rate.** High impact, S effort.
- Calibrate each cloned voice at session start: synthesise 2–3 fixed Telugu sentences per speaker, and seed the DurationEstimator's rate *and* overhead from them.
- Replace the 5.5 default with about 3.4 as the prior.
- Re-measure `tts_pace` for the host voice and for conversational lines before relying on 3.2–3.5 everywhere.

**R3. Translate whole sentences, then spread across slots.** High impact, M effort. Record the sync policy in an ADR.
- Merge ASR fragments into sentences before translating.
- Split the Telugu across the English slots by duration share, at Telugu clause boundaries: after a conjunctive participle, after the quotative ani, or after a finite verb.
- Treat all speech as off-screen, with sentence-level sync and a smooth speaking rate (Virkar 2022; Brannon 2023).

**R4. A style guide in the system prompt.** High impact, S effort. The maintainer reviews the examples as a native speaker.
- Colloquial standard Telugu (sishta vyāvahārika, based on the Central dialect), avoiding grāndhika vocabulary and strong regional markers.
- English only where educated speakers use it.
- English verbs as the bare verb + cheyyu or avvu. This is an inference to be checked by the maintainer.
- Fixed register per speaker pair, taken from the brief.
- Numbers written as they are spoken.
- Idioms and puns become a Telugu equivalent or a paraphrase, never a literal translation.
- Log CMI per scene as a **monitor**. Warn only well outside about 20–40%, given 25.5% for IndicVoices and 32% for the podcast set, and never enforce a quota.

**R5. Harden `claude_cli.py`.** High impact, M effort.
- **Version gate at startup.** Parse `claude --version`:
  - require **≥ 2.1.280** for Opus 5.5 (and ≥ 2.1.205 in any case, for schema validation);
  - otherwise show "run `claude update`";
  - keep `DISABLE_AUTOUPDATER`;
  - log the resolved model from `modelUsage`, which may list several models after a fallback.
- **Model flags.** Use `--model claude-opus-5-5` (the full ID) and `--fallback-model claude-sonnet-5`. The fallback covers overload and unavailability, not usage limits.
  - On "You've hit your Opus limit", re-run the scene with `--model claude-sonnet-5`.
  - Never use `fable`, `best` or fast mode. Pass `--settings '{"fastMode": false}'` and verify it on the M5 Pro.
- **Error classification:**
  - `hit your (session|weekly|opus|sonnet) limit` means a usage limit; parse "resets …".
  - `Failed to authenticate`, `OAuth (token|session) … (expired|revoked)` and `Not logged in` mean not signed in.
  - `temporarily limiting requests` is transient: retry with backoff.
  - `does not support this model` or `claude_code_version_too_old` means the CLI needs updating.
- **Structured output:**
  - treat `success` without `structured_output` as a failure;
  - stop accepting free-text JSON unless it validates locally against the schema.
- **Rate-limit events.** Use `--output-format stream-json --verbose` and read `rate_limit_event`:
  - status `allowed_warning` or `rejected`;
  - `rateLimitType`;
  - `utilization` when present;
  - `credits_required`;
  - `isUsingOverage` / `overageStatus` in the raw event.

  Pause with a UI message whenever overage would be used, unless the maintainer opts in.
- **Startup watchdog:**
  - no stream event within about 60–90 s means SIGINT, then SIGTERM, then SIGKILL after a grace period;
  - **back off rather than retrying immediately**;
  - tell the user an interactive Claude Code on the same version may be blocking (#91987).
- **Process model.** Prefer one sealed `-p` process per call, with a fixed working directory. A warm `--input-format stream-json` process saves roughly a second or two per call (anecdote), but it accumulates conversation history across scenes. Only adopt it if the measured spawn overhead matters. If adopted, recycle it per scene or per few scenes.
- **Logging.** Record input, output, cache-read and cache-creation tokens (including `ephemeral_1h_input_tokens`) per call. Commit a JSON from a real 1-hour run on the M5 Pro. `DISABLE_FEEDBACK_COMMAND=1` is harmless but adds little, since slash commands are already disabled.
- **Approval:** no new dependency. Whether Maata may ever spend usage credits is the maintainer's call.

**R6. Policy and disclosure.** High impact, S effort. **Needs maintainer approval**, because it changes a CLAUDE.md hard constraint.
- Sign in only through `claude auth login`.
- Never read `~/.claude` credentials, `CLAUDE_CODE_OAUTH_TOKEN` or `setup-token` output. Never bundle or patch the CLI.
- Show a one-time notice that transcript text goes to Anthropic under the user's plan, with a link to claude.ai/settings/data-privacy-controls.
- Before any public release, add an own-API-key option and get Anthropic's written answer on shelling out to a user's own signed-in CLI.
- Record the "no cloud AI" relaxation for text in an ADR.

**R7. Blind bake-off.** Medium impact, S effort.
- Use the ADR-018 rubric (naturalness, completeness, fit rate) on the podcast fixture.
- Arms:
  - Opus 5.5 at low effort;
  - Opus 5.5 at medium (the default);
  - Sonnet 5 at medium or high.
- Use the same prompts. Include YouTube's EN→TE auto-dub as a listen-only reference where one exists; never commit it.
- Pick the model and effort per step: brief, scene translation and fit.

**R8. No generic "review everything" pass.** Low impact, S effort. If a review pass is added later, run it at segment level with a simple prompt, and only if the bake-off shows a naturalness gain. Keep the deterministic fit and coverage checks as the safety net.

## Open questions

1. Which plan does the maintainer have (Pro, Max 5x or Max 20x), and how many video-hours a week does he expect to dub? No counts are published, and the size of the 2026-09-22 five-hour limit increase is unstated.
2. How many tokens does Telugu cost on Claude's current tokenizer? Measure through `usage` on the first real calls. Do cached tokens count less against plan limits?
3. Would Anthropic treat a distributed Maata that shells out to each user's own unmodified CLI as the allowed end-user sign-in, or as offering claude.ai login or rate limits? This needs a written answer (contact sales).
4. When will `--bare` become the default for `-p`, and will there be a way to keep subscription sign-in?
5. Does `--settings '{"fastMode": false}'` override a persisted fastMode under `--safe-mode`? Test on the M5 Pro.
6. How fast is the *host* voice and conversational Telugu? `tts_pace.json` covers the guest only, over 10 lines.
7. Is there a published normative Telugu rate in aksharas or syllables per second? Savitri and Jayaram (2008) couldn't be accessed. Durisala (2011) is word-level and has one speaker.
8. Will Anthropic reintroduce separate metering for `claude -p` and the Agent SDK, and when?
9. Can the #91987 lock hang be avoided by running Maata's CLI from a different installed version than the maintainer's interactive CLI? This is untested and would need its own pinning ADR.

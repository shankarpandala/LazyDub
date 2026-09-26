# Gap 6: YouTube's API terms and Anthropic's CLI terms for Maata

Researched 2026-09-24. Primary sources were fetched twice: once with curl (saved under `scratchpad/gap6/`) and once with WebFetch. Every quoted clause matched in both fetches. **DISCLOSED** means the text is in a primary source. **INFERRED** means it is this report's reading. This is not legal advice: it is desk research to help the maintainer brief counsel.

## Question

1. Do the YouTube API Services Terms and Developer Policies apply to Maata's design? Maata embeds the IFrame player with no API key, keeps it muted, and plays a Telugu dub made from audio that yt-dlp fetched.
2. If they apply, does III.I.7 ("must not apply alternate audio tracks to videos") cover the core design, and not just a background bed?
3. Will Anthropic allow a distributed app that shells out to each user's own signed-in `claude` CLI? Or is an API-key translation path required?

## Answer

### YouTube

**1. Yes, the terms apply, and Maata is an "API Client". Not having a key changes nothing.** (DISCLOSED; high confidence)
- **The definition fits Maata.** An API Client is "a website or software application … that accesses or uses the YouTube API Services". Nothing in the definition requires credentials.
- **The embed is a named API service.** The Developer Policies list "The YouTube IFrame Player API service" as a YouTube API service that "does not require authorization" (III.D.2).
- **Desktop apps are covered by name.** Required Minimum Functionality says API Clients that use the embedded player, "including the YouTube IFrame Player API", must identify themselves through the HTTP Referer. It names "WebView integrations such as a mobile app or desktop app". A Tauri app is exactly that.
- **YouTube enforces this.** Player error 153 means the request carried no Referer "or equivalent API Client identification". YouTube added this section on 2025-07-07.

**2. On a plain reading, the core design is covered, whatever happens with background audio.** (medium-high confidence; no YouTube ruling on this pattern was found)
- **III.I.7:** muting the video and playing a synced substitute voice track is what "apply alternate audio tracks to videos" describes.
- **The compliance guide goes further.** It tells API services not to "allow users to modify the audio or video portions of a video".
- **III.I.16 (missed by all ten reports):** it bars you to "modify, translate, create derivative works of" any YouTube API Services. The definition of YouTube API Services includes API Data, which includes audiovisual content. A Telugu dub is a translation.
- **The input is barred even if III.I.7 were read narrowly.** yt-dlp breaches:
  - III.I.14: no technology other than YouTube API Services may be used to access audiovisual content;
  - III.E.1.a: no downloading, caching or storing without prior written approval (this clause also covers "enable … others");
  - III.D.6: no undocumented APIs;
  - III.E.6: no scraping;
  - the YouTube Terms of Service, which bind every viewer: no downloading or altering content "except … as expressly authorized", no "automated means", and no circumventing features that restrict copying.
- **The strongest counter-arguments:**
  - `mute()` is a documented API call;
  - the dub plays in Maata's own audio element, so nothing is "applied" to YouTube's stream;
  - the guide praises third-party "captioning services for the hearing impaired" as independent value.

  None of these reaches the yt-dlp clauses.
- **Consequence:** the background-audio question is secondary. Every option, including a dry dub, rests on the same challenged design.

**3. A "personal use only" build does not escape the text, but it removes the distribution clauses.**
- **Neither document exempts personal use.** The Developer Policies have no personal, hobby or non-commercial exception; WebFetch was asked to check and confirmed this. The YouTube Terms allow personal, non-commercial viewing and showing videos through the embeddable player, but they bar downloading and modifying for everyone.
- **What changes for a personal build (INFERRED):** it no longer "enables others". It no longer depends on a signed installer from an identifiable publisher. It is exposed only through one person's traffic.
- **Size of the remaining risk:** this report found no source on how often YouTube enforces against an individual's personal use, and does not estimate it.

**4. III.I.7 and III.I.14 cannot be waived through the approval form.** (DISCLOSED)
- The policies define "must not" as "an absolute prohibition".
- The written-approval route is written into III.E.1.a (downloading) and the III.G sales items, not into III.I.7 or III.I.14. The approval form is linked from III.G.
- A bespoke agreement with YouTube is the only clean route (INFERRED). No evidence was found on whether one could be obtained.

### Anthropic

**5. Report 05 A2 is out of date: there is now a written answer, and it largely covers Maata.** (DISCLOSED; high confidence on the text, medium on how it applies)
- **When it appeared.** The Claude Code legal page gained a section, "Can customers offer Claude Code in their products?", between 2026-08-16 (absent from the Wayback capture) and 2026-08-30 (present).
- **The conditions it sets.** "preinstalling or running Claude Code in your products" needs:
  - acceptance of the Commercial Terms;
  - an unmodified binary, with no authentication method removed, disabled or restricted;
  - no paying for, reselling or intermediating end users' usage.

  Each end user must authenticate with their own API key, **"Claude subscription plan credentials"**, or cloud-provider credential.
- **An explicit carve-out.** The same update says the ban on routing requests through plan credentials does not stop "an end user from signing in to the unmodified Claude Code binary with their own Claude subscription".
- **A new rule.** Developers "may not collect, store, or intermediate Claude.ai credentials or session tokens"; sign-in must go through Anthropic's own flow.

**6. So an API-key path is not required by policy.** (INFERRED from 5)
- A distributed Maata that spawns the user's own installed, unmodified `claude`, signed in by that user through the CLI's own flow, fits the written conditions. Two conditions apply: the maintainer accepts the Commercial Terms, and Maata never handles tokens.
- Users without a plan can sign the same CLI in with their own API key, with no Maata-specific key path.

**7. Three risks remain, and they are the questions to put to Anthropic sales.**
- **Usage limits.** Pro and Max limits "assume ordinary, individual usage". Anthropic may enforce "without prior notice".
- **Billing.** The planned move of `claude -p` and Agent SDK use onto separate API-rate credits is paused, not cancelled. Support article 15036540 was updated 2026-06-16 and gives no new date.
- **Wording.** The developer bullet still says product builders "should use API key authentication".

**8. With no local LLM, a released Maata cannot translate without an Anthropic account and a network connection.**
- This contradicts two current rules in CLAUDE.md: "On-device inference only: no cloud AI" and "the engine makes no network calls except yt-dlp".
- Both need an ADR or amendment that records today's decision. This is separate from any terms question.

## Evidence

### A. YouTube (all fetched 2026-09-24)

| # | Source (date) | What it says (DISCLOSED) |
|---|---|---|
| Y1 | YouTube API Services Terms of Service, developers.google.com/youtube/terms/api-services-terms-of-service (last updated 2026-09-14) | "API Client" means a website or software application developed by you that "accesses, or uses" the YouTube API Services. "YouTube API Services" includes (iii) "data, content (including audiovisual content)" provided to API Clients ("API Data"). You are bound "By accessing and using" them. §7: each API Client must publish a privacy policy. §2.4: the ToS take precedence over the other documents. |
| Y2 | Developer Policies, developers.google.com/youtube/terms/developer-policies (last updated 2026-09-14) | **III.D.2:** the IFrame Player API service needs no authorization. **III.D.6:** "You must not use undocumented APIs without express permission." **III.E.1:** you and your API Clients must not "encourage, enable, or require others to" (a) download, cache or store audiovisual content without prior written approval, or (b) make content available offline. **III.E.6:** no scraping. **III.I.1:** no substitute for YouTube. **III.I.6:** do not modify any player functionality. **III.I.7:** do not separate, isolate or modify audio or video components ("you must not apply alternate audio tracks to videos"). **III.I.14:** no technology other than YouTube API Services to access audiovisual content. **III.I.16:** do not "modify, translate, create derivative works of" YouTube API Services. **Terminology:** "must not" is "an absolute prohibition". No personal-use exception. Numbering was checked against the page's list structure. |
| Y3 | Required Minimum Functionality, …/terms/required-minimum-functionality (last updated 2026-09-14) | **API Client Identity and Credentials:** embedded-player clients, including IFrame API clients, must send a Referer. For desktop WebViews it must be `https://<app ID>`. Non-compliance "might result in reduced functionality". No overlays in front of the player. Minimum size 200×200 px. |
| Y4 | IFrame Player API reference (last updated 2026-09-15) | Error **153**: no Referer "or equivalent API Client identification". |
| Y5 | Complying with YouTube Developer Policies (guide; last updated 2026-09-14) | Do not "allow users to modify the audio or video portions of a video". Do not separate or isolate audio. Do not diminish standard features "such as captions, volume controls". Mute and play overlays outside the player UI are acceptable. Third-party captioning for the hearing impaired is cited as "independent value". Do not fail to provide API Client identification. |
| Y6 | Revision history, …/terms/revision-history (page last updated 2026-09-11) | **2025-07-07:** added API Client Identity and Credentials. **2022-06-02:** added III.I.21. **2026-05-04 and 2026-06-01:** derived-metrics policies. No substantive entry after 2026-06-01. The "last updated 2026-09-14" stamp on the policy pages is therefore not a policy change (INFERRED). |
| Y7 | Quota and Compliance Audits, …/v3/guides/quota_and_compliance_audits (last updated 2026-09-14) | Audits gate Data API quota above the default. YouTube also runs periodic audits. There is no audit path for a keyless embed with no API project (INFERRED). Enforcement would instead come through Y3/Y4 identification and technical measures. |
| Y8 | YouTube Terms of Service, youtube.com/static?template=terms (US, effective 2023-12-15) | Personal, non-commercial viewing and "the embeddable YouTube player" are allowed. Barred: downloading or modifying content except "as expressly authorized by the Service" or with written permission from YouTube and rights holders; "automated means"; and circumventing features that "prevent or restrict the copying". |
| Y9 | yt-dlp wiki "EJS" (undated; fetched 2026-09-24) | Downloading from YouTube requires solving "JavaScript challenges presented by YouTube" with an external JS runtime. This bears on Y8's circumvention clause (INFERRED). |
| Y10 | RIAA §1201 notice to GitHub (2020-10-23); GitHub blog "youtube-dl is back" (2020-11-16) | The RIAA said youtube-dl circumvents YouTube's "rolling cipher" TPM. GitHub reinstated it after receiving information that it did not circumvent TPMs. The anti-circumvention theory has been asserted against this tool family, and it is a distribution-level risk for counsel. |
| Y11 | 17 U.S.C. §101 (Cornell LII, fetched 2026-09-24) | A "derivative work" includes "a translation". The dub is a derivative of the creator's work, separately from YouTube's terms. |
| Y12 | Repo, read 2026-09-24 | `app/src/lib/player.ts` loads `https://www.youtube.com/iframe_api` with host youtube-nocookie.com, `mute: 1`, `origin: location.origin`. ADR-006 serves the UI from `http://127.0.0.1:<port>/`, so the Referer is a loopback URL, not `https://io.github.shankarpandala.maata` as Y3 asks. `engine/src/maata_engine/resolve.py` stores the fetched audio under `cache_dir/<video_id>` (III.E.1.a). |

### B. Anthropic (all fetched 2026-09-24)

| # | Source (date) | What it says (DISCLOSED) |
|---|---|---|
| A1 | code.claude.com/docs/en/legal-and-compliance (undated). Wayback: the product section is absent 2026-08-16 10:07 UTC and present 2026-08-30 09:47, 2026-09-10 and 2026-09-23. | **"Can customers offer Claude Code in their products?":** needs the Commercial Terms, an unmodified binary with no auth method removed, disabled or restricted, and no paying, reselling or intermediating. Each end user brings their own API key, subscription credentials or cloud credential. **Authentication and credential use:** OAuth is for plan purchasers' "ordinary use". Developers building products "should use API key authentication". No offering Claude.ai login or routing through plan credentials "on behalf of their users". No collecting, storing or intermediating tokens. **Carve-out:** end users may sign in to the unmodified binary with their own subscription, "including where a platform hosts Claude Code". Limits "assume ordinary, individual usage". Enforcement "without prior notice". Questions go to sales. |
| A2 | Same page, Wayback captures 2026-02-19 to 2026-08-04 | The older text had only the OAuth and developer bullets, with no product section, no carve-out and no token clause. Report 05 A2 summarised a mix of old and new text. |
| A3 | code.claude.com/docs/en/agent-sdk/overview | "Unless previously approved", no claude.ai login or rate limits for third-party products built on the Agent SDK. Maata calls the CLI, not the SDK. |
| A4 | code.claude.com/docs/en/authentication (identical in two fetches) | `claude setup-token` makes a one-year subscription token "For CI pipelines, scripts". Anthropic therefore explicitly permits scripted use of a subscription through Claude Code. `--bare` does not read OAuth. |
| A5 | Consumer Terms, anthropic.com/legal/consumer-terms (effective 2025-10-08) | No "automated or non-human means" except via an API key "or where we otherwise explicitly permit it". A4 and the A1 carve-out are plausibly that permission (INFERRED). |
| A6 | Commercial Terms, anthropic.com/legal/commercial-terms (effective 2025-06-17) | Customers may use the Services "to power products and services" for their own users. No reselling "except as expressly approved". |
| A7 | support.claude.com/en/articles/15036540, "Use the Claude Agent SDK with your Claude plan" (updated 2026-06-16) | The Agent SDK and `claude -p` billing changes are paused, with no new date. |
| A8 | Repo `engine/src/maata_engine/claude_cli.py` (read 2026-09-24) | Runs `claude -p` with `--tools ""`, `--safe-mode` (which keeps subscription sign-in), `--no-session-persistence` and `--strict-mcp-config`. It handles no tokens and does not pass `--bare`. This is already consistent with A1. |

### C. Background-audio evidence (re-checked)

Federico et al. 2020 (arXiv 2001.06785; the saved copy was re-read) reports:
- +10.34 MUSHRA from adding the original background and reverb, for non-native listeners (significant);
- +1.05 for native listeners (not significant).

Maata's audience is native Telugu speakers, so the only controlled evidence for a bed does not show a benefit for them.

## Confidence

| Claim | Confidence | Why |
|---|---|---|
| Maata (a Tauri WebView plus the IFrame API, keyless) is an API Client bound by the API ToS and Developer Policies | **High** | Three primary documents (Y1 to Y3) and an enforcement error code (Y4) |
| The yt-dlp fetch conflicts with III.I.14, III.E.1.a, III.D.6, III.E.6 and the YouTube ToS | **High** | Plain text, unambiguous |
| The muted embed plus synced dub falls under III.I.7 ("alternate audio tracks") | **Medium-high** | The literal example fits, and III.I.16 and Y5 agree. There is no YouTube ruling, and counter-arguments exist. |
| A separated background bed (option C) is the most exposed option | **High** | "separate, isolate" plus the guide's example |
| The personal-use build is also outside the text | **High** | No exception exists. The size of the risk is not estimable from sources. |
| Anthropic's text permits a distributed app that runs the user's own unmodified CLI with the user's own subscription | **Medium-high** | The late-August 2026 text addresses it directly, but still says "should use API key" and warns on limits |
| An API-key path is not needed for policy reasons | **Medium** | It follows from A1, but it may still be wanted for usage limits and billing (A7) |

## Implications for Maata

**1. Settle the scope before the background bake-off.** Record the maintainer's choice as an ADR.
- **(a) Personal tool only.** Drop the signed-installer plan (Amendment 01 §3.7). The YouTube text still applies, but the "enable others" and distribution exposure go away.
- **(b) Distributed as designed.** This needs counsel first. Questions for counsel:
  1. Is a muted IFrame embed plus separately played synced audio "applying alternate audio tracks" (III.I.7) or a derivative translation (III.I.16)?
  2. What is the exposure for distributing an app built on yt-dlp (III.I.14, III.E.1, the YouTube ToS, and the §1201 theory in Y10)?
  3. What is the copyright status of a private, on-device translation of a creator's work (Y11)?
  4. Is a request through the III.G approval form, or a direct agreement with YouTube, realistic?
- **(c) Redesign for distribution (INFERRED; not verified here).** One route is a local-file mode, where the user supplies media they have rights to. That removes the YouTube terms entirely. Another is a creator-side tool: creators dub their own videos and publish the tracks through YouTube's own channels. Research YouTube's creator audio-track features before relying on this.

**2. How the scope changes the background ranking** (same evidence, different constraint):
- **Distributed:** B (synthetic room tone plus matching) is the only option. It plays no YouTube audio and separates nothing, so it adds no exposure beyond the core design. C (separated bed) and D (unmuting the player in gaps) both break the maintainer's rule, and C is also the textbook III.I.7 case.
- **Personal only:** B first, D second, C last.
  - D uses only documented player calls (`unMute` and `setVolume`), plays YouTube's own stream through YouTube's own player, and needs no separation model. It still breaks "original audio never played" and risks leaking English speech.
  - C adds the most exposure and needs a separation model. Its only evidence of benefit is for non-native listeners (C above).

  **Changed ranking vs reports 01/02/03/08/09:** do not build a separated bed for either scope unless the maintainer relaxes his rule and counsel clears III.I.7.

**3. Compliance fixes that hold under any scope** (low effort):
- Set the embed's Referer to `https://io.github.shankarpandala.maata` (Y3).
- Keep the player UI free of overlays, and keep its volume and caption controls usable.
- Keep the player at 200×200 px or larger.
- Publish a privacy policy that says transcripts go to Anthropic through the user's own CLI (Y1 §7).
- Keep fetched audio no longer than a session needs (Y12).

**4. For the Claude CLI path, if distributed:**
- The maintainer accepts the Commercial Terms.
- Maata detects the user's installed `claude` and never bundles or patches it.
- Maata tells users to sign in with the CLI's own `/login`.
- Maata never reads the keychain and never accepts a pasted `setup-token` token (the A1 token clause).
- Maata does not pass `--bare`, which would shut out subscription users (A1: do not "restrict any authentication method").
- Maata may pass a user-supplied `ANTHROPIC_API_KEY` as the user's own choice.
- Product text may say "runs Claude Code" in plain text, but must not use the Claude or Anthropic names or logos in Maata's branding (A1).

**5. Ask Anthropic sales in writing** (the contact link is on A1). The page now answers the general question, so ask only about what is specific to Maata:
1. Does a free, open-source desktop app that spawns the user's own installed CLI count as "running Claude Code in your products", so that the maintainer must accept the Commercial Terms?
2. Is batch translation of long transcripts through `claude -p` on Pro or Max "ordinary, individual usage"?
3. Will the paused billing change (A7) apply to such apps?

**6. Update the documents.**
- Amend CLAUDE.md and the spec's "no cloud AI" and "no network except yt-dlp" rules to match today's no-local-LLM decision.
- Note that a release has no offline translation fallback. Under the current decision, the local shortlist in `sota_ranking.md` §1 is reference only.

## Sources (all fetched 2026-09-24 unless noted)

- YouTube API Services Terms of Service (last updated 2026-09-14): https://developers.google.com/youtube/terms/api-services-terms-of-service
- YouTube API Services Developer Policies (last updated 2026-09-14): https://developers.google.com/youtube/terms/developer-policies
- Required Minimum Functionality (last updated 2026-09-14): https://developers.google.com/youtube/terms/required-minimum-functionality
- Complying with YouTube Developer Policies (last updated 2026-09-14): https://developers.google.com/youtube/terms/developer-policies-guide
- Terms revision history (last updated 2026-09-11): https://developers.google.com/youtube/terms/revision-history
- IFrame Player API reference (last updated 2026-09-15): https://developers.google.com/youtube/iframe_api_reference
- Player parameters (last updated 2026-09-16): https://developers.google.com/youtube/player_parameters
- Quota and Compliance Audits (last updated 2026-09-14): https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits
- YouTube Terms of Service (US, effective 2023-12-15): https://www.youtube.com/static?template=terms
- yt-dlp wiki, EJS (undated): https://github.com/yt-dlp/yt-dlp/wiki/EJS
- RIAA DMCA notice to GitHub (2020-10-23): https://github.com/github/dmca/blob/master/2020/10/2020-10-23-RIAA.md
- GitHub blog, "Standing up for developers: youtube-dl is back" (2020-11-16): https://github.blog/news-insights/policy-news-and-insights/standing-up-for-developers-youtube-dl-is-back/
- 17 U.S.C. §101: https://www.law.cornell.edu/uscode/text/17/101
- Claude Code legal and compliance (undated; product section first captured 2026-08-30): https://code.claude.com/docs/en/legal-and-compliance
- Wayback captures of that page: https://web.archive.org/web/20260816100738/https://code.claude.com/docs/en/legal-and-compliance and https://web.archive.org/web/20260830094710/https://code.claude.com/docs/en/legal-and-compliance
- Agent SDK overview: https://code.claude.com/docs/en/agent-sdk/overview
- Claude Code authentication: https://code.claude.com/docs/en/authentication
- Anthropic Consumer Terms (effective 2025-10-08): https://www.anthropic.com/legal/consumer-terms
- Anthropic Commercial Terms (effective 2025-06-17): https://www.anthropic.com/legal/commercial-terms
- Support article 15036540 (updated 2026-06-16): https://support.claude.com/en/articles/15036540
- Federico, Enyedi and Barra-Chicote, "From Speech-to-Speech Translation to Automatic Dubbing" (arXiv 2001.06785, 2020)
- Repo files (read 2026-09-24): `app/src/lib/player.ts`, `engine/src/maata_engine/resolve.py`, `engine/src/maata_engine/claude_cli.py`, `docs/DECISIONS.md` (ADR-006, ADR-010, ADR-011), `docs/SPEC-AMENDMENT-01.md`

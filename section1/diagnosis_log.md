# Chatbot Incident Diagnosis Log
**System:** GPT-4o Customer Support Chatbot  
**Environment:** Production (post-launch)  
**Reported Issues:** 3 distinct failure modes  
**No code or prompt changes were made after launch.**

---

## Problem 1 — Hallucinated Pricing

### Investigation Trail

**What I investigated first:**  
Since the bot "confidently" gave wrong answers about pricing (not vague or hedged answers), the first hypothesis was a **retrieval failure** — the bot was generating prices from memory rather than pulling from a grounded source. I checked whether a RAG (Retrieval-Augmented Generation) pipeline was in use, and whether the retrieved chunks were current.

**What I ruled out:**

| Hypothesis | Status | Reasoning |
|---|---|---|
| **Model temperature too high** | ❌ Ruled out | High temperature produces incoherent or varied answers, not confidently wrong ones. If temperature caused this, wrong answers would be inconsistent across identical queries. Pricing errors were reportedly consistent. |
| **Knowledge cutoff issue** | ❌ Ruled out first | GPT-4o's training cutoff is early 2024. Pricing data changes continuously — the model was never trained to know *your* product prices in the first place. Cutoff is irrelevant here; this is not a "stale knowledge" problem, it's a "no authoritative knowledge" problem. |
| **Prompt issue (prompt phrasing)** | ⚠️ Partially relevant | The prompt likely didn't explicitly instruct the model to refuse or defer when pricing data is unavailable. This is a contributing factor, not the root cause. |
| **Retrieval issue** | ✅ Root cause | The model is confabulating (hallucinating) pricing because it has no grounded source to retrieve from. |

### Root Cause

The model has no access to authoritative, real-time pricing data. GPT-4o — like all LLMs — will **fill gaps with plausible-sounding fabrications** when asked questions that fall outside its grounded context. Without a retrieval step that injects current pricing into the context window, the model invents answers that match the expected format of a price (e.g., "$49/month") with complete confidence.

This is a **retrieval issue**, compounded by a **prompt issue** (no explicit instruction to decline when data is missing).

### How to Distinguish Between Causes

- **Test for temperature:** Ask the same pricing question 10 times. If answers vary wildly → temperature. If they're consistently wrong in the same way → not temperature.  
- **Test for cutoff:** Ask about facts that *would* be in training data (e.g., "Who founded OpenAI?"). If those are accurate but pricing is wrong, cutoff isn't the issue — the model simply never had product-specific pricing.  
- **Test for retrieval:** Check logs: was a retrieval call made? Were pricing documents returned? If no retrieval occurred or returned empty chunks, root cause is confirmed.

### Fix

**Immediate (Prompt Hardening):**
Add an explicit instruction to the system prompt:

```
You are a customer support assistant for [Company].
You ONLY answer pricing questions using information explicitly provided to you in the [CONTEXT] section below.
If pricing information is not present in the [CONTEXT], respond with:
"I don't have the current pricing details on hand. Please visit [pricing-page-url] or contact our sales team."
Never infer, estimate, or guess prices.
```

**Structural (Retrieval Pipeline):**  
Implement a RAG pipeline that fetches current pricing from a structured source (database, CMS, or internal API) and injects it into the context before each response. The prompt should reference this:

```
[CONTEXT]
{retrieved_pricing_data}

Using ONLY the above context, answer the user's pricing question.
```

Set `temperature=0` or very low (0.1) for pricing-related queries to minimize creative deviation.

---

## Problem 2 — Language Switching (Responding in English to Hindi/Arabic Users)

### Investigation Trail

**What I investigated first:**  
The system prompt was the first thing I examined. In a `system prompt + user message` architecture, the system prompt is written in English and carries implicit cultural and linguistic framing. This is a well-documented LLM failure mode where the model prioritizes the language of its strongest instruction signal — which is typically the system prompt.

**What I ruled out:**

| Hypothesis | Status | Reasoning |
|---|---|---|
| **Model doesn't understand Hindi/Arabic** | ❌ Ruled out | GPT-4o is multilingual and performs well in Hindi and Arabic. Language comprehension is not the issue. |
| **User input detection bug** | ❌ Ruled out | No code change was made; this isn't a sudden regression from a code bug. |
| **Model temperature** | ❌ Irrelevant | Language selection is not a stochastic output; temperature doesn't govern which language the model responds in. |

### Root Cause: The Mechanism

In a `system prompt + user message` architecture, the model must infer intent from **two competing signals**:

1. The **system prompt** — written in English, defining persona, rules, tone, and behavior.
2. The **user message** — written in Hindi or Arabic.

When the system prompt does not explicitly instruct the model to match the user's language, the model applies a **default heuristic**: respond in the language of the longest or most dominant instruction block. Since the system prompt is typically longer and more structured than a single user message, English wins — especially when the user's input is short (e.g., a one-line question).

This is not a bug in the model; it is an **under-specified instruction**. The model is being asked to infer a behavior that was never explicitly defined.

This occurs "occasionally" rather than always because:
- Longer user messages in Hindi/Arabic provide stronger signal → model sometimes correctly mirrors.
- Short or ambiguous user messages provide weak signal → model defaults to system prompt language.

### Specific Prompt Fix

Add this instruction to the system prompt (place it prominently — ideally in the first or second sentence):

**Before (implicit, broken):**
```
You are a helpful customer support assistant for [Company]. 
Answer customer questions about our products and services.
```

**After (explicit, fixed):**
```
You are a helpful customer support assistant for [Company].
IMPORTANT: Always respond in the exact same language the user writes in.
If the user writes in Hindi, respond entirely in Hindi.
If the user writes in Arabic, respond entirely in Arabic.
If the user writes in English, respond in English.
Never switch languages mid-conversation unless the user explicitly changes language first.
Answer customer questions about our products and services.
```

**Why this works and is language-agnostic:**  
The instruction `"respond in the exact same language the user writes in"` requires no enumeration of supported languages. It generalizes to any language GPT-4o supports without needing per-language rules. The enumerated examples (Hindi, Arabic, English) serve as few-shot demonstrations that reinforce the rule without limiting it.

**Testability:**
- Send 20 test messages: 5 in Hindi, 5 in Arabic, 5 in English, 5 in mixed (e.g., code-switching).
- Pass criterion: ≥19/20 responses match the user's input language.
- Run regression after any system prompt update.

---

## Problem 3 — Latency Degradation (1.2s → 8–12s over Two Weeks)

### Investigation Trail

**What I investigated first:**  
Since no code changes were made, this is almost certainly an **infrastructure or operational scaling issue**, not a model or prompt regression. The pattern — gradual degradation over two weeks correlated with user base growth — immediately points to resource contention or rate-limit pressure.

### Three Distinct Root Causes (Ordered by Likelihood)

#### Cause 1 — Conversation History / Context Window Bloat (Most Likely — Investigate First)

**What it is:** If the chatbot stores conversation history and includes it in every API call, the token count of each request grows with every turn. A conversation that started at 500 tokens may now routinely pass 8,000–15,000 tokens after weeks of production use and longer sessions.

**Why it causes this pattern:** GPT-4o's time-to-first-token scales with input length. Longer prompts take longer to process. No code change is needed for this to worsen — it happens automatically as conversations grow.

**Why investigate first:** This is the most common production LLM latency pattern, is easily verified from API logs (check average input token count week-over-week), and has a cheap fix.

**How to verify:** Pull API call logs and plot `avg_input_tokens` vs. time. If it has grown proportionally with latency, this is the cause.

**Fix:** Implement a sliding window or summarization strategy — keep only the last N turns in context, or periodically summarize history into a compact paragraph.

---

#### Cause 2 — OpenAI API Rate Limiting / Tier Throttling

**What it is:** As user volume increases, the application may be hitting OpenAI's rate limits (requests per minute, tokens per minute). When rate limits are reached, the API queues or rejects requests, causing exponential backoff retries and visible latency spikes.

**Why it causes this pattern:** Rate limits don't apply in low-traffic testing. As concurrent users grow, the threshold is crossed and latency degrades non-linearly.

**How to verify:** Check for `429 Too Many Requests` or `RateLimitError` in application logs. Check OpenAI usage dashboard for RPM/TPM headroom.

**Fix:** Upgrade to a higher OpenAI usage tier, implement request queuing with proper backoff, and add a caching layer for repeated identical queries (e.g., FAQ responses).

---

#### Cause 3 — Downstream Infrastructure Saturation (Database / Vector Store / Application Server)

**What it is:** If the chatbot uses a vector database for RAG, or a relational DB for session/conversation storage, those systems can become bottlenecks under higher load. Connection pool exhaustion, slow query plans on growing tables, or underpowered application servers (CPU/RAM) can all add latency *before* the OpenAI API is even called.

**Why it causes this pattern:** Under-provisioned infrastructure performs well at low traffic but degrades gradually as concurrent users multiply — a textbook scaling failure.

**How to verify:** Use APM tools (Datadog, New Relic, or even application-level timing logs) to identify where time is spent per request: DB query time, retrieval time, OpenAI call time, post-processing time. If the OpenAI call itself is fast but total response time is slow, the bottleneck is elsewhere.

**Fix:** Add database indexes on high-frequency query fields, scale application servers horizontally, upgrade vector DB tier, or introduce a Redis caching layer for session data.

---

### Investigation Priority Order

1. **Context window bloat** — Check API logs for token count trend. Free to diagnose, high-probability root cause.  
2. **Rate limiting** — Check error logs for 429s. Straightforward to verify.  
3. **Infrastructure saturation** — Requires APM tooling; investigate if #1 and #2 are clear.

---

## Post-Mortem Summary (For Non-Technical Stakeholders)

Over the past two weeks, our customer support chatbot has experienced three distinct issues that affected user experience. Here's what happened and what we're doing to fix it.

**Incorrect Pricing:** The chatbot was giving users wrong prices for our products. This happened because the AI model doesn't actually "know" our prices — it was making educated guesses, which is a known limitation of large language models. The fix is to connect the chatbot directly to our pricing database so it always reads real prices instead of guessing. In the meantime, we've updated its instructions to say "I don't have that information — please check our pricing page" rather than risk a wrong answer.

**Language Switching:** Some users writing in Hindi or Arabic received replies in English. The chatbot's core instructions were written in English, and without being explicitly told to mirror the user's language, it defaulted to English in ambiguous situations. We've added a clear instruction: "always respond in the language the user writes in," which resolves this reliably.

**Slow Responses:** Response times grew from about 1 second to 8–12 seconds as more users joined. The most likely cause is that the chatbot was including the full history of each conversation in every request — as conversations got longer, each request got heavier. We're implementing a fix to trim older conversation history and cache common responses, which should restore fast response times.

All three issues stem from the system being under-specified for real-world conditions at scale — a common gap between controlled testing and live production environments.

# AI Prompts & Reasoning

This document logs the prompts I used with Claude (web) and Claude Code
while building this assignment, plus the reasoning behind each.

## Approach

I treated the AI as a pair-programmer, not an autocomplete. My workflow:
1. Plan the architecture myself before writing any prompts.
2. Ask the AI to scaffold one phase at a time, verify it works, then move on.
3. When output was wrong, refine the prompt rather than restart.
4. Test every requirement manually — the AI's "it works" is not a substitute
   for running the code.

The phases below mirror the implementation plan I built before coding.

---

## Phase 1 — Project scaffolding

**Prompt :** asked for the directory structure, requirements.txt
with pinned versions, .gitignore, and empty Python stubs.

**Why I asked it this way:** I wanted the skeleton in one shot so I could
move directly to logic. Pinning versions up front prevented dependency drift
later when adding langchain-mcp-adapters.

**What I had to fix:** the AI initially suggested unpinned packages. I
asked it to add explicit version constraints.

---

## Phase 2 — State schema

**Prompt :** asked for a TypedDict-based ConversationState with
specific fields (messages with add_messages reducer, clarity_status,
company_name, confidence_score, research_attempts, etc.) and a helper
initial_state() function.

**Why I asked it this way:** the state is the spine of any LangGraph app.
Defining it before any agent forced me to think about which fields each
agent reads and writes, which made the routing logic obvious later.

**What I decided myself:** field names and which fields persist across
turns (company_name) versus which reset per turn.

---

## Phase 3 — Tavily search tool

**Prompt :** asked for an async search_company() function with MCP
as the primary path (via langchain-mcp-adapters) and a SDK fallback
(tavily-python) gated on MCP connection success.

**Why I asked it this way:** the assignment says "Tavily MCP would be
preferred" — preferred, not required. Building MCP-first with a real
fallback covers both the spec and the demo case where MCP isn't running.

**What I had to fix:** the first version connected to MCP via HTTP/SSE on
a port. After confirming Tavily's official MCP server uses stdio with
subprocess spawning, I had it rewrite the connection to use stdio. The
SDK fallback was kept as the safety net.

---

## Phase 4 — Clarity Agent

**Prompt :** asked for an agent that uses structured output
(Pydantic ClarityDecision) to decide if a query is "clear" or
"needs_clarification", explicitly handling the pronoun-with-prior-context
case.

**Why I asked it this way:** the multi-turn requirement hinges on this
agent. If "what about their CEO?" after a Tesla turn triggers an
interrupt, the demo breaks. I gave the AI three concrete examples in the
prompt to anchor the desired behavior.

**What I had to fix:** the first version was too permissive — it would
guess a company name when none existed. I tightened the system prompt
with an explicit "do NOT guess, do NOT pick a popular company" rule, and
the interrupt then fired reliably.

---

## Phase 5 — Research Agent

**Prompt :** asked for an async agent that templates a search
query from company_name + user_question, calls Tavily, then uses a
second LLM call with structured output to summarize and self-assess
confidence on an anchored 0–10 scale.

**Why I asked it this way:** confidence scoring without an anchored
scale is meaningless. The 0–3 / 4–5 / 6–7 / 8–10 rubric in the prompt
gives the LLM a consistent reference point so the validator-routing
threshold of 6 is actually predictive of quality.

---

## Phase 6 — Validator Agent

**Prompt :** asked for an agent that judges whether all
accumulated research_findings (not just the latest) sufficiently answer
the user's question, returning sufficient/insufficient plus a reason.

**Why I asked it this way:** passing all findings instead of just the
latest matters for the retry loop — if retry 1 found half the answer and
retry 2 found the other half, the validator should see both.

---

## Phase 7 — Synthesis Agent

**Prompt:** asked for a single LLM call producing a Markdown
summary with inline citations, conversation continuity, and a low-
confidence disclaimer.

**Why I asked it this way:** the synthesis is what the user actually
sees. Citations and a graceful "I had limited information" disclaimer
were both explicit assignment requirements, so I baked them into the
prompt rather than hoping the model would infer them.

---

## Phase 8 — Graph wiring

**Prompt :** asked for a StateGraph with nodes, conditional
edges, MemorySaver checkpointer, and an interrupt-based clarification
node that loops back to clarity after user input.

**Why I asked it this way:** the routing rules were already explicit in
my implementation plan — clarity branches on clarity_status, research
branches on confidence_score, validator branches on validation_result
+ research_attempts. The prompt mirrored the rules I'd written down,
which left no room for the AI to invent its own routing.

**What I had to fix:** the first interrupt implementation didn't loop
back through clarity after resuming. I asked for a small clarification
node that returns the human response as a new message and sets
clarity_status to "pending" so the graph re-evaluates.

---

## Phase 9 — CLI driver

**Prompt :** asked for an async REPL with a stable thread_id,
interrupt detection after .ainvoke(), resume via Command(resume=...),
and visible agent-transition logging.

**Why I asked it this way:** the demo video needs to show routing
visibly. Without the [clarity → research → synthesis] transitions
printed, a grader watching the video can't tell what's happening.

---

## Phase 10 — Testing

I tested all five required scenarios manually:
clear query, ambiguous query (interrupt), multi-turn follow-up,
validator loop on low confidence, and max-attempts graceful
degradation. The biggest issue I caught was that I initially ran
scenarios back-to-back without restarting the app, which let prior
context bleed into ambiguity tests. Restarting between scenarios
fixed it.

---

## Phase 11 — Polish

I asked the AI for a README pass and a final code review (docstrings,
removing debug prints, no hardcoded keys). I wrote the design-notes
section myself because the rationales were my decisions.

---

## What I'd do differently next time

- Stand up the Tavily MCP server before writing the tool, not after.
  I lost time debugging connection issues that were really a transport
  mismatch.
- Write the Clarity Agent test cases (clear / ambiguous / follow-up)
  before writing the agent, so I could iterate the prompt against
  concrete failures instead of intuition.
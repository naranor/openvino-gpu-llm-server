# lean-ctx — Context Engineering Layer

MANDATORY: Use lean-ctx MCP tools for ALL reads, searches, and shell commands.

| FORBIDDEN | USE INSTEAD |
|-----------|-------------|
| Read / cat / head / tail | `ctx_read(path)` — cached, 8 compression modes, re-reads ~13 tokens |
| Shell / bash / terminal | `ctx_shell(command)` — pattern compression for git/npm/cargo output |
| Grep / rg / search | `ctx_search(pattern, path)` — compact, token-efficient results |
| ls / find / tree | `ctx_tree(path, depth)` — compact directory maps |

Keep using Write, StrReplace, Delete, Glob normally (no lean-ctx replacement).

REMINDER: NEVER use native Read, Shell, Grep, or ls. ALWAYS use ctx_read, ctx_shell, ctx_search, ctx_tree.# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Mandatory Project Rules

- **NO UNAPPROVED PUSHES:** NEVER push changes to remote repositories (GitHub, etc.) without explicit approval from the user. Local commits are allowed.
- **VENV FOR PIP EXPERIMENTS:** ALWAYS create and use a virtual environment (`venv`) when experimenting with `pip install --force-reinstall` or any potentially breaking dependency changes to avoid breaking the system environment.
- **VERIFY SIGNATURES:** ALWAYS verify API signatures, method names, and class structures (using `dir()`, `help()`, or test scripts) before implementing code using external libraries. Do not guess.
- **LOGGING PRESERVATION:** NEVER remove or comment out existing logging/debug statements. You may only add new logging. All logs must include high-resolution timestamps.
- **GPU-ONLY MANDATE:** Maintain the requirement for GPU acceleration on Intel UHD 620 using OpenVINO. Use 32k context stability via small batch processing (128-512 tokens) for all models.

---

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

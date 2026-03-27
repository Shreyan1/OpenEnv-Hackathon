# Agent Instructions

## Project Context

This is a **reinforcement learning memory management system** for LLM agents. The codebase trains a policy to decide what to store, retrieve, update, or ignore during multi-turn conversations, under a fixed memory token budget.

Key directories:
- `src/memory_management_agent/` — all source code
- `tests/test_core.py` — full test suite (run with `python -m unittest tests/test_core.py -v`)

See `CLAUDE.md` for full architecture, schemas, and usage examples.

## Working on This Project

### Before starting work

```bash
bd ready              # Find available issues
bd show <id>          # Read issue details
bd update <id> --claim  # Claim the issue atomically
```

### Workflow

1. Run tests before making changes: `python -m unittest tests/test_core.py -v`
2. Make changes
3. Run tests again — all must pass
4. Close issue and push (see Session Completion below)

### Core entry points

```python
# Run a single episode
from src.memory_management_agent import MemoryManagementEnv, RuleBasedMemoryAgent, run_episode
result = run_episode(RuleBasedMemoryAgent(), MemoryManagementEnv(), seed=42)

# Evaluate an agent across many episodes
from src.memory_management_agent import evaluate_split
report = evaluate_split(agent, env, visible_seeds=range(10), hidden_seeds=range(5000, 5005))

# Collect rollouts for training
from src.memory_management_agent import collect_rollouts, export_rollouts_jsonl
rollouts = collect_rollouts(agent, env, seeds=tuple(range(20)))
export_rollouts_jsonl(rollouts, "rollouts.jsonl")
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

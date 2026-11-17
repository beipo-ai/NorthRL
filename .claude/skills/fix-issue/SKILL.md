---
name: fix-issue
description: Fix a GitHub issue using git worktree for isolation. Fetches issue, creates worktree, implements fix, then creates PR.
---

# Fix Issue Workflow (Worktree-Based)

## Workflow

### Step 1: Verify Prerequisites

```bash
gh auth status
REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
```

### Step 2: Fetch Issue

```bash
ISSUE_NUM=<number>
gh issue view "$ISSUE_NUM" --repo "$REPO" --json number,title,body,state,labels
```

If closed, confirm with user before proceeding.

### Step 3: Create Worktree

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
git fetch origin

BRANCH="agent/issue-${ISSUE_NUM}"
WORKTREE_DIR="${REPO_ROOT}/../worktrees/NorthRL-issue-${ISSUE_NUM}"

git worktree add "$WORKTREE_DIR" -b "$BRANCH" "origin/main"
cd "$WORKTREE_DIR"
```

### Step 4: Read Documentation First

Before coding, read relevant docs per `.ai-instructions/developing/documentation.md`:
- `docs/en/developer-guide/` for architecture understanding
- Component-specific docs for the area being changed

### Step 5: Plan the Fix

Enter plan mode. Consider:
- Root cause analysis (for bugs)
- Files that need changes
- Impact on other components
- Documentation updates needed

### Step 6: Implement

Work in the worktree directory. Follow:
- `.ai-instructions/developing/` conventions
- Run `python -m black` and `python -m isort` on changed files
- Ensure license headers present

### Step 7: Test

Run project tests (use `testing` skill):
```bash
python -m pytest tests/ -v
python -m pylint northrl/ --rcfile=.pylintrc -rn -sn
```

### Step 8: Commit and PR

Use `git-commit` skill, then `github-pr` skill.
Include `Fixes #<issue-number>` in the PR body and return the PR URL to the user.
This workflow uses the current repository's issues and does not require a shared
knowledge repository, organization Project, or predefined status labels.

### Step 9: Handle Blockers

Explain the blocker, completed work, and information needed to the user. Preserve
the worktree so the work can resume. Update issue labels or comments only when
the user requests it or the repository's documented contribution process calls
for it.

### Step 10: Cleanup (after PR merged)

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
git worktree remove "$WORKTREE_DIR"
git branch -D "$BRANCH"
```

---
name: github-pr
description: Create a GitHub PR from a worktree branch. Use after committing changes.
---

# GitHub PR Workflow (Worktree-Aware)

## Prerequisites

Changes committed via `git-commit` skill.

## Workflow

### Step 1: Verify Current Location

```bash
git worktree list
pwd
BRANCH_NAME=$(git branch --show-current)
REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
echo "Current branch: $BRANCH_NAME"
```

### Step 2: Check for Existing PR

```bash
gh pr list --head "$BRANCH_NAME" --state open --repo "$REPO"
```

If PR exists, show it and exit.

### Step 3: Push Branch

```bash
git push --set-upstream origin "$BRANCH_NAME"
```

### Step 4: Create PR

```bash
gh pr create \
  --repo "$REPO" \
  --base main \
  --head "$BRANCH_NAME" \
  --title "type(scope): description" \
  --body "## Summary
- Key change 1
- Key change 2

## Testing
- [ ] Tests pass (`pytest`)
- [ ] Linting clean (`pylint`, `mypy`)
- [ ] Formatting applied (`black`, `isort`)

## Related Issues
Fixes #ISSUE_NUMBER"
```

Auto-extract title/body from commit messages. No AI branding.

### Step 5: Report the PR

Return the PR URL to the user. Link the relevant issue in the PR description;
no organization Project or predefined status labels are required. Update issue
labels or add a separate comment only when the user requests it or the
repository's documented contribution process calls for it.

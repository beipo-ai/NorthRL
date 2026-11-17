# Documentation Workflow

## Overview

NorthRL follows a **documentation-first development** approach. This ensures that code and documentation remain aligned, making the codebase easier to understand and maintain.

## Core Principles

### 1. Read Documentation Before Coding

**Always read relevant documentation before making any code changes.**

Key documentation files:
- **[`docs/en/developer-guide/README.md`](../../docs/en/developer-guide/README.md)** - Developer guide overview

- **Core Architecture Documentation:**
  - [`docs/en/developer-guide/02-core-architecture/overview.md`](../../docs/en/developer-guide/02-core-architecture/overview.md) - System architecture overview
  - [`docs/en/developer-guide/02-core-architecture/controller.md`](../../docs/en/developer-guide/02-core-architecture/controller.md) - Main controller implementation
  - [`docs/en/developer-guide/02-core-architecture/data-types.md`](../../docs/en/developer-guide/02-core-architecture/data-types.md) - Core data structures
  - [`docs/en/developer-guide/02-core-architecture/activity-tracking.md`](../../docs/en/developer-guide/02-core-architecture/activity-tracking.md) - Activity tracking system

- **Component Documentation:**
  - [`docs/en/developer-guide/05-rollout-workers/`](../../docs/en/developer-guide/05-rollout-workers/) - Rollout worker implementations
  - [`docs/en/developer-guide/06-trainers/`](../../docs/en/developer-guide/06-trainers/) - Trainer implementations
  - [`docs/en/developer-guide/06-trainers/remote-api-trainers.md`](../../docs/en/developer-guide/06-trainers/remote-api-trainers.md) - Remote training service integration

- **Configuration and Setup:**
  - [`docs/en/developer-guide/01-getting-started/`](../../docs/en/developer-guide/01-getting-started/) - Getting started guides
  - [`docs/en/developer-guide/11-configuration-reference/`](../../docs/en/developer-guide/11-configuration-reference/) - Complete configuration reference

- **Other docs in [`docs/en/developer-guide/`](../../docs/en/developer-guide/)** - Check for additional relevant documentation

### 2. Review Documentation After Each Edit

**After making any code changes, review the documentation to ensure alignment.**

Ask yourself:
- Does the documentation still accurately describe how the code works?
- Are there any examples in the docs that might be affected?
- Did I change any APIs or behavior that's documented?
- Are field descriptors and reflection behavior still accurate?

### 3. Update Documentation When Needed

**If your code changes affect documented behavior, update the documentation.**

Update docs when you:
- Modify core architecture or add new components
- Change API signatures or usage patterns
- Add, remove, or modify rollout workers, trainers, or services
- Change configuration schemas or parameters
- Modify data structures or communication protocols
- Update local startup procedures or remote service integration

## Documentation Structure

```
docs/
├── README.md              # Language selector and maintenance notes
├── en/                    # English documentation
│   ├── index.md           # Complete documentation index
│   ├── README.md          # Project overview
│   ├── developer-guide/   # Chapters 01–12
│   ├── recipe/            # Recipe guides
│   ├── tests/             # Test suite guide
│   └── user-guide.md      # User guide
├── zh/                    # Simplified Chinese; matching paths
└── imgs/                  # Shared images
```

Update English and Chinese pages together, using identical relative paths under
`docs/en/` and `docs/zh/`. Keep code examples, configuration keys, and API names
unchanged in translations. Each page links to its counterpart and language index;
retain explicit section anchors when translating headings. Update both indexes
when adding pages. The root README is the repository landing page; its public
project overview is also available in `docs/en/README.md` and `docs/zh/README.md`.
Keep complete local startup instructions and service settings in the existing
`developer-guide/01-getting-started/quick-start.md` and `configuration-setup.md`
pages. Other guides should link to them rather than duplicate those tutorials.
NorthRL runs through `python -m northrl.main` with an explicit recipe. Document
environment preparation in the caller's shell and model access through the
remote services. Keep recipe-specific convenience scripts with their recipes.

## Documentation Style Guide

When updating documentation:

1. **Use clear, descriptive headings** - Make it easy to scan
2. **Provide code examples** - Show Python usage and configuration examples
3. **Explain the "why"** - Don't just describe what code does, explain design decisions
4. **Use diagrams** - For complex architectures and data flows
5. **Keep examples working** - Test that examples actually run
6. **Link between docs** - Reference related documentation files
7. **Maintain consistency** - Follow the existing documentation style
8. **Include configuration examples** - Show relevant YAML/Python config snippets
9. **Keep the docs short and easy to read** - Each doc should usually be less than 300 lines


## Common Documentation Tasks

### Adding a New Component (Worker, Trainer, Service)

1. Read relevant component documentation in `docs/en/developer-guide/`
2. Implement the new component in Python
3. Update the component overview section
4. Add configuration examples
5. Provide usage examples and integration patterns
6. Update architecture diagrams if needed

### Modifying Existing Component

1. Read current documentation for that component
2. Make code changes
3. Update component description and examples
4. Verify all examples still work
5. Update configuration reference if parameters changed
6. Update any affected sections in other docs

### Changing Configuration Schema

1. Update configuration validation code
2. Update `docs/en/developer-guide/11-configuration-reference/`
3. Update affected recipe examples in `docs/en/developer-guide/09-recipes/`
4. Test that configuration examples are valid

## Documentation Quality Checklist

Before finalizing changes, verify:

- [ ] Code matches documented behavior
- [ ] All code examples in docs are valid and tested
- [ ] Python implementation matches documented API
- [ ] Configuration examples are valid and tested
- [ ] English and Chinese page paths, sections, and code examples match
- [ ] Language switches and both documentation indexes are up to date
- [ ] Links between documentation files are correct
- [ ] No broken references to moved/renamed files
- [ ] Examples use current API (no deprecated patterns)
- [ ] Documentation follows existing style and formatting
- [ ] Architecture diagrams are up-to-date if components changed

## Remember

**Good documentation is as important as good code.**

Documentation is the primary way developers understand the system. Keeping it accurate and up-to-date prevents bugs, reduces confusion, and makes the codebase more maintainable.

When in doubt, update the docs!

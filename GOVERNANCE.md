# Governance

This document describes the governance model for the Ununennium project.

---

## Project Structure

### Roles

| Role | Responsibilities |
|------|-----------------|
| **Lead Architect** | Technical vision, architecture decisions, release approval |
| **Maintainers** | Code review, merge authority, issue triage |
| **Contributors** | Feature development, bug fixes, documentation |
| **Community** | Feature requests, bug reports, discussions |

### Current Leadership

- **Lead Architect**: Olaf Yunus Laitinen Imanov
- **Maintainers**: Core team members with merge access

---

## Decision Making

### Technical Decisions

Technical decisions follow this process:

1. **Proposal**: Open a GitHub Discussion or Issue
2. **Discussion**: Community input period (minimum 7 days for significant changes)
3. **Review**: Maintainer evaluation
4. **Decision**: Lead Architect approval for major architectural changes
5. **Implementation**: Standard PR process

### Decision Categories

| Category | Decision Authority | Examples |
|----------|-------------------|----------|
| **Minor** | Any maintainer | Bug fixes, documentation, minor features |
| **Standard** | Two maintainers | New features, API additions |
| **Major** | Lead Architect + maintainers | Breaking changes, architecture changes |
| **Critical** | Lead Architect | License changes, governance changes |

---

## Release Policy

### Versioning

Ununennium follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes to public API
- **MINOR** (0.X.0): New features, backward-compatible
- **PATCH** (0.0.X): Bug fixes, backward-compatible

### Release Process

1. Feature freeze announcement
2. Release candidate (RC) testing period
3. Final testing and documentation review
4. Version bump and CHANGELOG update
5. Git tag and GitHub release
6. PyPI publication

### Release Schedule

| Release Type | Frequency | Description |
|--------------|-----------|-------------|
| Patch | As needed | Critical bug fixes |
| Minor | Quarterly | New features, enhancements |
| Major | Annually | Breaking changes (if necessary) |

---

## Deprecation Policy

### Deprecation Timeline

| Phase | Duration | Action |
|-------|----------|--------|
| Announcement | Immediate | Deprecation warning in documentation |
| Warning | 1 minor release | Runtime deprecation warnings |
| Removal | Next major release | Feature removed |

### Deprecation Requirements

When deprecating functionality:

1. Document the deprecation in CHANGELOG.md
2. Add runtime deprecation warnings
3. Provide migration path in documentation
4. Maintain deprecated functionality for at least one minor release

---

## RFC Process

For significant changes, we use a lightweight RFC (Request for Comments) process:

### When to Use RFC

- New major features
- Breaking API changes
- Significant architectural changes
- Changes affecting multiple modules

### RFC Template

```markdown
# RFC: [Title]

## Summary
Brief description of the proposal.

## Motivation
Why this change is needed.

## Detailed Design
Technical details of the implementation.

## Drawbacks
Potential downsides or risks.

## Alternatives
Other approaches considered.

## Unresolved Questions
Open issues for discussion.
```

---

## Communication Channels

| Channel | Purpose |
|---------|---------|
| GitHub Issues | Bug reports, feature requests |
| GitHub Discussions | Questions, RFCs, general discussion |
| Pull Requests | Code review, implementation discussion |

---

## Amendments

This governance document may be amended through the Major decision process. Amendments require Lead Architect approval and a 14-day community comment period.

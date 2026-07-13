# Development and Release Workflow

## Purpose

Use `develop` as the integration branch, `main` as the published-release
branch, and short-lived branches for normal feature, packaging, and hotfix
work.

## Branch roles

### `main`

- Represents released or immediately releasable code.
- Every PyPI release is tagged on an exact `main` commit, such as `v0.2.0`.
- Does not receive unfinished feature work.
- Is not deleted.

### `develop`

- Integrates reviewed work intended for a future release.
- Is the base branch for normal feature branches.
- Is merged into `main` only after the release gate passes.
- Is retained after every release.

### Feature branches

Create one short-lived branch per substantial feature:

```text
feat/browser-pooling
feat/timeline-selection
```

Feature branches start from current `develop` and merge back into `develop`
through a pull request.

### Packaging or release branches

Use a focused branch from synchronized `develop` when release preparation is
large enough to benefit from isolated review:

```text
release/0.2.0
```

Merge it into `develop` after profile, artifact, and clean-install checks pass.
Then open the release PR from `develop` to `main`.

### Hotfix branches

Create urgent fixes for an already published release from `main`, not from
`develop`:

```text
hotfix/0.2.1
```

This prevents unfinished `develop` features from entering a patch release.

## Normal feature workflow

1. Update local `develop` from `origin/develop`.
2. Create `feat/<feature-name>` from `develop`.
3. Implement and test one coherent feature.
4. Open a pull request from the feature branch to `develop`.
5. Review its complete diff and test evidence.
6. Merge it into `develop`.
7. Delete the short-lived feature branch after merge.

Squash merging a feature branch into `develop` is acceptable when its internal
commits are experimental or noisy. A regular merge is preferable when the
individual commits are intentionally reviewable and useful.

Direct commits to `develop` should be limited to explicitly coordinated,
small, sequential work while unrelated development is paused. The Stage 3
pre-merge hardening work was such an exception.

## Release workflow

1. Freeze unrelated changes to `develop`.
2. Run the release test, corpus, package, and installed-artifact gates.
3. Open a pull request from `develop` to `main`.
4. Use a regular merge commit; do not squash or rebase the long-lived branch.
5. Tag the exact tested `main` commit.
6. Publish the already-tested artifacts from that tag through trusted CI.
7. Smoke-test installation from public PyPI.
8. Fast-forward `develop` to the new `main` merge commit so both branches begin
   the next cycle from the same release boundary.

A regular `develop` to `main` merge preserves existing commit identities and
records a clear release boundary. Squashing the long-lived branch loses that
history and complicates later comparisons.

## Hotfix workflow

1. Create `hotfix/<patch-version>` from the current `main` release.
2. Make the smallest backwards-compatible fix.
3. Run focused regression and artifact checks.
4. Open a pull request from the hotfix branch to `main`.
5. Merge, tag the new patch version, and publish.
6. Fast-forward or merge the corrected `main` back into `develop`.

Both the old and new release remain in `main` history:

```text
main:     A [v0.2.0] ── B [v0.2.1]
develop:  A ── future work ── include B's fix
```

Git tags, not branch names, permanently identify published versions.

## Versioning

Use semantic versioning:

- Patch release: backwards-compatible fix, such as `0.2.1`.
- Minor release: backwards-compatible feature or significant pre-1.0 contract
  change, such as `0.3.0`.
- Major release after 1.0: incompatible public API change, such as `2.0.0`.

For MoVer:

- Renderer-first dependency packaging is `0.2.0`.
- A small fix immediately afterward is `0.2.1`.
- Browser pooling and configurable timeline selection can ship together as
  `0.3.0`, or in separate minor releases if independent review is safer.
- Optional prereleases may use `0.3.0a1`, `0.3.0b1`, and `0.3.0rc1`.

PyPI artifacts are immutable. Never overwrite a published version; make a new
version, tag, and release instead.

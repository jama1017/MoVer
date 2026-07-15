# MoVer 0.3.0

MoVer 0.3.0 generalizes GSAP capture so animations no longer need a global
timeline named `tl`.

## Highlights

- Automatically aggregates every GSAP root animation that exists when capture
  initializes, including legacy `tl`, renamed or sibling timelines, and
  standalone tweens.
- Preserves deterministic seeking and replay through one retained
  `gsap.exportRoot()` wrapper.
- Adds the optional `capture_duration` Python argument and
  `--capture-duration` CLI option for infinitely repeating animations.
- Detects unsupported loose root animations created after initialization
  instead of silently omitting them.

## Upgrade notes

- Existing pages with one `tl` remain supported without changes.
- Pages that previously had animations outside `tl` now capture those sibling
  root animations too.
- Any captured infinite repeat requires an explicit finite capture duration.
- Callback-created animations are supported when attached to an already
  captured parent timeline. Loose roots created after initialization remain
  unsupported.

## Release scope

This release contains the automatic all-root timeline contract. Experimental
batched raster capture and browser pooling are not included.

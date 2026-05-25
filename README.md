# hermes-plugins

Sovereign source of truth for Hermes plugin overrides maintained by apnex.

This repository hosts agent-agnostic Hermes plugins that are layered on top of
the stock plugins shipped inside the `hermes-agent` image at runtime. It is
consumed by `apnex/hermes` via a `plugin-sync` initContainer that clones this
repo into the plugins overlay path before the agent process starts — exactly
mirroring the `skill-sync` pattern already used for the skill library.

## Contents

| Path | Purpose |
|---|---|
| `honcho-tuned/` | Sovereign fork of the bundled `honcho` memory plugin from `NousResearch/hermes-agent`, with deterministic distillation injection (Tier 1) and configurable dialectic prompts (Tier 3a) per the honcho-reasoning-engagement proposal. Loaded as `honcho-tuned` to override the bundled `honcho` provider. |
| `UPSTREAM.md` | Provenance pin + rebase strategy for honcho-tuned. |
| `LICENSE` | MIT (preserved from upstream). |

## How it is consumed

`apnex/hermes` runs a `plugin-sync` initContainer that:

1. Clones this repo (pinned to a tag, e.g. `v1.0.0`) into an emptyDir volume.
2. Mounts that volume into the hermes container at the plugins overlay path.
3. Hermes' plugin loader sees `honcho-tuned` and registers it alongside the
   bundled providers; the active memory provider is selected by config.

The bundled `honcho` plugin remains untouched in the image — `honcho-tuned`
is a peer that supersedes it via configuration, never a patch on top.

## Versioning

- `v0.1.0` — Initial fork, byte-identical to upstream `plugins/memory/honcho/`
  at `NousResearch/hermes-agent@a91a57fa5a13` (tag `v2026.5.16`). Renamed only.
- `v1.0.0` — Tier 1 distillation block + Tier 3a configurable dialectic prompts
  applied. (Pending — see honcho-tuned-fork-and-deploy plan §P2.)

The diff between `v0.1.0` and `v1.0.0` is purely the engagement patches, with
no upstream noise mixed in.

## License

MIT — see `LICENSE`. Copyright and attribution to Nous Research preserved
from upstream.

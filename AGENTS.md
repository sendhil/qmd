# Fork Maintenance Instructions

This is `sendhil/qmd`, a thin personal fork of `tobi/qmd`.

## Non-negotiable invariants

- `main` mirrors `upstream/main`; fork changes belong on `non-chinese-defaults`.
- Default model URIs must comply with the policy in `docs/UPSTREAM_MAINTENANCE.md`.
- Never restore a Qwen, DeepSeek, BGE/BAAI, GTE/Alibaba, MiniCPM, GLM, Yi, or InternLM default while resolving conflicts.
- Explicit user model overrides remain supported.
- Never silently fall back to or download a different model family.
- Preserve QMD's full deep pipeline and `pi-memory` compatibility.
- Preserve the audited Jina reranker default unless a newly audited replacement passes the real-model gate.
- Do not run networked model tests until their explicit URIs have been audited.

Before syncing, releasing, or changing a model, read and follow
`docs/UPSTREAM_MAINTENANCE.md` completely.

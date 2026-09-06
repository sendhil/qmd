# Upstream Maintenance

## Remotes and branches

- `origin`: `https://github.com/sendhil/qmd.git`
- `upstream`: `https://github.com/tobi/qmd.git`
- `main`: clean mirror of `upstream/main`
- `non-chinese-defaults`: distributable fork branch

Verify before changing anything:

```sh
git remote -v
git status --short --branch
```

Stop if the worktree is dirty or either remote points somewhere unexpected.

## Synchronize upstream

```sh
git fetch --prune upstream
git switch main
git merge --ff-only upstream/main
git push origin main
git switch non-chinese-defaults
git merge main
```

Never rebase an already distributed fork tag. Resolve conflicts by accepting
unrelated upstream behavior while preserving the invariants in `AGENTS.md`.

## Conflict hotspots

Review `src/llm.ts`, `src/query-expansion-profile.ts`, model configuration,
CLI initialization, trust gates, package scripts, README model tables, and
`node-llama-cpp` upgrades. If upstream adds a generic profile abstraction,
adapt the fork to it and delete redundant fork code.

## Model audit

The currently approved defaults are:

- Embedding: `hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf`
- Generation: `hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/qmd-query-expansion-granite-2b-grpo-q4_k_m.gguf`
- Reranking: `hf:ggml-org/jina-reranker-v1-turbo-en-GGUF/Jina-Bert-Implementation-38M-F16.gguf`

Preserve the Jina reranker default unless a newly audited replacement passes
the real-model gate below. Do not restore the previously considered IBM
reranker merely because it appears in an older design or implementation note.

For every proposed default record its developer, base model, adapters/merges,
distillation teachers, training-label provenance, license, GGUF converter and
source revision, artifact size, and QMD API test results. Multilingual Chinese
support is allowed; Chinese-developed model ancestry is not.

### Approved Jina reranker audit record

- Developer and jurisdiction: Jina AI, a German/EU company.
- Architecture and base: English JinaBERT cross-encoder, 37.8 million
  parameters.
- Distillation: distilled from the unavailable `jina-reranker-v1-base-en`
  teacher checkpoint.
- Provenance exception: the inaccessible teacher weights and partially opaque
  teacher/training-label provenance cannot be independently audited. The user
  explicitly accepted this under the practical-lineage policy.
- License: Apache-2.0.
- Conversion: official `ggml-org` GGUF conversion at
  `hf:ggml-org/jina-reranker-v1-turbo-en-GGUF/Jina-Bert-Implementation-38M-F16.gguf`.
- Verified artifact: 76,971,168 bytes; SHA-256
  `71abc010bb3dce97812ee971509a5cb6ff6f6b8cfffd8480129242f605521fca`.
- QMD gate: expected document ranked first in 6/7 fixtures and within the top
  two in 7/7; the intent-aware full deep-search fixture ranked first. All
  scores were finite and within `[0, 1]`, and QMD reported the exact configured
  reranker rather than a fallback.

Preserve this Jina default unless a replacement receives an equally complete
audit and passes the real-model gate.

### Approved Granite generator provenance exception

The generator URI is
`hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/qmd-query-expansion-granite-2b-grpo-q4_k_m.gguf`.
It is a QMD-specific GRPO checkpoint based on IBM Granite 3.3 2B. Some of its
label dataset has incomplete generator provenance; the user explicitly
accepted that limitation under the same practical-lineage policy. Preserve
this caveat during upstream merges and model changes. The complete policy and
test thresholds are recorded in
[`docs/superpowers/specs/2026-09-05-non-chinese-model-defaults-design.md`](superpowers/specs/2026-09-05-non-chinese-model-defaults-design.md#model-policy).

## Validation

First confirm default URIs manually. Then run:

```sh
npm test
npm run lint
GGML_METAL_NO_RESIDENCY=1 npm run smoke:non-chinese-models
npm pack --dry-run
```

The real-model smoke command is Node-based, opt-in, and may download approved
models. Never run it until its resolved URIs pass the audit. A replacement is
not approved unless this real-model gate passes with the proposed URI.

## Tag and install

Use a tag composed from the current upstream version and fork revision. The
published `non-chinese-v2.8.3.1` tag cannot complete npm's Git-dependency
`prepare` build because TypeScript was not a direct development dependency.
It is immutable: never move or recommend it. The planned corrective release is:

```sh
git tag -a non-chinese-v2.8.3.2 -m "QMD 2.8.3 with approved model defaults and Git install fix"
git push origin non-chinese-defaults
git push origin non-chinese-v2.8.3.2
npm install -g github:sendhil/qmd#non-chinese-v2.8.3.2
```

Do not reuse or move a published tag.

## Rollback

Until a usable immutable fork release exists, remove the forked package rather
than reinstalling upstream's denied defaults:

```sh
npm uninstall -g @tobilu/qmd
command -v qmd || true
```

The expected result is that `qmd` is no longer found. Do not delete model
caches or indexes. Never use defective `non-chinese-v2.8.3.1` as a rollback
target. For later releases, roll back by installing the preceding verified,
immutable `non-chinese-v*` tag. Never reset or force-push the shared branch.

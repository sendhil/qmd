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

## Validation

First confirm default URIs manually. Then run:

```sh
npm test
npm run lint
npm run smoke:non-chinese-models
npm pack --dry-run
```

The real-model smoke command is Node-based, opt-in, and may download approved
models. Never run it until its resolved URIs pass the audit. A replacement is
not approved unless this real-model gate passes with the proposed URI.

## Tag and install

Use a tag composed from the current upstream version and fork revision. For the
initial release this is:

```sh
git tag -a non-chinese-v2.8.3.1 -m "QMD 2.8.3 with approved model defaults"
git push origin non-chinese-defaults
git push origin non-chinese-v2.8.3.1
npm install -g github:sendhil/qmd#non-chinese-v2.8.3.1
```

Do not reuse or move a published tag.

## Rollback

For the first release there is no earlier approved fork release. Remove the
forked package rather than reinstalling upstream's denied defaults:

```sh
npm uninstall -g @tobilu/qmd
command -v qmd || true
```

The expected result is that `qmd` is no longer found. Do not delete model
caches or indexes. For later releases, roll back by installing the preceding
immutable `non-chinese-v*` tag. Never reset or force-push the shared branch.

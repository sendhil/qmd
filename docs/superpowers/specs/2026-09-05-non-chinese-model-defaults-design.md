# Non-Chinese Model Defaults Fork Design

## Summary

Maintain `sendhil/qmd` as a thin personal fork of `tobi/qmd` that works out of the box with `pi-memory` and standalone QMD use without downloading Chinese-developed model weights by default. The fork retains QMD's complete deep-search pipeline: automatic query expansion, BM25 and vector retrieval, HyDE, fusion, and cross-encoder reranking.

The fork changes only model policy, query-expansion model adaptation, tests, and supporting documentation. It does not fork QMD's storage, indexing, retrieval, CLI, MCP, or SDK architecture.

## Goals

- A fresh installation from `sendhil/qmd` downloads only approved default models.
- `pi-memory` can discover the `qmd` executable normally and use keyword, semantic, and deep modes without modification.
- A plain deep query still receives automatic `lex:`, `vec:`, and `hyde:` expansion.
- QMD remains useful independently for other Markdown collections, CLI use, SDK use, and MCP use.
- The fork remains straightforward for an agent to update from `tobi/qmd`.
- Custom model URI overrides continue to work.
- Failures never silently download or switch to Qwen or another unapproved model.

## Model Policy

For the initial fork, "non-Chinese model" means that deployed weights must not descend from a model developed by a Chinese organization. This excludes Qwen, DeepSeek, BGE/BAAI, GTE/Alibaba, MiniCPM, GLM, Yi, InternLM, and their derivatives, merges, distillations, and post-trained variants.

Multilingual support, including support for Chinese text, does not by itself violate this policy. The initial fork accepts partially opaque label-dataset provenance for the community QMD expansion checkpoint, and documents that limitation. It also accepts Jina's documented knowledge-distillation lineage even though the Jina base reranker teacher checkpoint is not public, so the complete teacher-weight provenance cannot be independently audited. Both exceptions are accepted under the user's practical-lineage policy; strictly audited training-data provenance and retraining are outside the initial scope.

Before changing a default model, maintenance must review its developer, base-weight ancestry, adapters or merges, distillation lineage, training-label provenance when available, license, GGUF conversion provenance, and behavior through QMD's exact `node-llama-cpp` APIs.

## Approved Initial Defaults

| Role | Default URI | Rationale |
| --- | --- | --- |
| Embedding | `hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf` | Existing Google DeepMind default; no change required. |
| Expansion | `hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/qmd-query-expansion-granite-2b-grpo-q4_k_m.gguf` | QMD-specific `lex:/vec:/hyde:` checkpoint based on IBM Granite 3.3 2B. |
| Reranking | `hf:ggml-org/jina-reranker-v1-turbo-en-GGUF/Jina-Bert-Implementation-38M-F16.gguf` | Official ggml-org conversion of Jina AI's Apache-2.0 English JinaBERT cross-encoder; EU-developed, 37.8M parameters, and verified through QMD's exact ranking API. |

Approximate model downloads are 1.55 GB for the expansion model and 77 MB for the reranker, plus the existing EmbeddingGemma model. Model binaries remain in QMD's normal cache and are never committed to Git.

### Reranker candidate decision

The originally proposed IBM Granite R2 GGUF, `hf:keisuke-miyako/granite-embedding-reranker-english-r2-gguf-q8_0/granite-embedding-reranker-english-r2-Q8_0.gguf`, loaded successfully but failed the approved relevance gate: expected-document ranks were `[1, 5, 5, 2, 1, 2, 5]`, only two of seven cases ranked first, and three fell below second. Testing the exact ModernBERT pair-token template did not improve the top-one count, so the issue was not resolved by separator formatting.

The corrected Mixedbread candidate, `hf:cstr/mxbai-rerank-base-v1-GGUF/mxbai-rerank-base-v1-q8_0.gguf`, was also rejected. Its 198.82 MB GGUF downloaded, but QMD's pinned `node-llama-cpp` runtime refused to load it because the file does not define the required BERT token-type count. Direct reranking and full search both failed before scoring.

The selected Jina artifact is 76,971,168 bytes and uses the `jina-bert-v2` architecture with a context length of 8,192. It is the [official ggml-org conversion](https://huggingface.co/ggml-org/jina-reranker-v1-turbo-en-GGUF) of an Apache-2.0 English model from EU-based Jina AI. With expected input positions rotated to prevent order-based false positives, its seven expected-document ranks were `[1, 1, 1, 1, 1, 1, 2]`; QMD reported the exact configured model for every case and all scores were finite in `[0, 1]`. The full intent-aware search ranked the expected web-performance document first in 21.98 seconds. [Jina's published model card](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en) reports BEIR NDCG@10 of 49.60 and a LlamaIndex RAG hit rate of 85.13. The model is knowledge-distilled from Jina's base reranker; that teacher checkpoint is not public, and the user explicitly accepts that limitation under the practical-lineage policy.

## Architecture

### Default resolution

QMD's existing precedence remains unchanged:

1. Configuration in `index.yml`
2. `QMD_*_MODEL` environment variables
3. Fork defaults

Only the built-in expansion and reranking URI constants change. EmbeddingGemma remains the default embedding model. Configuration or environment overrides remain explicit user choices and are not restricted by the fork.

### Query-expansion profiles

Query generation becomes model-profile-aware instead of assuming every generator is Qwen. A small internal profile describes:

- Optional system prompt
- User prompt template
- Output grammar
- Generation token limit and sampling settings

The built-in Granite profile uses the checkpoint's documented system instruction, omits Qwen's `/no_think`, and constrains generation to exactly:

- One `hyde:` line
- Two `lex:` lines
- Three `vec:` lines

Content remains limited to a single line per expansion. QMD's existing parsing, query-term preservation filter, cache, fallback, retrieval, fusion, and reranking behavior remains in place.

The profile selector is based on a normalized, case-insensitive configured generator URI. It selects the Granite profile when the URI or local filename contains `qmd-query-expansion-granite-2b`, the legacy Qwen profile when it contains `qwen`, and the generic profile otherwise. URI normalization is limited to lowercasing and extracting the repository/path and basename; it does not resolve the network resource.

Existing Qwen behavior remains as a compatibility profile for users who deliberately configure a Qwen URI, but no Qwen URI is a default and no Qwen model is downloaded implicitly. Unknown custom generators always use the documented generic profile; they do not produce a configuration error and are never silently treated as Qwen. The generic profile uses the same neutral system instruction, bounded six-line grammar, and token limit as Granite, without model-family-specific tokens such as `/no_think`.

The grammar fixes raw output order as one `hyde:`, two `lex:`, then three `vec:` lines. Existing query-term preservation parsing may discard a generated line, so the parsed result can contain fewer than six entries. A valid parsed result must retain at least one entry of each type and never more than six entries; otherwise QMD uses its existing original-query fallback.

### Failure behavior

Download, grammar, context, and generation failures use QMD's existing original-query fallback. The fallback contains the user's query as lexical and vector input and does not select or download another model. Reranker failure follows existing QMD behavior and must not switch model families.

### Pi-memory integration

No `pi-memory` code changes are required. Its `memory_search` deep mode continues to invoke `qmd query`; QMD performs expansion, hybrid retrieval, and reranking. The fork continues to expose the `qmd` binary and keeps the upstream package layout so Pi's executable detection works normally.

## Distribution

Development occurs on the `non-chinese-defaults` branch. Users may test the branch directly:

```sh
npm install -g github:sendhil/qmd#non-chinese-defaults
```

Work installations should use immutable fork tags, for example:

```sh
npm install -g github:sendhil/qmd#non-chinese-v2.8.3.1
```

Fork tags identify the upstream QMD version plus a fork revision. The package retains its upstream name and `qmd` binary because it is installed directly from GitHub rather than published alongside `@tobilu/qmd` in a registry.

## Upstream Maintenance

The fork uses these remotes:

- `origin`: `https://github.com/sendhil/qmd.git`
- `upstream`: `https://github.com/tobi/qmd.git`

`main` mirrors `upstream/main`. Fork behavior lives only on `non-chinese-defaults`. Updates follow this sequence:

1. Fetch `upstream` and inspect its release notes and model-related changes.
2. Fast-forward local `main` to `upstream/main` and push `origin/main`.
3. Merge the updated `main` into `non-chinese-defaults`; do not rewrite already published fork tags.
4. Resolve conflicts while preserving the model policy and profile behavior.
5. Run the complete unit, type, package, and fork-specific policy tests.
6. Run opt-in Node-based model smoke tests for expansion and reranking.
7. Inspect the package contents and confirm that first-run defaults contain only approved URIs.
8. Push `non-chinese-defaults` and create a new immutable fork tag.

A root `AGENTS.md` will summarize these invariants and direct agents to `docs/UPSTREAM_MAINTENANCE.md`. The detailed runbook will include commands, conflict guidance, model-audit requirements, validation commands, tagging, rollback, and a rule against running tests that download upstream defaults before verifying their URIs.

## Documentation

The README will document:

- The purpose and scope of the fork
- Direct GitHub installation from a branch or immutable tag
- First-run model names and approximate download sizes
- `pi-memory` installation and automatic collection behavior
- Standalone QMD collection, CLI, SDK, and MCP use
- Model-policy definition and dataset-provenance caveat
- How to override models explicitly
- A link to the upstream-maintenance runbook

`AGENTS.md` and `docs/UPSTREAM_MAINTENANCE.md` are required deliverables, not optional follow-up documentation.

## Testing

### Deterministic tests

- Assert the three exported default URIs equal the approved models.
- Assert no built-in default URI contains a denied model-family or organization token.
- Verify configuration and environment variables still override defaults with existing precedence.
- Verify the Granite profile supplies its system prompt and does not include `/no_think`.
- Verify the Granite grammar permits exactly one HyDE, two lexical, and three vector lines.
- Verify URI normalization and selection of Granite, explicit Qwen, and generic profiles.
- Verify unknown generators receive the bounded generic profile.
- Verify generation failure returns the original-query fallback without resolving another model.
- Run upstream unit tests under both Node and Bun where upstream already requires both.
- Run type checking and packaged-install smoke tests.

### Opt-in model smoke tests

Networked model tests run under Node and are excluded from ordinary unit tests. They cover at least:

- Database decision
- Package-manager preference
- Production deployment region
- Unit-test command
- Dark-mode preference
- An exact code identifier
- An ambiguous query with intent

After one unmeasured warm-up query, each expansion fixture has a 30-second timeout. Its raw output must match the fixed six-line grammar with no duplicate lines. Its parsed result must contain between three and six entries, include at least one entry of each type, contain no duplicate `(type, text)` pair, and retain at least one fixture-defined anchor term in a lexical or vector entry. Lexical entries may contain at most 120 characters, vector entries at most 240 characters, and the HyDE entry at most 400 characters. At least six of the seven fixtures must pass all content gates; timeout, malformed output, or fallback is a fixture failure.

The ambiguous-query fixture supplies an intent and validates the complete search path rather than requiring the expansion model itself to reproduce intent text. Its expected document must rank in the top two.

Reranking smoke tests contain seven query sets with one expected document and at least four realistic distractors. The expected document rotates across input positions, and each result must identify the exact configured reranker so fallback output cannot pass by preserving input order. Scores must be finite numbers in `[0, 1]`; the expected document must rank first in at least six sets and may not rank below second in any set. Results record ranks and score margins for later comparison. A successful GGUF load alone is not a passing result.

### Download-policy verification

Before tagging, inspect the resolved fresh configuration and package output. The test fails if any default model URI points to a denied family. Networked smoke tests must receive explicit model URIs and must never exercise an unreviewed upstream default accidentally.

## Upgrade Conflict Rules

Model-related upstream changes require deliberate review rather than automatic conflict resolution. In particular, inspect changes to:

- Default model constants and configuration generation
- `expandQuery()` prompts, grammar, parsing, caching, and fallback behavior
- Reranking APIs and `node-llama-cpp` versions
- Trust checks for project-local or remote model URIs
- CLI initialization that persists resolved defaults

Prefer adapting the fork profile layer to upstream abstractions over copying or replacing large upstream functions. Unrelated upstream changes should be accepted as-is.

## Security and Trust

The fork does not weaken QMD's trust gates for remote or project-local model URIs. Model downloads continue through QMD and `node-llama-cpp`. Community GGUF artifacts are pinned by repository and filename; immutable artifact hashes may be added later if the loader gains first-class support. A model URI change is treated as a supply-chain change and requires the audit described above.

## Non-goals

- Training a new query-expansion model in the initial implementation
- Guaranteeing that approved multilingual models saw no Chinese-language text
- Enforcing a denylist against user-supplied model overrides
- Publishing a competing npm package
- Modifying `pi-memory`
- Refactoring QMD's retrieval, storage, MCP, or SDK layers
- Bundling model binaries in the repository or package

## Acceptance Criteria

- A clean GitHub installation exposes a working `qmd` CLI.
- Fresh default configuration resolves only the three approved model URIs.
- Plain `qmd query` invokes automatic Granite expansion and the full deep pipeline.
- Raw expansion is bounded and ordered as one HyDE, two lexical, and three vector lines; parsed output retains at least one of each type or falls back safely.
- The selected Jina reranker runs through QMD's existing ranking API and passes the approved relevance thresholds.
- `pi-memory` deep search works without changes to `pi-memory`.
- No failure path silently downloads or selects Qwen.
- Upstream tests and fork-specific deterministic tests pass.
- Node-based model smoke tests pass before a release tag is created.
- `AGENTS.md`, the maintenance runbook, and user-facing installation documentation are complete.

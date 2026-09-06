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

Name fork releases `v<upstream>-sendhil.<revision>`, resetting the revision to
`1` for each new upstream baseline. For example, the canonical first release
from upstream 2.8.3 is `v2.8.3-sendhil.1`, and the first release after adopting
upstream 2.8.4 will be `v2.8.4-sendhil.1`.

The published `non-chinese-v2.8.3.1` and `non-chinese-v2.8.3.2` tags are
immutable historical tags from the old naming scheme: never move or delete
them. The `.1` tag also predates the direct TypeScript build dependency.
TypeScript is now explicit, but npm's open [global Git-dependency prepare
bug](https://github.com/npm/cli/issues/8440) still makes a global Git-dependency
install from `non-chinese-defaults` unsuitable. Publish and test a prebuilt
GitHub release asset instead.

Run the following only after all validation passes and the release commit has
been committed, rebased, and pushed. The sequence builds once, validates those
exact local bytes before tagging, uploads without rebuilding, then validates
the downloaded release and its public URL:

```sh
set -eu

RELEASE_TAG=v2.8.3-sendhil.1
RELEASE_ASSET=qmd-v2.8.3-sendhil.1.tgz

git status --short --branch
test "$(git branch --show-current)" = non-chinese-defaults
test -z "$(git status --porcelain)"
git pull --rebase origin non-chinese-defaults
git push origin non-chinese-defaults

RELEASE_BUILD_COMMIT=$(git rev-parse --short=7 HEAD)
RELEASE_DIR=/private/tmp/qmd-v2.8.3-sendhil.1-release-$RELEASE_BUILD_COMMIT
LOCAL_PREFIX=/private/tmp/qmd-v2.8.3-sendhil.1-local-$RELEASE_BUILD_COMMIT
DOWNLOAD_DIR=/private/tmp/qmd-v2.8.3-sendhil.1-download-$RELEASE_BUILD_COMMIT
URL_PREFIX=/private/tmp/qmd-v2.8.3-sendhil.1-url-$RELEASE_BUILD_COMMIT
RELEASE_PATH=$RELEASE_DIR/$RELEASE_ASSET

# Prove the canonical tag is unused locally and remotely before mutation.
test -z "$(git tag --list "$RELEASE_TAG")"
REMOTE_TAG=$(git ls-remote --tags origin "refs/tags/$RELEASE_TAG")
test -z "$REMOTE_TAG"

# Build exactly once; pack without lifecycle scripts so these bytes are not rebuilt.
mkdir "$RELEASE_DIR"
npm run build
npm pack --ignore-scripts --pack-destination "$RELEASE_DIR"
mv "$RELEASE_DIR/tobilu-qmd-2.8.3.tgz" "$RELEASE_PATH"

# Inspect the archive and its build stamp, then record its lowercase SHA-256.
tar -tzf "$RELEASE_PATH"
tar -xOf "$RELEASE_PATH" package/dist/cli/build-info.json
PACKED_BUILD_COMMIT=$(tar -xOf "$RELEASE_PATH" package/dist/cli/build-info.json | jq -r .commit)
test "$PACKED_BUILD_COMMIT" = "$RELEASE_BUILD_COMMIT"
RELEASE_SHA256=$(shasum -a 256 "$RELEASE_PATH" | awk '{print $1}')
test "${#RELEASE_SHA256}" -eq 64
printf '%s  %s\n' "$RELEASE_SHA256" "$RELEASE_PATH"

# Validate the exact local artifact before creating the immutable tag.
npm install -g --prefix "$LOCAL_PREFIX" "$RELEASE_PATH"
test "$("$LOCAL_PREFIX/bin/qmd" --version)" = "qmd 2.8.3 ($RELEASE_BUILD_COMMIT)"
git tag -a "$RELEASE_TAG" -m "QMD $RELEASE_TAG"
test "$(git rev-list -n 1 "$RELEASE_TAG")" = "$(git rev-parse HEAD)"

# The upstream-oriented hook treats every v* tag as an npm package-version tag;
# the explicit gates above replace that inapplicable check for this fork tag.
git push --no-verify origin "$RELEASE_TAG"
gh release create "$RELEASE_TAG" "$RELEASE_PATH" --repo sendhil/qmd --title "QMD v2.8.3-sendhil.1" --latest=false --notes "Canonical prebuilt sendhil/qmd fork release. Supersedes the historical non-chinese-v2.8.3.x naming."

# Download and compare the published bytes before testing the public URL.
mkdir "$DOWNLOAD_DIR"
gh release download "$RELEASE_TAG" --repo sendhil/qmd --pattern "$RELEASE_ASSET" --dir "$DOWNLOAD_DIR"
DOWNLOADED_PATH=$DOWNLOAD_DIR/$RELEASE_ASSET
DOWNLOADED_SHA256=$(shasum -a 256 "$DOWNLOADED_PATH" | awk '{print $1}')
test "$DOWNLOADED_SHA256" = "$RELEASE_SHA256"
cmp "$RELEASE_PATH" "$DOWNLOADED_PATH"

npm install -g --prefix "$URL_PREFIX" https://github.com/sendhil/qmd/releases/download/v2.8.3-sendhil.1/qmd-v2.8.3-sendhil.1.tgz
test "$("$URL_PREFIX/bin/qmd" --version)" = "qmd 2.8.3 ($RELEASE_BUILD_COMMIT)"
```

If any check fails before `git tag`, stop without creating or pushing the tag.
If anything fails after the tag is pushed, report it and repair the release
without moving or deleting the tag. Never rebuild between the first local
artifact validation and upload. Only after the downloaded SHA-256 and both
isolated installations agree may the README command replace a working global
installation. Never reuse or move a published tag.

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
immutable `v<upstream>-sendhil.<revision>` release asset. Never reset or
force-push the shared branch.

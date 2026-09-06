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

- Embedding: `hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf#0f741b5a6585bd53aeb15cd1372c56f2a0f65e12`
- Generation: `hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/qmd-query-expansion-granite-2b-grpo-q4_k_m.gguf#449c09bced7605802af16c2b431fd40e5e871b8c`
- Reranking: `hf:ggml-org/jina-reranker-v1-turbo-en-GGUF/Jina-Bert-Implementation-38M-F16.gguf#8582fa8560bcdd3c5cbc9015514edff0f3b1871f`

`node-llama-cpp` uses `#` for an HF revision; never substitute `@`. The exact
former floating URI for each built-in is a compatibility alias only in its
matching role: QMD canonicalizes it to the pinned URI before config persistence
and trust checks. Arbitrary user overrides do not acquire built-in checksums.

### Immutable artifact manifest

| Role | Revision | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| EmbeddingGemma | `0f741b5a6585bd53aeb15cd1372c56f2a0f65e12` | 333,590,944 | `b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63` |
| Granite expansion | `449c09bced7605802af16c2b431fd40e5e871b8c` | 1,545,302,752 | `a488061ddcf8ee3e18912cd29cb33cbe6396084946de75544650ac5620e5b6ed` |
| Jina reranker | `8582fa8560bcdd3c5cbc9015514edff0f3b1871f` | 76,971,168 | `71abc010bb3dce97812ee971509a5cb6ff6f6b8cfffd8480129242f605521fca` |

QMD streams every built-in artifact through size, GGUF-magic, and SHA-256
validation on cache reuse, download, load, `qmd pull`, and `qmd doctor`. A
mismatch is never loaded; QMD removes only the revision-specific target and
downloads it again, while a bad fresh download fails hard. Built-ins do not use
mutable `/resolve/main` or ETag freshness checks. A pinned embedding URI also
intentionally changes its fingerprint, causing one safe re-embed.

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
  `hf:ggml-org/jina-reranker-v1-turbo-en-GGUF/Jina-Bert-Implementation-38M-F16.gguf#8582fa8560bcdd3c5cbc9015514edff0f3b1871f`.
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
`hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/qmd-query-expansion-granite-2b-grpo-q4_k_m.gguf#449c09bced7605802af16c2b431fd40e5e871b8c`.
It is a QMD-specific GRPO checkpoint based on IBM Granite 3.3 2B. Some of its
label dataset has incomplete generator provenance; the user explicitly
accepted that limitation under the same practical-lineage policy. Preserve
this caveat during upstream merges and model changes. The complete policy and
test thresholds are recorded in
[`docs/superpowers/specs/2026-09-05-non-chinese-model-defaults-design.md`](superpowers/specs/2026-09-05-non-chinese-model-defaults-design.md#model-policy).

## Validation

For a release, commit the planned release changes and pull/rebase before any
validation. This ensures the full suite validates the exact post-pull commit
that will be built and tagged:

```sh
git status --short --branch
test "$(git branch --show-current)" = non-chinese-defaults
test -z "$(git status --porcelain)"
git pull --rebase origin non-chinese-defaults
test -z "$(git status --porcelain)"
```

On that exact HEAD, first confirm default URIs manually. Then run the full
validation suite:

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
policy-protected historical tags from the old naming scheme: never move or
delete them. The `.1` tag also predates the direct TypeScript build dependency.
TypeScript is now explicit, but npm's open [global Git-dependency prepare
bug](https://github.com/npm/cli/issues/8440) still makes a global Git-dependency
install from `non-chinese-defaults` unsuitable. Publish and test a prebuilt
GitHub release asset instead.

GitHub's immutable-release setting is not enabled for this repository. Fork
policy therefore treats every published fork tag and asset as immutable: never
move or reuse a tag, and never replace or delete its asset.
`v2.8.3-sendhil.1` remains immutable historical evidence. The remediated
canonical release for this baseline is `v2.8.3-sendhil.2`; record its hash only
after the exact archive passes the local and public-byte checks below. Release
verification depends on those hash and byte comparisons.

Run the following only after the full validation suite above passes on the
committed, rebased HEAD. The sequence pushes that exact validated commit,
builds once, validates the local bytes before tagging, uploads without
rebuilding, then validates the downloaded release and its public URL:

```sh
set -eu

RELEASE_TAG=v2.8.3-sendhil.2
PACKAGE_VERSION=$(node -p 'require("./package.json").version')
PACKAGE_FILENAME=$(node -p 'const p = require("./package.json"); `${p.name.replace(/^@/, "").replaceAll("/", "-")}-${p.version}.tgz`')
RELEASE_TAG_PREFIX=v$PACKAGE_VERSION-sendhil.
RELEASE_REVISION=${RELEASE_TAG#"$RELEASE_TAG_PREFIX"}
test "$RELEASE_REVISION" != "$RELEASE_TAG"
case "$RELEASE_REVISION" in
  ""|*[!0-9]*) exit 1 ;;
esac
test "$RELEASE_REVISION" -ge 1

RELEASE_ASSET=qmd-$RELEASE_TAG.tgz
RELEASE_TITLE="QMD $RELEASE_TAG"
RELEASE_URL=https://github.com/sendhil/qmd/releases/download/$RELEASE_TAG/$RELEASE_ASSET
RELEASE_BUILD_COMMIT=$(git rev-parse --short=7 HEAD)
RELEASE_DIR=/private/tmp/qmd-$RELEASE_TAG-release-$RELEASE_BUILD_COMMIT
LOCAL_PREFIX=/private/tmp/qmd-$RELEASE_TAG-local-$RELEASE_BUILD_COMMIT
DOWNLOAD_DIR=/private/tmp/qmd-$RELEASE_TAG-download-$RELEASE_BUILD_COMMIT
URL_PREFIX=/private/tmp/qmd-$RELEASE_TAG-url-$RELEASE_BUILD_COMMIT
NPM_CACHE=/private/tmp/qmd-$RELEASE_TAG-npm-cache-$RELEASE_BUILD_COMMIT
URL_CACHE=/private/tmp/qmd-$RELEASE_TAG-url-cache-$RELEASE_BUILD_COMMIT
RELEASE_PATH=$RELEASE_DIR/$RELEASE_ASSET
DOWNLOADED_PATH=$DOWNLOAD_DIR/$RELEASE_ASSET
EXPECTED_VERSION="qmd $PACKAGE_VERSION ($RELEASE_BUILD_COMMIT)"

git status --short --branch
test "$(git branch --show-current)" = non-chinese-defaults
test -z "$(git status --porcelain)"
git push origin non-chinese-defaults
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/non-chinese-defaults)"
PREVIOUS_LATEST_TAG=$(gh api repos/sendhil/qmd/releases/latest --jq .tag_name)
test -n "$PREVIOUS_LATEST_TAG"

# Prove the canonical tag is unused locally and remotely before mutation.
test -z "$(git tag --list "$RELEASE_TAG")"
REMOTE_TAG=$(git ls-remote --tags origin "refs/tags/$RELEASE_TAG")
test -z "$REMOTE_TAG"

# Build exactly once; pack without lifecycle scripts so these bytes are not rebuilt.
mkdir "$RELEASE_DIR"
npm run build
npm pack --ignore-scripts --cache "$NPM_CACHE" --pack-destination "$RELEASE_DIR"
mv "$RELEASE_DIR/$PACKAGE_FILENAME" "$RELEASE_PATH"

# Inspect the archive and its build stamp, then record its lowercase SHA-256.
tar -tzf "$RELEASE_PATH"
tar -xOf "$RELEASE_PATH" package/dist/cli/build-info.json
PACKED_BUILD_COMMIT=$(tar -xOf "$RELEASE_PATH" package/dist/cli/build-info.json | jq -r .commit)
test "$PACKED_BUILD_COMMIT" = "$RELEASE_BUILD_COMMIT"
RELEASE_SHA256=$(shasum -a 256 "$RELEASE_PATH" | awk '{print $1}')
test "${#RELEASE_SHA256}" -eq 64
printf '%s  %s\n' "$RELEASE_SHA256" "$RELEASE_PATH"

# Validate the exact local artifact before creating the policy-protected tag.
npm install -g --prefix "$LOCAL_PREFIX" --cache "$NPM_CACHE" "$RELEASE_PATH"
test "$("$LOCAL_PREFIX/bin/qmd" --version)" = "$EXPECTED_VERSION"
git tag -a "$RELEASE_TAG" -m "$RELEASE_TITLE"
test "$(git rev-list -n 1 "$RELEASE_TAG")" = "$(git rev-parse HEAD)"

# The upstream-oriented hook treats every v* tag as an npm package-version tag;
# the explicit gates above replace that inapplicable check for this fork tag.
git push --no-verify origin "$RELEASE_TAG"
gh release create "$RELEASE_TAG" "$RELEASE_PATH" --repo sendhil/qmd --title "$RELEASE_TITLE" --latest=false --notes "Canonical prebuilt sendhil/qmd fork release. Supersedes the historical non-chinese-v2.8.3.x naming."

# GitHub has previously ignored --latest=false for this fork. Restore the
# release that was latest before this immutable asset was published.
if test "$(gh api repos/sendhil/qmd/releases/latest --jq .tag_name)" = "$RELEASE_TAG"; then
  gh release edit "$PREVIOUS_LATEST_TAG" --repo sendhil/qmd --latest
fi
test "$(gh api repos/sendhil/qmd/releases/latest --jq .tag_name)" = "$PREVIOUS_LATEST_TAG"

# Download and compare the published bytes before testing the public URL.
mkdir "$DOWNLOAD_DIR"
gh release download "$RELEASE_TAG" --repo sendhil/qmd --pattern "$RELEASE_ASSET" --dir "$DOWNLOAD_DIR"
DOWNLOADED_SHA256=$(shasum -a 256 "$DOWNLOADED_PATH" | awk '{print $1}')
test "$DOWNLOADED_SHA256" = "$RELEASE_SHA256"
cmp "$RELEASE_PATH" "$DOWNLOADED_PATH"

npm install -g --prefix "$URL_PREFIX" --cache "$URL_CACHE" "$RELEASE_URL"
test "$("$URL_PREFIX/bin/qmd" --version)" = "$EXPECTED_VERSION"
```

If any check fails before `git tag`, stop without creating or pushing the tag.
If anything fails after the tag is pushed, report it and repair the release
without moving or deleting the tag. Never rebuild between the first local
artifact validation and upload. Only after the downloaded SHA-256 and both
isolated installations agree may the README command replace a working global
installation. Never reuse or move a published tag.

## Rollback

Until a usable policy-protected fork release exists, remove the forked package
rather than reinstalling upstream's denied defaults:

```sh
npm uninstall -g @tobilu/qmd
command -v qmd || true
```

The expected result is that `qmd` is no longer found. Do not delete model
caches or indexes. Never use defective `non-chinese-v2.8.3.1` as a rollback
target. For later releases, roll back by installing the preceding verified,
policy-protected `v<upstream>-sendhil.<revision>` release asset. Never reset or
force-push the shared branch.

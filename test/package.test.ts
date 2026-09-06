import { describe, expect, test } from "vitest";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { tmpdir } from "node:os";

const root = new URL("..", import.meta.url);
const pkg = JSON.parse(readFileSync(new URL("package.json", root), "utf8"));

describe("package test task", () => {
  test("ships fork-safe installation guidance and only the runtime QMD skill", () => {
    const npmCache = mkdtempSync(join(tmpdir(), "qmd-package-test-npm-cache-"));
    let packed: Array<{ files: Array<{ path: string }> }>;
    try {
      packed = JSON.parse(execFileSync(
        "npm",
        ["pack", "--dry-run", "--json", "--ignore-scripts"],
        {
          cwd: root.pathname,
          encoding: "utf8",
          env: { ...process.env, npm_config_cache: npmCache },
        },
      ));
    } finally {
      rmSync(npmCache, { recursive: true, force: true });
    }
    const markdownPaths = packed[0].files
      .map(({ path }) => path)
      .filter((path) => path.endsWith(".md"));
    const packagedMarkdown = markdownPaths
      .map((path) => readFileSync(new URL(path.replace(/^package\//, ""), root), "utf8"))
      .join("\n");

    expect(markdownPaths).toContain("skills/qmd/SKILL.md");
    expect(markdownPaths).not.toContain("skills/release/SKILL.md");
    expect(packagedMarkdown).not.toMatch(/(?:npm|bun) install -g @tobilu\/qmd/);
    expect(packagedMarkdown).not.toContain("git clone https://github.com/tobi/qmd");
    expect(packagedMarkdown).not.toContain("claude plugin marketplace add tobi/qmd");
  });

  test("routes fork releases away from the upstream release script and publish workflow", () => {
    const releaseScript = readFileSync(new URL("scripts/release.sh", root), "utf8");
    const publishWorkflow = readFileSync(new URL(".github/workflows/publish.yml", root), "utf8");
    const ciWorkflow = readFileSync(new URL(".github/workflows/ci.yml", root), "utf8");

    expect(releaseScript).toContain('UPSTREAM_ORIGIN="https://github.com/tobi/qmd.git"');
    expect(releaseScript).toContain("docs/UPSTREAM_MAINTENANCE.md");
    expect(releaseScript).toContain("must be the canonical upstream repository");
    expect(publishWorkflow).toContain("github.repository == 'tobi/qmd'");
    expect(ciWorkflow).toContain("branches: [main, non-chinese-defaults]");
  });

  test("recommends the exact canonical prebuilt fork install", () => {
    const readme = readFileSync(new URL("README.md", root), "utf8");
    const releaseUrl =
      "https://github.com/sendhil/qmd/releases/download/v2.8.3-sendhil.3/qmd-v2.8.3-sendhil.3.tgz";

    expect(readme).toContain(`npm install -g ${releaseUrl}`);
    expect(readme).toContain("v2.8.3-sendhil.2 is immutable historical evidence");
  });

  test("does not recommend historical release assets", () => {
    const readme = readFileSync(new URL("README.md", root), "utf8");

    expect(readme).not.toMatch(
      /npm install -g [^\n]*releases\/download\/non-chinese-v2\.8\.3\.[12]\//,
    );
    expect(readme).not.toMatch(
      /npm install -g [^\n]*releases\/download\/v2\.8\.3-sendhil\.[12]\//,
    );
  });

  test("does not recommend common Git branch install forms", () => {
    const readme = readFileSync(new URL("README.md", root), "utf8");

    for (const command of [
      "npm install -g github:sendhil/qmd#non-chinese-defaults",
      "npm install -g sendhil/qmd#non-chinese-defaults",
      "npm install -g git+https://github.com/sendhil/qmd.git#non-chinese-defaults",
      "npm install -g https://github.com/sendhil/qmd.git#non-chinese-defaults",
    ]) {
      expect(readme).not.toContain(command);
    }
  });

  test("describes release immutability as fork policy rather than GitHub enforcement", () => {
    const readme = readFileSync(new URL("README.md", root), "utf8");

    expect(readme).toContain("GitHub's immutable-release setting is not enabled");
    expect(readme).toContain("fork policy treats the tag and asset as immutable");
    expect(readme).not.toContain("immutable GitHub release");
  });

  test("excludes fork release tags from the npm publish workflow", () => {
    const publishWorkflow = readFileSync(
      new URL(".github/workflows/publish.yml", root),
      "utf8",
    );

    expect(publishWorkflow).toContain(
      'tags: ["v*", "!v*-sendhil.*"]',
    );
  });

  test("validates the pulled release commit before the full release suite", () => {
    const maintenance = readFileSync(
      new URL("docs/UPSTREAM_MAINTENANCE.md", root),
      "utf8",
    );
    const validationStart = maintenance.indexOf("## Validation");
    const releaseStart = maintenance.indexOf("## Tag and install");
    const rollbackStart = maintenance.indexOf("## Rollback");
    const validationSection = maintenance.slice(validationStart, releaseStart);
    const releaseSection = maintenance.slice(releaseStart, rollbackStart);

    expect(validationStart).toBeGreaterThanOrEqual(0);
    expect(releaseStart).toBeGreaterThan(validationStart);
    expect(rollbackStart).toBeGreaterThan(releaseStart);
    expect(validationSection).toContain(
      "git pull --rebase origin non-chinese-defaults",
    );
    expect(validationSection.indexOf("git pull --rebase")).toBeLessThan(
      validationSection.indexOf("npm test"),
    );
    expect(releaseSection).not.toContain("git pull --rebase");
  });

  test("derives reusable release values from the tag and package metadata", () => {
    const maintenance = readFileSync(
      new URL("docs/UPSTREAM_MAINTENANCE.md", root),
      "utf8",
    );
    const releaseSection = maintenance.slice(
      maintenance.indexOf("## Tag and install"),
      maintenance.indexOf("## Rollback"),
    );
    const commandBlock = releaseSection.match(/```sh\n([\s\S]*?)\n```/)?.[1] ?? "";

    expect(commandBlock.match(/v2\.8\.3-sendhil\.3/g)).toHaveLength(1);
    expect(commandBlock).toMatch(
      /PACKAGE_VERSION=\$\(node -p [^\n]*package\.json/,
    );
    expect(commandBlock).toMatch(
      /PACKAGE_FILENAME=\$\(node -p [^\n]*package\.json/,
    );
    expect(commandBlock).toContain("RELEASE_ASSET=qmd-$RELEASE_TAG.tgz");
    expect(commandBlock).toContain('RELEASE_TITLE="QMD $RELEASE_TAG"');
    expect(commandBlock).toContain(
      "RELEASE_URL=https://github.com/sendhil/qmd/releases/download/$RELEASE_TAG/$RELEASE_ASSET",
    );
    expect(commandBlock).toContain(
      "RELEASE_DIR=/private/tmp/qmd-$RELEASE_TAG-release-$RELEASE_BUILD_COMMIT",
    );
    expect(commandBlock).toContain(
      'mv "$RELEASE_DIR/$PACKAGE_FILENAME" "$RELEASE_PATH"',
    );
    expect(commandBlock).toContain('--title "$RELEASE_TITLE"');
    expect(commandBlock).toContain('"$RELEASE_URL"');
    expect(commandBlock).toContain(
      'PREVIOUS_LATEST_TAG=$(gh api repos/sendhil/qmd/releases/latest --jq .tag_name)',
    );
    expect(commandBlock).toContain('gh release edit "$PREVIOUS_LATEST_TAG" --repo sendhil/qmd --latest');
  });

  test("declares TypeScript directly for Git prepare builds", () => {
    expect(pkg.scripts.prepare).toContain("scripts/build.mjs");
    expect(pkg.devDependencies?.typescript).toBe("5.9.3");
  });

  test("exposes an opt-in non-Chinese model smoke test", () => {
    expect(pkg.scripts["smoke:non-chinese-models"]).toBe(
      "tsx scripts/smoke-non-chinese-models.ts",
    );
  });

  test("keeps the model smoke gate explicit, fallback-safe, and lifecycle-safe", () => {
    const testAllScript = readFileSync(new URL("scripts/test-all.mjs", root), "utf8");
    const smokeScript = readFileSync(
      new URL("scripts/smoke-non-chinese-models.ts", root),
      "utf8",
    );

    expect(testAllScript).not.toContain("smoke:non-chinese-models");
    expect(smokeScript).toContain("embedModel: DEFAULT_EMBED_MODEL_URI");
    expect(smokeScript).toContain("generateModel: DEFAULT_GENERATE_MODEL_URI");
    expect(smokeScript).toContain("rerankModel: DEFAULT_RERANK_MODEL_URI");
    expect(smokeScript).toContain("model !== DEFAULT_RERANK_MODEL_URI");
    expect(smokeScript).toContain("const expectedPosition = fixtureIndex %");
    expect(smokeScript).toContain("controller.abort");
    expect(smokeScript).not.toContain("Promise.race");
    expect(smokeScript).toContain("if (root) await rm(root, { recursive: true, force: true })");
  });

  test("runs typecheck, unit tests, and package smoke checks", () => {
    expect(pkg.scripts.test).toContain("scripts/test-all.mjs");

    expect(pkg.scripts["test:types"]).toContain("tsconfig.build.json --noEmit");
    expect(pkg.scripts["test:unit"]).toContain("vitest.mjs");
    expect(pkg.scripts["test:unit"]).toContain("bun test");
    expect(pkg.scripts["test:unit"]).toContain("CI=true");

    expect(pkg.scripts["test:package"]).toContain("scripts/package-smoke.mjs");

    const testAllScript = readFileSync(new URL("scripts/test-all.mjs", root), "utf8");
    expect(testAllScript).toContain("TypeScript build typecheck");
    expect(testAllScript).toContain("Vitest suite under Node");
    expect(testAllScript).toContain("Bun test suite");
    expect(testAllScript).toContain("Package smoke");

    const packageSmokeScript = readFileSync(new URL("scripts/package-smoke.mjs", root), "utf8");
    expect(packageSmokeScript).toContain("scripts/build.mjs");
    expect(packageSmokeScript).toContain("scripts/check-package-grammars.mjs");
    expect(packageSmokeScript).toContain("compiled CLI under Node");
    expect(packageSmokeScript).toContain("compiled CLI under Bun");
    expect(packageSmokeScript).toContain("package wrapper");
  });
});

describe("package grammar distribution", () => {
  test("installs AST grammar wasm packages as required runtime dependencies", () => {
    for (const dep of ["tree-sitter-typescript", "tree-sitter-python", "tree-sitter-go", "tree-sitter-rust"]) {
      expect(pkg.dependencies, `${dep} should be a required dependency`).toHaveProperty(dep);
      expect(pkg.optionalDependencies ?? {}, `${dep} should not be optional`).not.toHaveProperty(dep);
    }
  });

  test("documents a packaging smoke check for grammar wasm availability", () => {
    expect(pkg.scripts, "package.json scripts").toHaveProperty("smoke:package-grammars");
    expect(String(pkg.scripts["smoke:package-grammars"])).toContain("check-package-grammars");

    expect(pkg.files, "published package files").toContain("scripts/build.mjs");
    expect(pkg.files, "published package files").toContain("scripts/check-package-grammars.mjs");
    expect(pkg.files, "published package files").toContain("scripts/package-smoke.mjs");
    expect(pkg.files, "published package files").toContain("scripts/test-all.mjs");
    expect(pkg.files, "published package files").toContain("skills/qmd/");
    const qmdSkill = readFileSync(new URL("skills/qmd/SKILL.md", root), "utf8");
    expect(qmdSkill).toContain("# QMD - Query Markdown Documents");
    expect(qmdSkill).toContain("## How search works");
    expect(qmdSkill).toContain("## MCP Tool: `query`");
    expect(qmdSkill).not.toContain("This file is a discovery stub");

    const firstSixtyLines = qmdSkill.split(/\r?\n/).slice(0, 60).join("\n");
    expect(firstSixtyLines).toContain("Search for candidate documents");
    expect(firstSixtyLines).toContain("qmd search");
    expect(firstSixtyLines).toContain('qmd multi-get "#abc123,#def432"');
    expect(firstSixtyLines).toContain("Retrieved:");
    expect(firstSixtyLines).toContain("qmd query");
    // The skill must teach structured, self-authored queries near the top.
    expect(firstSixtyLines).toContain("Default to structured");

    const scriptPath = join(root.pathname, "scripts", "check-package-grammars.mjs");
    const script = readFileSync(scriptPath, "utf8");
    expect(script).toContain("tree-sitter-typescript/tree-sitter-typescript.wasm");
    expect(script).toContain("tree-sitter-typescript/tree-sitter-tsx.wasm");
  });
});

describe("Nix flake package layout", () => {
  test("installPhase copies skills/ next to src/ so findPackageRoot can resolve them (#722)", () => {
    const flake = readFileSync(new URL("flake.nix", root), "utf8");

    // The bun wrapper runs $out/lib/qmd/src/cli/qmd.ts. findPackageRoot() walks
    // up from that file looking for a sibling skills/ directory, so skills must
    // land at the same $out/lib/qmd prefix as src — not only in the source tree.
    expect(flake).toContain("cp -r src $out/lib/qmd/");
    expect(flake).toContain("cp -r skills $out/lib/qmd/");
    expect(flake).toContain("cp package.json $out/lib/qmd/");
  });

  test("makeWrapper seeds the same pre-import env as bin/qmd (#723)", () => {
    const flake = readFileSync(new URL("flake.nix", root), "utf8");
    const launcher = readFileSync(new URL("bin/qmd", root), "utf8");

    // Nix installs skip bin/qmd and exec bun src/cli/qmd.ts. The wrapper must
    // still set these BEFORE the native binding loads, matching the launcher.
    for (const env of [
      "LLAMA_LOG_LEVEL",
      "GGML_LOG_LEVEL",
      "GGML_BACKEND_SILENT",
      "GGML_METAL_NO_RESIDENCY",
      "QMD_METAL_KEEP_RESIDENCY",
    ]) {
      expect(launcher, `bin/qmd should set ${env}`).toContain(env);
      expect(flake, `flake.nix wrapper should set ${env}`).toContain(env);
    }

    expect(flake).toContain('--run');
    expect(flake).toContain('$1" = mcp');
    expect(flake).toContain('$(uname -s)" = Darwin');
    expect(flake).toContain('LLAMA_LOG_LEVEL:-error');
    expect(flake).toContain('GGML_LOG_LEVEL:-error');
    expect(flake).toContain('GGML_BACKEND_SILENT:-1');
    expect(flake).toContain('GGML_METAL_NO_RESIDENCY:-1');
  });
});

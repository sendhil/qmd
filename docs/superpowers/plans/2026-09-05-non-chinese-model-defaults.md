# Non-Chinese Model Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sendhil/qmd` work out of the box with Google/IBM model defaults while preserving automatic deep search, custom model overrides, `pi-memory` compatibility, and a low-friction upstream update workflow.

**Architecture:** Keep upstream QMD intact except for a focused model-policy layer. Move query-expansion prompt/grammar selection into a small profile module, use an IBM Granite profile by default, retain explicit Qwen compatibility, and give unknown custom generators a bounded generic profile. Add deterministic unit tests plus an opt-in Node-only real-model smoke harness, then document direct GitHub installation and upstream maintenance.

**Tech Stack:** TypeScript, Node.js 22+, Bun, Vitest, node-llama-cpp 3.20, GGUF models from Hugging Face, npm GitHub dependencies, Git.

---

## File Map

- Create `src/query-expansion-profile.ts`: model URI normalization, profile selection, prompts, grammars, and generation limits.
- Create `test/query-expansion-profile.test.ts`: deterministic profile and model-policy tests that never load a model.
- Modify `src/llm.ts`: approved default URIs, expanded chat-session type, profile use, parsed-output validation, deduplication, and safe fallback.
- Modify `test/llm.test.ts`: mock-based coverage of prompt wiring, bounded output, filtering, and fallback behavior.
- Create `scripts/smoke-non-chinese-models.ts`: opt-in Node-based checks against the real Granite expansion and reranking GGUFs.
- Create `test/fixtures/non-chinese-model-smoke.ts`: queries, anchors, expected documents, and distractors for the smoke harness.
- Modify `package.json`: expose the opt-in smoke command and include required maintenance documentation in GitHub/package builds where appropriate.
- Create `AGENTS.md`: concise fork invariants and automatic routing to the detailed maintenance runbook.
- Create `docs/UPSTREAM_MAINTENANCE.md`: exact upstream synchronization, validation, tagging, and rollback procedure.
- Modify `README.md`: fork purpose, installation, model downloads, `pi-memory`, standalone QMD, overrides, and provenance caveat.
- Modify `CHANGELOG.md`: record fork defaults and model-aware expansion behavior under Unreleased.

### Task 1: Lock the approved default-model policy with failing tests

**Files:**
- Modify: `test/llm.test.ts`

- [ ] **Step 1: Write tests for the exact approved defaults and denylist**

Add these imports to `test/llm.test.ts`:

```ts
import {
  DEFAULT_EMBED_MODEL_URI,
  DEFAULT_GENERATE_MODEL_URI,
  DEFAULT_RERANK_MODEL_URI,
} from "../src/llm.js";
```

Add this deterministic test block:

```ts
describe("fork default model policy", () => {
  const defaults = [
    DEFAULT_EMBED_MODEL_URI,
    DEFAULT_GENERATE_MODEL_URI,
    DEFAULT_RERANK_MODEL_URI,
  ];

  test("uses the approved Google and IBM model artifacts", () => {
    expect(DEFAULT_EMBED_MODEL_URI).toBe(
      "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
    );
    expect(DEFAULT_GENERATE_MODEL_URI).toBe(
      "hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/qmd-query-expansion-granite-2b-grpo-q4_k_m.gguf",
    );
    expect(DEFAULT_RERANK_MODEL_URI).toBe(
      "hf:keisuke-miyako/granite-embedding-reranker-english-r2-gguf-q8_0/granite-embedding-reranker-english-r2-Q8_0.gguf",
    );
  });

  test("does not default to a denied model lineage", () => {
    const denied = /qwen|deepseek|baai|\bbge\b|alibaba|\bgte\b|minicpm|chatglm|\bglm\b|\byi\b|internlm/i;
    for (const uri of defaults) expect(uri).not.toMatch(denied);
  });
});
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```sh
node ./node_modules/vitest/vitest.mjs run test/llm.test.ts --reporter=verbose
```

Expected: the two exact-URI assertions for generation and reranking fail because upstream defaults still point to Qwen-derived artifacts.

- [ ] **Step 3: Commit the failing policy tests**

```sh
git add test/llm.test.ts
git commit -m "test: lock non-Chinese default model policy"
```

### Task 2: Replace the built-in model defaults

**Files:**
- Modify: `src/llm.ts:281-295`
- Modify: `test/llm.test.ts:621-625`

- [ ] **Step 1: Change only the generation and reranking constants**

Replace the model constants in `src/llm.ts` with:

```ts
const DEFAULT_EMBED_MODEL =
  "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL =
  "hf:keisuke-miyako/granite-embedding-reranker-english-r2-gguf-q8_0/granite-embedding-reranker-english-r2-Q8_0.gguf";
const DEFAULT_GENERATE_MODEL =
  "hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/qmd-query-expansion-granite-2b-grpo-q4_k_m.gguf";
```

Delete the commented-out Qwen default beside these constants so an agent cannot accidentally restore it while resolving an upstream conflict. Do not remove explicit custom-model support elsewhere.

In the existing `LlamaCpp model resolution (config > env > default)` test block, replace the three duplicated `HARDCODED_*` strings with the exported constants:

```ts
const HARDCODED_EMBED = DEFAULT_EMBED_MODEL_URI;
const HARDCODED_RERANK = DEFAULT_RERANK_MODEL_URI;
const HARDCODED_GENERATE = DEFAULT_GENERATE_MODEL_URI;
```

This keeps the precedence test focused on precedence; the exact values remain locked by Task 1's separate policy assertions.

Replace the two default-reranker-specific `Qwen3` overhead comments with model-neutral wording while retaining the conservative 4096-token context and 512-token overhead values:

```ts
// Ranking templates add model-specific prompt tokens. The conservative 4096
// context keeps long real-world chunks below the model limit after truncation.
```

and:

```ts
// Reserve a conservative 512-token budget for model-specific ranking template
// overhead so a document cannot exceed the configured context after truncation.
```

Do not remove Qwen references that specifically document explicit embedding or generator compatibility.

- [ ] **Step 2: Run model resolution and CLI configuration tests**

Run:

```sh
node ./node_modules/vitest/vitest.mjs run test/llm.test.ts test/cli.test.ts --reporter=verbose
```

Expected: PASS. Existing configuration and environment-variable precedence tests continue to pass, and the new exact-default tests pass.

- [ ] **Step 3: Commit the default swap**

```sh
git add src/llm.ts test/llm.test.ts
git commit -m "feat: use Google and IBM model defaults"
```

### Task 3: Add query-expansion profiles using TDD

**Files:**
- Create: `src/query-expansion-profile.ts`
- Create: `test/query-expansion-profile.test.ts`

- [ ] **Step 1: Write failing profile-selection tests**

Create `test/query-expansion-profile.test.ts`:

```ts
import { describe, expect, test } from "vitest";
import {
  BOUNDED_EXPANSION_GRAMMAR,
  resolveQueryExpansionProfile,
} from "../src/query-expansion-profile.js";

describe("resolveQueryExpansionProfile", () => {
  test("selects Granite for a Hugging Face URI", () => {
    const profile = resolveQueryExpansionProfile(
      "hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/model.gguf",
    );
    expect(profile.id).toBe("granite");
    expect(profile.systemPrompt).toBe(
      "Expand a search query into typed lines (lex/vec/hyde) for a hybrid search engine.",
    );
    expect(profile.prompt("auth config")).toBe("Expand this search query: auth config");
    expect(profile.maxTokens).toBe(300);
  });

  test("selects Granite for a case-insensitive local filename", () => {
    expect(resolveQueryExpansionProfile(
      "/models/QMD-QUERY-EXPANSION-GRANITE-2B-Q4_K_M.gguf",
    ).id).toBe("granite");
  });

  test("retains Qwen behavior only for an explicit Qwen URI", () => {
    const profile = resolveQueryExpansionProfile("hf:example/Qwen3/custom.gguf");
    expect(profile.id).toBe("qwen");
    expect(profile.systemPrompt).toBeUndefined();
    expect(profile.prompt("auth config")).toBe("/no_think Expand this search query: auth config");
  });

  test("uses a bounded neutral profile for unknown generators", () => {
    const profile = resolveQueryExpansionProfile("/models/custom-model.gguf");
    expect(profile.id).toBe("generic");
    expect(profile.prompt("auth config")).toBe("Expand this search query: auth config");
    expect(profile.grammar).toBe(BOUNDED_EXPANSION_GRAMMAR);
    expect(profile.maxTokens).toBe(300);
    expect(profile.prompt("auth config")).not.toContain("/no_think");
  });

  test("the bounded grammar fixes one hyde, two lex, and three vec lines", () => {
    expect(BOUNDED_EXPANSION_GRAMMAR).toContain(
      "root ::= hyde-line lex-line lex-line vec-line vec-line vec-line",
    );
  });
});
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```sh
node ./node_modules/vitest/vitest.mjs run test/query-expansion-profile.test.ts --reporter=verbose
```

Expected: FAIL because `src/query-expansion-profile.ts` does not exist.

- [ ] **Step 3: Implement the profile module**

Create `src/query-expansion-profile.ts`:

```ts
export type QueryExpansionProfileId = "granite" | "qwen" | "generic";

export type QueryExpansionProfile = {
  id: QueryExpansionProfileId;
  systemPrompt?: string;
  prompt: (query: string) => string;
  grammar: string;
  maxTokens: number;
};

export const BOUNDED_EXPANSION_GRAMMAR = String.raw`
root ::= hyde-line lex-line lex-line vec-line vec-line vec-line
hyde-line ::= "hyde: " content "\n"
lex-line ::= "lex: " content "\n"
vec-line ::= "vec: " content "\n"
content ::= [^\n]+
`;

const LEGACY_QWEN_GRAMMAR = String.raw`
root ::= line+
line ::= type ": " content "\n"
type ::= "lex" | "vec" | "hyde"
content ::= [^\n]+
`;

const SYSTEM_PROMPT =
  "Expand a search query into typed lines (lex/vec/hyde) for a hybrid search engine.";

const neutralProfile = (id: "granite" | "generic"): QueryExpansionProfile => ({
  id,
  systemPrompt: SYSTEM_PROMPT,
  prompt: (query) => `Expand this search query: ${query}`,
  grammar: BOUNDED_EXPANSION_GRAMMAR,
  maxTokens: 300,
});

export function resolveQueryExpansionProfile(modelUri: string): QueryExpansionProfile {
  const normalized = modelUri.trim().toLowerCase();
  if (normalized.includes("qmd-query-expansion-granite-2b")) {
    return neutralProfile("granite");
  }
  if (normalized.includes("qwen")) {
    return {
      id: "qwen",
      prompt: (query) => `/no_think Expand this search query: ${query}`,
      grammar: LEGACY_QWEN_GRAMMAR,
      maxTokens: 600,
    };
  }
  return neutralProfile("generic");
}
```

- [ ] **Step 4: Run the profile tests under Node and Bun**

Run:

```sh
node ./node_modules/vitest/vitest.mjs run test/query-expansion-profile.test.ts --reporter=verbose
bun test --preload ./src/test-preload.ts test/query-expansion-profile.test.ts
```

Expected: both commands PASS without downloading any model.

- [ ] **Step 5: Commit the isolated profile layer**

```sh
git add src/query-expansion-profile.ts test/query-expansion-profile.test.ts
git commit -m "feat: add model-aware query expansion profiles"
```

### Task 4: Wire profiles into QMD and preserve safe fallback behavior

**Files:**
- Modify: `src/llm.ts:13-27`
- Modify: `src/llm.ts:1602-1699`
- Modify: `test/llm.test.ts`

- [ ] **Step 1: Write a mock-based failing test for Granite prompt wiring**

Add `resolveQueryExpansionProfile` coverage through `LlamaCpp.expandQuery()` in `test/llm.test.ts`. The mock must capture constructor and prompt arguments:

First add `type Queryable` and `type QueryExpansionDiagnostics` to the existing `../src/llm.js` import. Then define this helper beside the new tests:

```ts
async function runMockedExpansion(
  output: string | Error,
  onDiagnostics?: (diagnostics: QueryExpansionDiagnostics) => void,
): Promise<Queryable[]> {
  const sequence = { dispose: vi.fn(async () => {}) };
  const context = {
    getSequence: vi.fn(() => sequence),
    dispose: vi.fn(async () => {}),
  };
  const llama = { createGrammar: vi.fn(async ({ grammar }) => ({ grammar })) };

  setNodeLlamaCppModuleForTest({
    LlamaLogLevel: { error: "error" },
    resolveModelFile: vi.fn(),
    getLlama: vi.fn(),
    LlamaChatSession: class {
      async prompt() {
        if (output instanceof Error) throw output;
        return output;
      }
    } as any,
  });

  const llm = new LlamaCpp({ generateModel: DEFAULT_GENERATE_MODEL_URI }) as any;
  llm._ciMode = false;
  llm.touchActivity = vi.fn();
  llm.ensureLlama = vi.fn(async () => llama);
  llm.ensureGenerateModel = vi.fn(async () => {});
  llm.generateModel = {
    createContext: vi.fn(async () => context),
    dispose: vi.fn(async () => {}),
  };

  try {
    return await llm.expandQuery("auth setup", { onDiagnostics });
  } finally {
    await llm.dispose();
    setNodeLlamaCppModuleForTest(null);
  }
}
```

```ts
test("expandQuery applies the Granite system prompt and bounded generation", async () => {
  const sessionOptions: Array<Record<string, unknown>> = [];
  const promptCalls: Array<{ prompt: string; options: Record<string, unknown> }> = [];
  const sequence = { dispose: vi.fn(async () => {}) };
  const context = {
    getSequence: vi.fn(() => sequence),
    dispose: vi.fn(async () => {}),
  };
  const createGrammar = vi.fn(async ({ grammar }) => ({ grammar }));

  setNodeLlamaCppModuleForTest({
    LlamaLogLevel: { error: "error" },
    resolveModelFile: vi.fn(),
    getLlama: vi.fn(async () => ({ createGrammar }) as any),
    LlamaChatSession: class {
      constructor(options: Record<string, unknown>) { sessionOptions.push(options); }
      async prompt(prompt: string, options: Record<string, unknown>) {
        promptCalls.push({ prompt, options });
        return [
          "hyde: Authentication uses signed session tokens.",
          "lex: authentication session token",
          "lex: auth configuration",
          "vec: how authentication sessions are configured",
          "vec: where authentication tokens are validated",
          "vec: authentication login flow",
          "",
        ].join("\n");
      }
    } as any,
  });

  const llm = new LlamaCpp({ generateModel: DEFAULT_GENERATE_MODEL_URI }) as any;
  llm._ciMode = false;
  llm.touchActivity = vi.fn();
  llm.ensureLlama = vi.fn(async () => ({ createGrammar }));
  llm.ensureGenerateModel = vi.fn(async () => {});
  llm.generateModel = {
    createContext: vi.fn(async () => context),
    dispose: vi.fn(async () => {}),
  };

  try {
    const result = await llm.expandQuery("authentication setup");
    expect(sessionOptions[0]).toMatchObject({
      systemPrompt: "Expand a search query into typed lines (lex/vec/hyde) for a hybrid search engine.",
    });
    expect(promptCalls[0]?.prompt).toBe("Expand this search query: authentication setup");
    expect(promptCalls[0]?.options.maxTokens).toBe(300);
    expect(createGrammar.mock.calls[0]?.[0].grammar).toContain(
      "root ::= hyde-line lex-line lex-line vec-line vec-line vec-line",
    );
    expect(result.map((item: Queryable) => item.type)).toEqual([
      "hyde", "lex", "lex", "vec", "vec", "vec",
    ]);
  } finally {
    await llm.dispose();
    setNodeLlamaCppModuleForTest(null);
  }
});
```

- [ ] **Step 2: Write failing tests for deduplication, incomplete output, and generation failure**

Use the same mocked context helper to add these cases:

```ts
test("expandQuery deduplicates identical typed output", async () => {
  const result = await runMockedExpansion([
    "hyde: Information about auth setup",
    "lex: auth setup",
    "lex: auth setup",
    "vec: how to configure auth",
    "vec: how to configure auth",
    "vec: auth setup instructions",
  ].join("\n"));

  expect(result).toEqual([
    { type: "hyde", text: "Information about auth setup" },
    { type: "lex", text: "auth setup" },
    { type: "vec", text: "how to configure auth" },
    { type: "vec", text: "auth setup instructions" },
  ]);
});

test("expandQuery falls back when parsing loses an expansion type", async () => {
  let diagnostics: QueryExpansionDiagnostics | undefined;
  const result = await runMockedExpansion([
    "lex: auth setup",
    "lex: auth settings",
    "vec: auth configuration",
  ].join("\n"), (value) => { diagnostics = value; });

  expect(result).toEqual([
    { type: "hyde", text: "Information about auth setup" },
    { type: "lex", text: "auth setup" },
    { type: "vec", text: "auth setup" },
  ]);
  expect(diagnostics).toMatchObject({ usedFallback: true });
});

test("expandQuery reports generation failure and keeps the original query", async () => {
  let diagnostics: QueryExpansionDiagnostics | undefined;
  const result = await runMockedExpansion(
    new Error("generation failed"),
    (value) => { diagnostics = value; },
  );

  expect(result).toEqual([
    { type: "lex", text: "auth setup" },
    { type: "vec", text: "auth setup" },
  ]);
  expect(diagnostics).toEqual({ rawOutput: null, usedFallback: true });
});
```

- [ ] **Step 3: Run the focused tests and verify they fail**

Run:

```sh
node ./node_modules/vitest/vitest.mjs run test/llm.test.ts --reporter=verbose
```

Expected: prompt/system-profile assertions fail; duplicate output is retained; incomplete output does not use the three-entry fallback.

- [ ] **Step 4: Add typed expansion diagnostics and expand the local constructor type**

Define these exported types immediately after `Queryable` in `src/llm.ts`:

```ts
export type QueryExpansionDiagnostics = {
  rawOutput: string | null;
  usedFallback: boolean;
};

export type QueryExpansionOptions = {
  context?: string;
  includeLexical?: boolean;
  onDiagnostics?: (diagnostics: QueryExpansionDiagnostics) => void;
};
```

Replace all three inline expansion option types—in `ILLMSession`, `LLM`, and `LLMSession.expandQuery()`—plus the concrete `LlamaCpp.expandQuery()` signature with `QueryExpansionOptions`. This optional observer exists so the release smoke test can distinguish generated output from a fallback without changing ordinary callers or return values.

Change `NodeLlamaCppModule.LlamaChatSession` in `src/llm.ts` to:

```ts
LlamaChatSession: new (options: {
  contextSequence: unknown;
  systemPrompt?: string;
}) => {
  prompt: (prompt: string, options?: Record<string, unknown>) => Promise<string>;
};
```

- [ ] **Step 5: Resolve and apply the profile in `expandQuery()`**

Import the resolver:

```ts
import { resolveQueryExpansionProfile } from "./query-expansion-profile.js";
```

At the start of `expandQuery()`, after resolving `includeLexical`, add:

```ts
const profile = resolveQueryExpansionProfile(this.generateModelUri);
const prompt = profile.prompt(query);
const reportDiagnostics = (diagnostics: QueryExpansionDiagnostics): void => {
  try {
    options.onDiagnostics?.(diagnostics);
  } catch {
    // Diagnostics are observational and must never change search behavior.
  }
};
```

Replace the inline grammar string with:

```ts
const grammar = await llama.createGrammar({ grammar: profile.grammar });
```

Construct the session and prompt with:

```ts
const session = new LlamaChatSession({
  contextSequence: sequence,
  ...(profile.systemPrompt ? { systemPrompt: profile.systemPrompt } : {}),
});

const result = await session.prompt(prompt, {
  grammar,
  maxTokens: profile.maxTokens,
  temperature: 0.7,
  topK: 20,
  topP: 0.8,
  repeatPenalty: {
    lastTokens: 64,
    presencePenalty: 0.5,
  },
});
```

Replace the nearby Qwen-specific sampling comments in both `generate()` and `expandQuery()` with:

```ts
// Non-greedy sampling avoids repetition loops in supported generator profiles.
```

- [ ] **Step 6: Validate and deduplicate parsed expansions before filtering lexical output**

Replace the current mapping/filtering return block with exactly:

```ts
const seen = new Set<string>();
const queryables: Queryable[] = [];

for (const line of lines) {
  const colonIdx = line.indexOf(":");
  if (colonIdx === -1) continue;
  const type = line.slice(0, colonIdx).trim();
  if (type !== "lex" && type !== "vec" && type !== "hyde") continue;
  const text = line.slice(colonIdx + 1).trim();
  if (!text || !hasQueryTerm(text)) continue;
  const key = `${type}\u0000${text}`;
  if (seen.has(key)) continue;
  seen.add(key);
  queryables.push({ type, text });
}

const complete = (["lex", "vec", "hyde"] as const)
  .every((type) => queryables.some((item) => item.type === type));

if (complete) {
  const filtered = includeLexical
    ? queryables
    : queryables.filter((item) => item.type !== "lex");
  if (filtered.length > 0) {
    reportDiagnostics({ rawOutput: result, usedFallback: false });
    return filtered;
  }
}

reportDiagnostics({ rawOutput: result, usedFallback: true });
const fallback: Queryable[] = [
  { type: "hyde", text: `Information about ${query}` },
  { type: "lex", text: query },
  { type: "vec", text: query },
];
return includeLexical ? fallback : fallback.filter((item) => item.type !== "lex");
```

In the exception handler, call this before constructing the existing lexical/vector fallback:

```ts
reportDiagnostics({ rawOutput: null, usedFallback: true });
```

Keep that exception fallback as lexical plus vector original-query entries. Do not resolve another model in either fallback.

- [ ] **Step 7: Run focused tests under Node and Bun**

Run:

```sh
node ./node_modules/vitest/vitest.mjs run test/query-expansion-profile.test.ts test/llm.test.ts --reporter=verbose
bun test --preload ./src/test-preload.ts test/query-expansion-profile.test.ts test/llm.test.ts
```

Expected: PASS without any model downloads.

- [ ] **Step 8: Run type checking**

Run:

```sh
npm run test:types
```

Expected: PASS; the expanded `LlamaChatSession` constructor type accepts the optional system prompt.

- [ ] **Step 9: Commit runtime integration**

```sh
git add src/llm.ts test/llm.test.ts
git commit -m "feat: apply model-specific expansion profiles"
```

### Task 5: Add opt-in real-model smoke tests

**Files:**
- Create: `test/fixtures/non-chinese-model-smoke.ts`
- Create: `scripts/smoke-non-chinese-models.ts`
- Modify: `package.json`
- Modify: `test/package.test.ts`

- [ ] **Step 1: Write the failing package-script assertion**

Add to the `package test task` block in `test/package.test.ts`:

```ts
test("exposes an opt-in non-Chinese model smoke test", () => {
  expect(pkg.scripts["smoke:non-chinese-models"]).toBe(
    "tsx scripts/smoke-non-chinese-models.ts",
  );
});
```

- [ ] **Step 2: Run the assertion and verify it fails**

Run:

```sh
node ./node_modules/vitest/vitest.mjs run test/package.test.ts --reporter=verbose
```

Expected: FAIL because `smoke:non-chinese-models` is absent.

- [ ] **Step 3: Create the typed fixture file**

Create `test/fixtures/non-chinese-model-smoke.ts` exporting seven expansion cases and seven rerank cases. Give the arrays explicit element types so optional `context` is safe to access:

```ts
export type ExpansionSmokeCase = {
  query: string;
  context?: string;
  anchors: readonly string[];
};

export type RerankSmokeCase = {
  query: string;
  expected: string;
  distractors: readonly [string, string, string, string, ...string[]];
};

export const expansionCases: readonly ExpansionSmokeCase[] = [
  { query: "which database did we choose", anchors: ["database"] },
  { query: "package manager preference", anchors: ["package", "manager"] },
  { query: "production deploy region", anchors: ["production", "deploy", "region"] },
  { query: "how to run unit tests", anchors: ["unit", "test"] },
  { query: "dark mode preference", anchors: ["dark", "mode"] },
  { query: "where is resolveGenerateModel called", anchors: ["resolvegeneratemodel"] },
  {
    query: "performance",
    context: "web page loading and Core Web Vitals, not employee reviews",
    anchors: ["performance"],
  },
] as const;

export const rerankCases: readonly RerankSmokeCase[] = [
  {
    query: "which database did we choose",
    expected: "The architecture decision selected PostgreSQL as the primary application database.",
    distractors: [
      "The database migration runs during deployment.",
      "Redis stores short-lived cache entries.",
      "SQLite is used only by local test fixtures.",
      "The team chose Vitest for unit testing.",
    ],
  },
  {
    query: "package manager preference",
    expected: "Use Bun as the preferred package manager for installing dependencies in this repository.",
    distractors: [
      "The package publishes from the main branch.",
      "npm hosts the generated release archive.",
      "The manager approves production deployments.",
      "Dependencies are pinned in the lockfile.",
    ],
  },
  {
    query: "production deploy region",
    expected: "Production deploys to AWS us-west-2, the Oregon region.",
    distractors: [
      "Staging deploys after every merge.",
      "Regional pricing is reviewed quarterly.",
      "Production logs are retained for thirty days.",
      "The local server listens on port 3000.",
    ],
  },
  {
    query: "how to run unit tests",
    expected: "Run the unit test suite with npm test from the repository root.",
    distractors: [
      "Integration tests use temporary databases.",
      "The unit price is displayed in dollars.",
      "Run the development server with npm run dev.",
      "Test fixtures live under the test directory.",
    ],
  },
  {
    query: "dark mode preference",
    expected: "The user prefers dark mode and wants it enabled by default.",
    distractors: [
      "The display supports several color modes.",
      "Preference data is stored locally.",
      "Light mode uses a white page background.",
      "The CLI has a verbose output mode.",
    ],
  },
  {
    query: "where is resolveGenerateModel called",
    expected: "The LlamaCpp constructor calls resolveGenerateModel in src/llm.ts to select the generator URI.",
    distractors: [
      "resolveEmbedModel selects the embedding artifact.",
      "The model cache directory is created on demand.",
      "Generation contexts are disposed after expansion.",
      "The CLI resolves collection paths before searching.",
    ],
  },
  {
    query: "performance",
    expected: "Web performance is measured with Core Web Vitals such as LCP, INP, and CLS.",
    distractors: [
      "Employee performance reviews evaluate goals and collaboration.",
      "Database performance improves after adding an index.",
      "Athletic performance depends on recovery and training.",
      "The quarterly performance begins at eight o'clock.",
    ],
  },
] as const;
```

- [ ] **Step 4: Implement the Node-only smoke harness**

Create `scripts/smoke-non-chinese-models.ts` with these gates:

```ts
import { performance } from "node:perf_hooks";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore } from "../src/index.js";
import {
  DEFAULT_EMBED_MODEL_URI,
  DEFAULT_GENERATE_MODEL_URI,
  DEFAULT_RERANK_MODEL_URI,
  LlamaCpp,
  type QueryExpansionDiagnostics,
} from "../src/llm.js";
import { expansionCases, rerankCases } from "../test/fixtures/non-chinese-model-smoke.js";

const DENIED = /qwen|deepseek|baai|\bbge\b|alibaba|\bgte\b|minicpm|chatglm|\bglm\b|\byi\b|internlm/i;

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function withTimeout<T>(operation: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      operation,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out`)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function main(): Promise<void> {
  assert(!DENIED.test(DEFAULT_EMBED_MODEL_URI), "embedder violates model policy");
  assert(!DENIED.test(DEFAULT_GENERATE_MODEL_URI), "generator violates model policy");
  assert(!DENIED.test(DEFAULT_RERANK_MODEL_URI), "reranker violates model policy");

  const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
  try {
    await llm.expandQuery("warm up query expansion");
    let expansionPasses = 0;

    for (const fixture of expansionCases) {
      const started = performance.now();
      let diagnostics: QueryExpansionDiagnostics | undefined;
      const output = await withTimeout(
        llm.expandQuery(fixture.query, {
          context: fixture.context,
          onDiagnostics: (value) => { diagnostics = value; },
        }),
        30_000,
        `expansion: ${fixture.query}`,
      );
      const elapsedMs = performance.now() - started;
      const rawLines = diagnostics?.rawOutput?.trimEnd().split("\n") ?? [];
      const expectedTypes = ["hyde", "lex", "lex", "vec", "vec", "vec"];
      const rawShapeValid = rawLines.length === expectedTypes.length
        && rawLines.every((line, index) =>
          line.startsWith(`${expectedTypes[index]}: `) && /^(hyde|lex|vec): .+$/.test(line),
        )
        && new Set(rawLines).size === rawLines.length;
      const counts = new Map(output.map((item) => [
        item.type,
        output.filter((other) => other.type === item.type).length,
      ]));
      const keys = output.map((item) => `${item.type}\u0000${item.text}`);
      const anchorFound = output
        .filter((item) => item.type === "lex" || item.type === "vec")
        .some((item) => fixture.anchors.some((anchor) =>
          item.text.toLowerCase().includes(anchor.toLowerCase()),
        ));

      const passed = elapsedMs <= 30_000
        && diagnostics?.usedFallback === false
        && rawShapeValid
        && output.length >= 3 && output.length <= 6
        && (counts.get("hyde") ?? 0) >= 1
        && (counts.get("lex") ?? 0) >= 1
        && (counts.get("vec") ?? 0) >= 1
        && new Set(keys).size === keys.length
        && output.filter((item) => item.type === "lex").every((item) => item.text.length <= 120)
        && output.filter((item) => item.type === "vec").every((item) => item.text.length <= 240)
        && output.filter((item) => item.type === "hyde").every((item) => item.text.length <= 400)
        && anchorFound;
      if (passed) expansionPasses++;
      console.log(JSON.stringify({
        kind: "expansion",
        query: fixture.query,
        elapsedMs,
        passed,
        diagnostics,
        output,
      }));
    }

    assert(expansionPasses >= 6, `only ${expansionPasses}/7 expansion fixtures passed`);

    let rerankTopOnes = 0;
    for (const fixture of rerankCases) {
      const documents = [fixture.expected, ...fixture.distractors]
        .map((text, index) => ({ file: index === 0 ? "expected.md" : `distractor-${index}.md`, text }));
      const { results } = await llm.rerank(fixture.query, documents);
      const rank = results.findIndex((item) => item.file === "expected.md") + 1;
      const expectedResult = results.find((item) => item.file === "expected.md");
      const bestDistractor = results.find((item) => item.file !== "expected.md");
      const scoreMargin = (expectedResult?.score ?? Number.NaN)
        - (bestDistractor?.score ?? Number.NaN);
      assert(results.every((item) => Number.isFinite(item.score) && item.score >= 0 && item.score <= 1),
        `invalid reranker score: ${fixture.query}`);
      assert(rank > 0 && rank <= 2, `expected document ranked ${rank}: ${fixture.query}`);
      if (rank === 1) rerankTopOnes++;
      console.log(JSON.stringify({ kind: "rerank", query: fixture.query, rank, scoreMargin, results }));
    }

    assert(rerankTopOnes >= 6, `only ${rerankTopOnes}/7 rerank fixtures ranked expected first`);
  } finally {
    await llm.dispose();
  }

  // Use a fresh store only after releasing the direct-test model contexts, so
  // the smoke check does not keep two complete model sets resident at once.
  const root = await mkdtemp(join(tmpdir(), "qmd-non-chinese-smoke-"));
  const docs = join(root, "docs");
  await mkdir(docs);
  const documents = {
    "web-performance.md": "# Web performance\nCore Web Vitals measure page loading. Improve LCP by optimizing server response and image delivery.\n",
    "employee-performance.md": "# Employee performance\nQuarterly reviews evaluate goals, collaboration, and career growth.\n",
    "database-performance.md": "# Database performance\nA covering index reduced SQL query latency during the benchmark.\n",
    "sports-performance.md": "# Athletic performance\nRecovery, sleep, and progressive training improve race results.\n",
    "music-performance.md": "# Music performance\nThe concert performance begins at eight in the main hall.\n",
  } as const;
  for (const [name, body] of Object.entries(documents)) {
    await writeFile(join(docs, name), body);
  }

  let store: Awaited<ReturnType<typeof createStore>> | undefined;
  try {
    store = await createStore({
      dbPath: join(root, "index.sqlite"),
      config: {
        collections: { smoke: { path: docs, pattern: "**/*.md" } },
        models: {
          embed: DEFAULT_EMBED_MODEL_URI,
          generate: DEFAULT_GENERATE_MODEL_URI,
          rerank: DEFAULT_RERANK_MODEL_URI,
        },
      },
    });
    await store.update();
    await store.embed();
    const results = await withTimeout(
      store.search({
        query: "performance",
        intent: "web page loading and Core Web Vitals, not employee reviews",
        collection: "smoke",
        limit: 5,
      }),
      60_000,
      "intent-aware deep search",
    );
    const rank = results.findIndex((item) => item.file.endsWith("web-performance.md")) + 1;
    assert(rank > 0 && rank <= 2, `intent-aware deep search ranked expected document ${rank}`);
    console.log(JSON.stringify({ kind: "deep-search", query: "performance", rank, results }));
  } finally {
    if (store) await store.close();
    await rm(root, { recursive: true, force: true });
  }
}

await main();
```

- [ ] **Step 5: Add the opt-in package script**

Add to `package.json` scripts:

```json
"smoke:non-chinese-models": "tsx scripts/smoke-non-chinese-models.ts"
```

This command must not be included in `npm test` because it downloads approximately 1.7 GB and depends on local inference hardware.

- [ ] **Step 6: Run deterministic package tests**

Run:

```sh
node ./node_modules/vitest/vitest.mjs run test/package.test.ts --reporter=verbose
npm run test:types
```

Expected: PASS without downloading models.

- [ ] **Step 7: Run the real-model smoke harness under Node**

Run:

```sh
GGML_METAL_NO_RESIDENCY=1 npm run smoke:non-chinese-models
```

Expected: at least 6/7 expansion fixtures pass; all expected rerank documents rank in the top two and at least 6/7 rank first; process exits 0. Do not substitute Bun for this release gate.

- [ ] **Step 8: Commit the smoke harness**

```sh
git add package.json test/package.test.ts test/fixtures/non-chinese-model-smoke.ts scripts/smoke-non-chinese-models.ts
git commit -m "test: add opt-in Granite model smoke checks"
```

### Task 6: Add durable agent and upstream-maintenance instructions

**Files:**
- Create: `AGENTS.md`
- Create: `docs/UPSTREAM_MAINTENANCE.md`
- Modify: `.gitignore`

- [ ] **Step 1: Allowlist the two required Markdown files**

Add to `.gitignore` after the existing Markdown exceptions:

```gitignore
!AGENTS.md
!docs/UPSTREAM_MAINTENANCE.md
```

Keep the broad upstream `*.md` ignore rule unchanged.

- [ ] **Step 2: Create the root agent contract**

Create `AGENTS.md`:

```md
# Fork Maintenance Instructions

This is `sendhil/qmd`, a thin personal fork of `tobi/qmd`.

## Non-negotiable invariants

- `main` mirrors `upstream/main`; fork changes belong on `non-chinese-defaults`.
- Default model URIs must comply with the policy in `docs/UPSTREAM_MAINTENANCE.md`.
- Never restore a Qwen, DeepSeek, BGE/BAAI, GTE/Alibaba, MiniCPM, GLM, Yi, or InternLM default while resolving conflicts.
- Explicit user model overrides remain supported.
- Never silently fall back to or download a different model family.
- Preserve QMD's full deep pipeline and `pi-memory` compatibility.
- Do not run networked model tests until their explicit URIs have been audited.

Before syncing, releasing, or changing a model, read and follow
`docs/UPSTREAM_MAINTENANCE.md` completely.
```

- [ ] **Step 3: Create the detailed maintenance runbook**

Create `docs/UPSTREAM_MAINTENANCE.md` with these exact sections and executable commands:

````md
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
models. Never run it until its resolved URIs pass the audit.

## Tag and install

Use a tag composed from the current upstream version and fork revision. For the initial release this is:

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
````

- [ ] **Step 4: Verify the instructions are tracked and discoverable**

Run:

```sh
git check-ignore -v AGENTS.md docs/UPSTREAM_MAINTENANCE.md
git status --short
```

Expected: `git check-ignore` reports the negated allowlist rules, and `git status` lists both files as untracked rather than hiding them.

- [ ] **Step 5: Commit maintenance instructions**

```sh
git add .gitignore AGENTS.md docs/UPSTREAM_MAINTENANCE.md
git commit -m "docs: add upstream maintenance runbook"
```

### Task 7: Document installation, Pi memory, and standalone QMD use

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a fork notice and direct-install instructions near README Quick Start**

Add a concise fork section before `## Quick Start` containing:

````md
## sendhil/qmd fork

This branch preserves QMD's complete local deep-search pipeline while using
Google and IBM models by default. It does not download Qwen or another
Chinese-developed model unless you explicitly override a model URI.

Install an immutable fork release:

```sh
npm install -g github:sendhil/qmd#non-chinese-v2.8.3.1
```

The first semantic/deep operation downloads EmbeddingGemma, the IBM
Granite-based QMD query expander (about 1.55 GB), and IBM Granite R2 reranker
(about 161 MB). Models stay in QMD's normal local cache.

The query-expansion checkpoint uses an IBM base model, but part of its label
dataset has incomplete generator provenance. See the model-policy discussion
in `docs/UPSTREAM_MAINTENANCE.md` before adopting it under a stricter policy.
````

- [ ] **Step 2: Add Pi-memory setup**

Add:

````md
### Pi memory

Install the fork first so `pi-memory` discovers this `qmd` executable:

```sh
npm install -g github:sendhil/qmd#non-chinese-v2.8.3.1
pi install npm:pi-memory
```

`pi-memory` creates its `pi-memory` collection automatically. If QMD was
installed after Pi started, restart Pi or initialize manually:

```sh
qmd collection add ~/.pi/agent/memory --name pi-memory
qmd context add /daily "Daily append-only work logs organized by date" -c pi-memory
qmd context add / "Curated long-term memory: decisions, preferences, facts, lessons" -c pi-memory
qmd embed
```

In Pi, run `memory_status`, then test `memory_search` using `deep` mode.
Remove or disable another memory extension that registers the same tool name.
````

- [ ] **Step 3: Replace every other current-default reference in README**

Make these exact replacements outside the Model Configuration section:

```diff
-                          │  (qwen3-reranker)     │
+                          │ (granite-reranker)    │
```

```diff
-1. **Query Expansion**: Original query (×2 for weighting) + 1 LLM variation
+1. **Query Expansion**: Original query (×2 for weighting) + typed LLM variations
```

Replace the three-row GGUF model table with:

```md
| Model | Purpose | Size |
|-------|---------|------|
| `embeddinggemma-300M-Q8_0` | Vector embeddings (default) | ~300 MB |
| `granite-embedding-reranker-english-r2-Q8_0` | Re-ranking | ~161 MB |
| `qmd-query-expansion-granite-2b-grpo-q4_k_m` | Query expansion (fine-tuned) | ~1.55 GB |
```

In the sample `models:` configuration, preserve the embed line and replace the other two lines with:

```yaml
  rerank: "hf:keisuke-miyako/granite-embedding-reranker-english-r2-gguf-q8_0/granite-embedding-reranker-english-r2-Q8_0.gguf"
  generate: "hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/qmd-query-expansion-granite-2b-grpo-q4_k_m.gguf"
```

Keep the later Qwen embedding example because it is explicitly labeled as an opt-in custom override, not a default.

- [ ] **Step 4: Update model documentation for standalone QMD**

Replace the current Model Configuration section, from its opening paragraph through the query-expansion subsection, with:

````md
## Model Configuration

The default models are defined in `src/llm.ts` as Hugging Face URIs:

```typescript
const DEFAULT_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL = "hf:keisuke-miyako/granite-embedding-reranker-english-r2-gguf-q8_0/granite-embedding-reranker-english-r2-Q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/qmd-query-expansion-granite-2b-grpo-q4_k_m.gguf";
```

Override any role without changing source through the `models:` block in
`index.yml` (see [Configuring `index.yml`](#configuring-indexyml)) or with
`QMD_EMBED_MODEL`, `QMD_GENERATE_MODEL`, and `QMD_RERANK_MODEL`. Explicit
configuration keeps precedence over fork defaults. Re-run `qmd embed` after
changing the embedding model.

All standalone collection, CLI, SDK, and MCP workflows elsewhere in this
README behave as documented; only the built-in model choices and expansion
profile selection differ in this fork.

### EmbeddingGemma prompt format

```
// For queries
"task: search result | query: {query}"

// For documents
"title: {title} | text: {content}"
```

### IBM Granite R2 reranker

Uses node-llama-cpp's `createRankingContext()` and `rankAndSort()` APIs for
cross-encoder reranking. It returns documents sorted by a relevance score from
0.0 through 1.0.

### Query expansion profiles

The default Granite QMD expander receives a neutral system prompt and a bounded
grammar that emits one `hyde:`, two `lex:`, and three `vec:` lines. Unknown
custom generator URIs receive the same bounded generic profile. An explicitly
configured URI containing `qwen` receives QMD's legacy Qwen compatibility
prompt and grammar; QMD never selects or downloads it as a silent fallback.
````

- [ ] **Step 5: Add the changelog entry**

Under `## [Unreleased]`, add:

```md
### Changed

- Fork defaults now use Google's EmbeddingGemma plus IBM Granite query
  expansion and reranking models, avoiding Chinese-developed model weights on
  a fresh installation. Query expansion selects a model-aware prompt and
  grammar profile while preserving explicit custom-model overrides.

### Added

- Added an opt-in Node-based real-model smoke test and an agent-facing
  upstream-maintenance runbook for the `non-chinese-defaults` branch.
```

Merge with an existing `Changed` or `Added` subsection rather than creating duplicate headings.

- [ ] **Step 6: Check documentation for stale default claims**

Run:

```sh
rg -n "DEFAULT_(RERANK|GENERATE)_MODEL|Qwen3-Reranker|Qwen3 \(Query Expansion\)|qmd-query-expansion-1.7B" README.md CHANGELOG.md src/llm.ts
```

Expected: Qwen appears only in clearly labeled explicit-override or historical changelog contexts. No current default table or code constant references a Qwen artifact.

- [ ] **Step 7: Commit user documentation**

```sh
git add README.md CHANGELOG.md
git commit -m "docs: explain fork installation and Pi memory setup"
```

### Task 8: Run full verification and prepare the first immutable release

**Files:**
- Verify: all changed files

- [ ] **Step 1: Install dependencies without rewriting the lock state**

Run:

```sh
bun install --frozen-lockfile
```

Expected: dependencies install successfully and `git status --short` shows no dependency-file changes.

- [ ] **Step 2: Run the complete upstream test pipeline**

Run:

```sh
npm test
```

Expected: type checking, Node Vitest, Bun tests, and package smoke all PASS.

- [ ] **Step 3: Run lint and diff hygiene**

Run:

```sh
npm run lint
git diff --check main...HEAD
```

Expected: both commands exit 0.

- [ ] **Step 4: Run the approved real-model release gate**

Run:

```sh
GGML_METAL_NO_RESIDENCY=1 npm run smoke:non-chinese-models
```

Expected: expansion and reranking thresholds from Task 5 pass under Node and the command exits 0.

- [ ] **Step 5: Verify the packed GitHub-install payload**

Run:

```sh
npm pack --dry-run
```

Expected: output includes `bin/qmd`, compiled `dist/`, runtime scripts, and skills; it does not include cached `.gguf` files, SQLite indexes, or test fixtures.

- [ ] **Step 6: Test a recoverable local global installation**

Run:

```sh
npm pack
npm install -g ./tobilu-qmd-2.8.3.tgz
qmd --version
qmd doctor
```

Expected: the `qmd` command runs under Node and reports the fork's upstream package version. Record the existing global QMD source before replacement so it can be reinstalled if this check fails. Do not delete model caches or indexes.

- [ ] **Step 7: Verify Pi-memory integration manually**

With `pi-memory` installed and any conflicting memory extension disabled, start a new Pi session and run:

```text
memory_status
memory_search(query: "a known memory phrase", mode: "deep")
```

Expected: status finds the `pi-memory` QMD collection and embeddings; deep search completes through expansion and reranking without downloading a denied model. Inspect QMD's model cache or `qmd status` to confirm only approved defaults were resolved.

- [ ] **Step 8: Review the branch delta before publication**

Run:

```sh
git log --oneline main..HEAD
git diff --stat main...HEAD
git status --short --branch
```

Expected: only the focused policy/profile, tests, smoke harness, and documentation commits appear; the worktree is clean.

- [ ] **Step 9: Push the branch after user approval**

```sh
git push -u origin non-chinese-defaults
```

Expected: GitHub updates `sendhil/qmd/non-chinese-defaults`. Pushing is an external mutation and must occur only after the user approves the verified branch.

- [ ] **Step 10: Create and push the first immutable tag after user approval**

```sh
git tag -a non-chinese-v2.8.3.1 -m "QMD 2.8.3 with approved model defaults"
git push origin non-chinese-v2.8.3.1
```

Expected: the immutable tag is available for `npm install -g github:sendhil/qmd#non-chinese-v2.8.3.1`.

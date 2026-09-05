import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import {
  getLlama,
  getLlamaGpuTypes,
  LlamaChatSession as RealLlamaChatSession,
  LlamaLogLevel,
  resolveModelFile,
  type LLamaChatPromptOptions,
  type LlamaContextSequence,
  type LlamaOptions,
} from "node-llama-cpp";
import { createStore } from "../src/index.js";
import {
  DEFAULT_EMBED_MODEL_URI,
  DEFAULT_GENERATE_MODEL_URI,
  DEFAULT_RERANK_MODEL_URI,
  LlamaCpp,
  setNodeLlamaCppModuleForTest,
} from "../src/llm.js";
import {
  expansionCases,
  rerankCases,
} from "../test/fixtures/non-chinese-model-smoke.js";

const DENIED = /qwen|deepseek|baai|\bbge\b|alibaba|\bgte\b|minicpm|chatglm|\bglm\b|\byi\b|internlm/i;
let capturedRaw: string | null = null;
let activeExpansionSignal: AbortSignal | undefined;

class CapturingLlamaChatSession {
  private readonly inner: RealLlamaChatSession;

  constructor(options: { contextSequence: unknown; systemPrompt?: string }) {
    this.inner = new RealLlamaChatSession({
      contextSequence: options.contextSequence as LlamaContextSequence,
      ...(options.systemPrompt ? { systemPrompt: options.systemPrompt } : {}),
    });
  }

  async prompt(prompt: string, options?: Record<string, unknown>): Promise<string> {
    const raw = await this.inner.prompt(
      prompt,
      {
        ...options,
        ...(activeExpansionSignal ? { signal: activeExpansionSignal } : {}),
      } as LLamaChatPromptOptions,
    );
    capturedRaw = raw;
    return raw;
  }
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function expandWithDeadline(
  llm: LlamaCpp,
  query: string,
  context: string | undefined,
  timeoutMs: number,
  label: string,
): Promise<Awaited<ReturnType<LlamaCpp["expandQuery"]>>> {
  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort(new Error(`${label} timed out`));
  }, timeoutMs);
  activeExpansionSignal = controller.signal;
  try {
    return await llm.expandQuery(query, { context });
  } finally {
    clearTimeout(timer);
    activeExpansionSignal = undefined;
  }
}

async function main(): Promise<void> {
  const gateFailures: string[] = [];
  assert(!DENIED.test(DEFAULT_EMBED_MODEL_URI), "embedder violates model policy");
  assert(!DENIED.test(DEFAULT_GENERATE_MODEL_URI), "generator violates model policy");
  assert(!DENIED.test(DEFAULT_RERANK_MODEL_URI), "reranker violates model policy");

  setNodeLlamaCppModuleForTest({
    getLlama: (options) => getLlama(options as LlamaOptions),
    getLlamaGpuTypes,
    resolveModelFile,
    LlamaChatSession: CapturingLlamaChatSession,
    LlamaLogLevel,
  });

  const llm = new LlamaCpp({
    embedModel: DEFAULT_EMBED_MODEL_URI,
    generateModel: DEFAULT_GENERATE_MODEL_URI,
    rerankModel: DEFAULT_RERANK_MODEL_URI,
    inactivityTimeoutMs: 0,
  });
  try {
    await expandWithDeadline(
      llm,
      "warm up query expansion",
      undefined,
      30_000,
      "expansion warmup",
    );
    let expansionPasses = 0;

    for (const fixture of expansionCases) {
      const started = performance.now();
      capturedRaw = null;
      const output = await expandWithDeadline(
        llm,
        fixture.query,
        fixture.context,
        30_000,
        `expansion: ${fixture.query}`,
      );
      const elapsedMs = performance.now() - started;
      const rawLines = capturedRaw?.trimEnd().split("\n") ?? [];
      const expectedTypes = ["hyde", "lex", "lex", "vec", "vec", "vec"];
      const rawExpansionValid = rawLines.length === expectedTypes.length
        && rawLines.every((line, index) =>
          line.startsWith(`${expectedTypes[index]}: `) && /^(hyde|lex|vec): .+$/.test(line),
        )
        && new Set(rawLines).size === rawLines.length;
      const counts = new Map(output.map((item) => [
        item.type,
        output.filter((other) => other.type === item.type).length,
      ]));
      const keys = output.map((item) => `${item.type}\u0000${item.text}`);
      const successfulFallback = JSON.stringify([
        { type: "hyde", text: `Information about ${fixture.query}` },
        { type: "lex", text: fixture.query },
        { type: "vec", text: fixture.query },
      ]);
      const exceptionFallback = JSON.stringify([
        { type: "lex", text: fixture.query },
        { type: "vec", text: fixture.query },
      ]);
      const serializedOutput = JSON.stringify(output);
      const usedFallback = serializedOutput === successfulFallback
        || serializedOutput === exceptionFallback;
      const anchorFound = output
        .filter((item) => item.type === "lex" || item.type === "vec")
        .some((item) => fixture.anchors.some((anchor) =>
          item.text.toLowerCase().includes(anchor.toLowerCase()),
        ));

      const passed = elapsedMs <= 30_000
        && capturedRaw !== null
        && !usedFallback
        && rawExpansionValid
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
        rawOutput: capturedRaw,
        usedFallback,
        output,
      }));
    }

    if (expansionPasses < 6) {
      gateFailures.push(`only ${expansionPasses}/7 expansion fixtures passed`);
    }

    let rerankTopOnes = 0;
    const rerankFailures: string[] = [];
    for (const [fixtureIndex, fixture] of rerankCases.entries()) {
      const documents = fixture.distractors.map((text, index) => ({
        file: `distractor-${index + 1}.md`,
        text,
      }));
      const expectedPosition = fixtureIndex % (documents.length + 1);
      documents.splice(expectedPosition, 0, {
        file: "expected.md",
        text: fixture.expected,
      });
      const { results, model } = await llm.rerank(fixture.query, documents);
      const rank = results.findIndex((item) => item.file === "expected.md") + 1;
      const expectedResult = results.find((item) => item.file === "expected.md");
      const bestDistractor = results.find((item) => item.file !== "expected.md");
      const scoreMargin = (expectedResult?.score ?? Number.NaN)
        - (bestDistractor?.score ?? Number.NaN);
      const scoresValid = results.every((item) =>
        Number.isFinite(item.score) && item.score >= 0 && item.score <= 1,
      );
      if (model !== DEFAULT_RERANK_MODEL_URI) {
        rerankFailures.push(`unexpected reranker model ${model}: ${fixture.query}`);
      }
      if (!scoresValid) rerankFailures.push(`invalid reranker score: ${fixture.query}`);
      if (rank <= 0 || rank > 2) {
        rerankFailures.push(`expected document ranked ${rank}: ${fixture.query}`);
      }
      if (rank === 1) rerankTopOnes++;
      console.log(JSON.stringify({
        kind: "rerank",
        query: fixture.query,
        expectedInputPosition: expectedPosition + 1,
        rank,
        scoreMargin,
        model,
        results,
      }));
    }

    gateFailures.push(...rerankFailures);
    if (rerankTopOnes < 6) {
      gateFailures.push(`only ${rerankTopOnes}/7 rerank fixtures ranked expected first`);
    }
  } finally {
    await llm.dispose();
    setNodeLlamaCppModuleForTest(null);
  }

  let root: string | undefined;
  let store: Awaited<ReturnType<typeof createStore>> | undefined;
  try {
    root = await mkdtemp(join(tmpdir(), "qmd-non-chinese-smoke-"));
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
    const searchStarted = performance.now();
    const results = await store.search({
      query: "performance",
      intent: "web page loading and Core Web Vitals, not employee reviews",
      collection: "smoke",
      limit: 5,
    });
    const elapsedMs = performance.now() - searchStarted;
    const rank = results.findIndex((item) => item.file.endsWith("web-performance.md")) + 1;
    if (elapsedMs > 60_000) {
      gateFailures.push(`intent-aware deep search took ${elapsedMs}ms`);
    }
    if (rank <= 0 || rank > 2) {
      gateFailures.push(`intent-aware deep search ranked expected document ${rank}`);
    }
    console.log(JSON.stringify({
      kind: "deep-search",
      query: "performance",
      elapsedMs,
      rank,
      results,
    }));
  } finally {
    try {
      if (store) await store.close();
    } finally {
      if (root) await rm(root, { recursive: true, force: true });
    }
  }

  assert(gateFailures.length === 0, gateFailures.join("; "));
}

await main();

import { describe, expect, test } from "vitest";
import {
  BUILTIN_MODEL_MANIFEST,
  DEFAULT_EMBED_MODEL_URI,
  DEFAULT_GENERATE_MODEL_URI,
  DEFAULT_RERANK_MODEL_URI,
  canonicalizeBuiltinModelUri,
  resolveGenerateModel,
  resolveModels,
} from "../src/llm.js";
import { gatedModels, type BuiltinModels } from "../src/trust.js";

describe("fork default model policy", () => {
  const defaults = [
    DEFAULT_EMBED_MODEL_URI,
    DEFAULT_GENERATE_MODEL_URI,
    DEFAULT_RERANK_MODEL_URI,
  ];

  test("uses immutable approved Google, IBM, and Jina model artifacts", () => {
    expect(DEFAULT_EMBED_MODEL_URI).toBe(
      "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf#0f741b5a6585bd53aeb15cd1372c56f2a0f65e12",
    );
    expect(DEFAULT_GENERATE_MODEL_URI).toBe(
      "hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/qmd-query-expansion-granite-2b-grpo-q4_k_m.gguf#449c09bced7605802af16c2b431fd40e5e871b8c",
    );
    expect(DEFAULT_RERANK_MODEL_URI).toBe(
      "hf:ggml-org/jina-reranker-v1-turbo-en-GGUF/Jina-Bert-Implementation-38M-F16.gguf#8582fa8560bcdd3c5cbc9015514edff0f3b1871f",
    );
  });

  test("records a full revision, size, and SHA-256 for every built-in artifact", () => {
    expect(BUILTIN_MODEL_MANIFEST).toEqual({
      embed: {
        role: "embed",
        logicalUri: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
        uri: DEFAULT_EMBED_MODEL_URI,
        sizeBytes: 333590944,
        sha256: "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63",
      },
      generate: {
        role: "generate",
        logicalUri: "hf:nichenke/qmd-query-expansion-granite-2b-grpo-gguf/qmd-query-expansion-granite-2b-grpo-q4_k_m.gguf",
        uri: DEFAULT_GENERATE_MODEL_URI,
        sizeBytes: 1545302752,
        sha256: "a488061ddcf8ee3e18912cd29cb33cbe6396084946de75544650ac5620e5b6ed",
      },
      rerank: {
        role: "rerank",
        logicalUri: "hf:ggml-org/jina-reranker-v1-turbo-en-GGUF/Jina-Bert-Implementation-38M-F16.gguf",
        uri: DEFAULT_RERANK_MODEL_URI,
        sizeBytes: 76971168,
        sha256: "71abc010bb3dce97812ee971509a5cb6ff6f6b8cfffd8480129242f605521fca",
      },
    });

    for (const spec of Object.values(BUILTIN_MODEL_MANIFEST)) {
      expect(spec.uri).toMatch(/#[0-9a-f]{40}$/);
      expect(spec.sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(spec.sizeBytes).toBeGreaterThan(0);
    }
  });

  test("canonicalizes historical floating defaults before the trust gate", () => {
    const builtins: BuiltinModels = {
      embed: DEFAULT_EMBED_MODEL_URI,
      generate: DEFAULT_GENERATE_MODEL_URI,
      rerank: DEFAULT_RERANK_MODEL_URI,
    };

    for (const [slot, spec] of Object.entries(BUILTIN_MODEL_MANIFEST)) {
      expect(canonicalizeBuiltinModelUri(spec.logicalUri, spec.role)).toBe(spec.uri);
      expect(gatedModels({ [slot]: spec.logicalUri }, builtins)).toEqual([]);
    }

    expect(resolveModels({
      embed: BUILTIN_MODEL_MANIFEST.embed.logicalUri,
      generate: BUILTIN_MODEL_MANIFEST.generate.logicalUri,
      rerank: BUILTIN_MODEL_MANIFEST.rerank.logicalUri,
    })).toEqual(builtins);
  });

  test("keeps a legacy URI used as a different role's explicit override custom", () => {
    const builtins: BuiltinModels = {
      embed: DEFAULT_EMBED_MODEL_URI,
      generate: DEFAULT_GENERATE_MODEL_URI,
      rerank: DEFAULT_RERANK_MODEL_URI,
    };
    const crossRoleOverride = BUILTIN_MODEL_MANIFEST.embed.logicalUri;

    expect(resolveGenerateModel({ generate: crossRoleOverride })).toBe(crossRoleOverride);
    expect(gatedModels({ generate: crossRoleOverride }, builtins)).toEqual([
      { slot: "generate", uri: crossRoleOverride },
    ]);
  });

  test("does not default to a denied model lineage", () => {
    const denied = /qwen|deepseek|baai|\bbge\b|alibaba|\bgte\b|minicpm|chatglm|\bglm\b|\byi\b|internlm/i;
    for (const uri of defaults) expect(uri).not.toMatch(denied);
  });
});

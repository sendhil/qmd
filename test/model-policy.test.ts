import { describe, expect, test } from "vitest";
import {
  DEFAULT_EMBED_MODEL_URI,
  DEFAULT_GENERATE_MODEL_URI,
  DEFAULT_RERANK_MODEL_URI,
} from "../src/llm.js";

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

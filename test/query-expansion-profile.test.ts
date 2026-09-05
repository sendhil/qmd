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
    expect(profile.maxTokens).toBe(600);
    expect(profile.grammar).toContain("root ::= line+");
    expect(profile.grammar).not.toBe(BOUNDED_EXPANSION_GRAMMAR);
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

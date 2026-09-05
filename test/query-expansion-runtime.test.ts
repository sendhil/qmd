import { describe, expect, test, vi } from "vitest";
import {
  DEFAULT_GENERATE_MODEL_URI,
  LlamaCpp,
  setNodeLlamaCppModuleForTest,
  type Queryable,
} from "../src/llm.js";

async function runMockedExpansion(
  output: string | Error,
  options: {
    includeLexical?: boolean;
    generateModel?: string;
    ensureGenerateModelError?: Error;
    resolveModelFile?: ReturnType<typeof vi.fn>;
  } = {},
): Promise<Queryable[]> {
  const sequence = { dispose: vi.fn(async () => {}) };
  const context = {
    getSequence: vi.fn(() => sequence),
    dispose: vi.fn(async () => {}),
  };
  const llama = { createGrammar: vi.fn(async ({ grammar }) => ({ grammar })) };

  setNodeLlamaCppModuleForTest({
    LlamaLogLevel: { error: "error" },
    resolveModelFile: options.resolveModelFile ?? vi.fn(),
    getLlama: vi.fn(),
    LlamaChatSession: class {
      async prompt() {
        if (output instanceof Error) throw output;
        return output;
      }
    } as any,
  });

  const llm = new LlamaCpp({
    generateModel: options.generateModel ?? DEFAULT_GENERATE_MODEL_URI,
  }) as any;
  llm._ciMode = false;
  llm.touchActivity = vi.fn();
  llm.ensureLlama = vi.fn(async () => llama);
  llm.ensureGenerateModel = vi.fn(async () => {
    if (options.ensureGenerateModelError) throw options.ensureGenerateModelError;
  });
  llm.generateModel = {
    createContext: vi.fn(async () => context),
    dispose: vi.fn(async () => {}),
  };

  try {
    return await llm.expandQuery("auth setup", {
      includeLexical: options.includeLexical,
    });
  } finally {
    await llm.dispose();
    setNodeLlamaCppModuleForTest(null);
  }
}

describe("query expansion runtime profiles", () => {
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
            "lex: authentication configuration",
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
    const result = await runMockedExpansion([
      "lex: auth setup",
      "lex: auth settings",
      "vec: auth configuration",
    ].join("\n"));

    expect(result).toEqual([
      { type: "hyde", text: "Information about auth setup" },
      { type: "lex", text: "auth setup" },
      { type: "vec", text: "auth setup" },
    ]);
  });

  test("expandQuery omits lexical output from an incomplete-output fallback", async () => {
    const result = await runMockedExpansion([
      "lex: auth setup",
      "vec: auth configuration",
    ].join("\n"), { includeLexical: false });

    expect(result).toEqual([
      { type: "hyde", text: "Information about auth setup" },
      { type: "vec", text: "auth setup" },
    ]);
  });

  test("expandQuery falls back when generator model initialization fails", async () => {
    const resolveModelFile = vi.fn();
    const result = await runMockedExpansion("unused", {
      ensureGenerateModelError: new Error("model initialization failed"),
      resolveModelFile,
    });

    expect(result).toEqual([
      { type: "lex", text: "auth setup" },
      { type: "vec", text: "auth setup" },
    ]);
    expect(resolveModelFile).not.toHaveBeenCalled();
  });

  test("expandQuery rejects more than six unique valid lines", async () => {
    const result = await runMockedExpansion([
      "hyde: Information about auth setup",
      "lex: auth setup",
      "lex: auth settings",
      "vec: how to configure auth",
      "vec: auth setup instructions",
      "vec: auth login flow",
      "vec: auth token handling",
    ].join("\n"), {
      generateModel: "hf:example/Qwen3/custom.gguf",
    });

    expect(result).toEqual([
      { type: "hyde", text: "Information about auth setup" },
      { type: "lex", text: "auth setup" },
      { type: "vec", text: "auth setup" },
    ]);
  });

  test("expandQuery keeps the original query after generation failure", async () => {
    const result = await runMockedExpansion(new Error("generation failed"));

    expect(result).toEqual([
      { type: "lex", text: "auth setup" },
      { type: "vec", text: "auth setup" },
    ]);
  });

  test("expandQuery omits lexical output after generation failure", async () => {
    const result = await runMockedExpansion(new Error("generation failed"), {
      includeLexical: false,
    });

    expect(result).toEqual([
      { type: "vec", text: "auth setup" },
    ]);
  });
});

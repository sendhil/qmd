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

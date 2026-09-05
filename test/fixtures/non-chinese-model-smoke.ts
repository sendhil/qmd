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

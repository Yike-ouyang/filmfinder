import type {
  Decision,
  DecisionKind,
  Recommendation,
  RecommendationBatch,
} from "@/types/recommendations";

export const DECISION_KINDS: readonly DecisionKind[] = [
  "interested",
  "passed",
  "seen",
];

export function recommendationKey(recommendation: Recommendation): string {
  return String(recommendation.internal_movie_id);
}

export function countDecisions(decisions: Decision[]): Record<DecisionKind, number> {
  return decisions.reduce<Record<DecisionKind, number>>(
    (counts, decision) => {
      counts[decision.kind] += 1;
      return counts;
    },
    { interested: 0, passed: 0, seen: 0 },
  );
}

export function currentRecommendation(
  batch: RecommendationBatch,
): Recommendation | null {
  return batch.response.recommendations[batch.decisions.length] ?? null;
}

export function decisionLabel(kind: DecisionKind): string {
  switch (kind) {
    case "interested":
      return "Interested";
    case "passed":
      return "Passed";
    case "seen":
      return "Already seen";
  }
}

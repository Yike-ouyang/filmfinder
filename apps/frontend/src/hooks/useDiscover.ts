"use client";

import { useCallback, useEffect, useMemo, useReducer } from "react";
import { countDecisions, recommendationKey } from "@/lib/recommendations";
import { updateDecisions } from "@/lib/storage";
import type {
  Decision,
  DecisionKind,
  RecommendationBatch,
} from "@/types/recommendations";

export type ExitDirection = "left" | "right" | "down";

interface DiscoverState {
  batch: RecommendationBatch;
}

type DiscoverAction =
  | { type: "decide"; decision: Decision }
  | { type: "undo" };

function discoverReducer(
  state: DiscoverState,
  action: DiscoverAction,
): DiscoverState {
  switch (action.type) {
    case "decide":
      return {
        batch: {
          ...state.batch,
          decisions: [...state.batch.decisions, action.decision],
        },
      };
    case "undo":
      if (state.batch.decisions.length === 0) {
        return state;
      }
      return {
        batch: {
          ...state.batch,
          decisions: state.batch.decisions.slice(0, -1),
        },
      };
    default:
      return state;
  }
}

function exitDirectionFor(kind: DecisionKind): ExitDirection {
  if (kind === "interested") {
    return "right";
  }
  if (kind === "passed") {
    return "left";
  }
  return "down";
}

export function useDiscover(initialBatch: RecommendationBatch) {
  const [state, dispatch] = useReducer(discoverReducer, {
    batch: initialBatch,
  });
  const { batch } = state;
  const recommendations = batch.response.recommendations;
  const decisions = batch.decisions;
  const index = decisions.length;
  const current = recommendations[index] ?? null;
  const completed = current === null;
  const lastRecommendation = recommendations[recommendations.length - 1] ?? null;
  const stats = useMemo(() => countDecisions(decisions), [decisions]);
  const exitDirection = useMemo<ExitDirection>(() => {
    const last = decisions[decisions.length - 1];
    return last ? exitDirectionFor(last.kind) : "left";
  }, [decisions]);

  useEffect(() => {
    updateDecisions(batch, batch.decisions);
  }, [batch]);

  const decide = useCallback(
    (kind: DecisionKind) => {
      if (!current) {
        return;
      }
      dispatch({
        type: "decide",
        decision: { key: recommendationKey(current), kind },
      });
    },
    [current],
  );

  const undo = useCallback(() => {
    if (decisions.length === 0) {
      return;
    }
    dispatch({ type: "undo" });
  }, [decisions.length]);

  return {
    batch,
    current,
    decisions,
    index,
    total: recommendations.length,
    completed,
    lastRecommendation,
    stats,
    exitDirection,
    decide,
    undo,
  };
}

import { useCallback, useState } from 'react';
import {
  bulkApprove as apiBulkApprove,
  getResults,
  getStreamUrl,
  startReview as apiStartReview,
  updateEvaluation as apiUpdateEvaluation,
} from '../api/client';
import { mockEvaluations } from '../mocks/mockData';
import type { Evaluation, ReviewPhase, UpdateEvaluationRequest } from '../types';

const USE_MOCK = import.meta.env.VITE_MOCK === 'true';

export function useReview(sessionId: string) {
  const [evaluations, setEvaluations] = useState<Map<number, Evaluation>>(
    new Map(),
  );
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [phase, setPhase] = useState<ReviewPhase>('idle');
  const [error, setError] = useState<string | null>(null);

  const refreshEvaluations = useCallback(async () => {
    try {
      const session = await getResults(sessionId);
      setEvaluations(
        new Map(session.evaluations.map((e) => [e.requirement_id, e])),
      );
    } catch (e) {
      console.error('Failed to refresh evaluations', e);
    }
  }, [sessionId]);

  const startMockStream = useCallback(
    (total: number) => {
      let idx = 0;
      const interval = setInterval(() => {
        if (idx >= mockEvaluations.length) {
          clearInterval(interval);
          // Simulate critic pass
          setTimeout(() => {
            setPhase('critic');
            setTimeout(() => {
              setPhase('complete');
            }, 1000);
          }, 500);
          return;
        }
        const evaluation = mockEvaluations[idx];
        setEvaluations((prev) =>
          new Map(prev).set(evaluation.requirement_id, evaluation),
        );
        setProgress({ completed: idx + 1, total });
        idx++;
      }, 600);
    },
    [],
  );

  const startReview = useCallback(async () => {
    setError(null);
    if (USE_MOCK) {
      setPhase('reviewing');
      startMockStream(mockEvaluations.length);
      return;
    }

    try {
      await apiStartReview(sessionId);
      setPhase('reviewing');

      const eventSource = new EventSource(getStreamUrl(sessionId));

      eventSource.addEventListener('evaluation', (e: MessageEvent) => {
        const data = JSON.parse(e.data) as {
          evaluation: Evaluation;
          progress: number;
          total: number;
        };
        setEvaluations((prev) =>
          new Map(prev).set(
            data.evaluation.requirement_id,
            data.evaluation,
          ),
        );
        setProgress({ completed: data.progress, total: data.total });
      });

      eventSource.addEventListener('critic_complete', () => {
        setPhase('critic');
        void refreshEvaluations();
      });

      eventSource.addEventListener('done', () => {
        setPhase('complete');
        eventSource.close();
      });

      eventSource.onerror = () => {
        setError('Connection to review stream lost.');
        eventSource.close();
      };
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start review');
    }
  }, [sessionId, startMockStream, refreshEvaluations]);

  const updateEvaluation = useCallback(
    async (requirementId: number, updates: UpdateEvaluationRequest) => {
      if (USE_MOCK) {
        setEvaluations((prev) => {
          const next = new Map(prev);
          const existing = next.get(requirementId);
          if (existing) next.set(requirementId, { ...existing, ...updates });
          return next;
        });
        return;
      }
      try {
        const updated = await apiUpdateEvaluation(sessionId, requirementId, updates);
        setEvaluations((prev) => new Map(prev).set(requirementId, updated));
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to update evaluation');
      }
    },
    [sessionId],
  );

  const bulkApprove = useCallback(
    async (ids: number[]) => {
      if (!USE_MOCK) {
        try {
          await apiBulkApprove(sessionId, ids);
        } catch (e) {
          setError(e instanceof Error ? e.message : 'Bulk approve failed');
          return;
        }
      }
      setEvaluations((prev) => {
        const next = new Map(prev);
        ids.forEach((id) => {
          const e = next.get(id);
          if (e) next.set(id, { ...e, status: 'approved' });
        });
        return next;
      });
    },
    [sessionId],
  );

  return {
    evaluations,
    progress,
    phase,
    error,
    startReview,
    updateEvaluation,
    bulkApprove,
  };
}

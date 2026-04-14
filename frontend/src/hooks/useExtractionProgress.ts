import { useEffect, useState } from 'react';
import { getExtractionStreamUrl } from '../api/client';
import type { ExtractionStep, Requirement } from '../types';

/**
 * Subscribes to the extraction SSE stream for a long-document session.
 *
 * - Yields `currentStep` updates as `extraction_progress` events arrive.
 * - Sets `requirements` once `extraction_complete` is received, then closes
 *   the EventSource.
 * - Sets `error` and closes the EventSource on connection failure.
 *
 * Pass `sessionId = null` to keep the hook dormant (no connection opened).
 */
export function useExtractionProgress(sessionId: string | null) {
  const [currentStep, setCurrentStep] = useState<ExtractionStep | null>(null);
  const [requirements, setRequirements] = useState<Requirement[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    const es = new EventSource(getExtractionStreamUrl(sessionId));

    es.addEventListener('extraction_progress', (e: MessageEvent) => {
      setCurrentStep(JSON.parse(e.data) as ExtractionStep);
    });

    es.addEventListener('extraction_complete', (e: MessageEvent) => {
      const data = JSON.parse(e.data) as {
        requirements: Requirement[];
        total_requirements: number;
      };
      setRequirements(data.requirements);
      es.close();
    });

    es.onerror = () => {
      setError('Connection lost during extraction. Refresh to check status.');
      es.close();
    };

    return () => es.close();
  }, [sessionId]);

  return { currentStep, requirements, error };
}

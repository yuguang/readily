import type {
  Evaluation,
  ReviewSession,
  UpdateEvaluationRequest,
  UploadResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form });
  return handleResponse<UploadResponse>(res);
}

export async function startReview(sessionId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/review/${sessionId}/start`, {
    method: 'POST',
  });
  await handleResponse<unknown>(res);
}

export async function getResults(sessionId: string): Promise<ReviewSession> {
  const res = await fetch(`${API_BASE}/review/${sessionId}/results`);
  return handleResponse<ReviewSession>(res);
}

export async function updateEvaluation(
  sessionId: string,
  reqId: number,
  updates: UpdateEvaluationRequest,
): Promise<Evaluation> {
  const res = await fetch(
    `${API_BASE}/review/${sessionId}/results/${reqId}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    },
  );
  return handleResponse<Evaluation>(res);
}

export async function bulkApprove(
  sessionId: string,
  ids: number[],
): Promise<void> {
  const res = await fetch(`${API_BASE}/review/${sessionId}/bulk-approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ requirement_ids: ids }),
  });
  await handleResponse<unknown>(res);
}

export async function exportCsv(sessionId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/review/${sessionId}/export`);
  if (!res.ok) throw new Error(`Export failed: ${res.statusText}`);
  return res.blob();
}

export function getStreamUrl(sessionId: string): string {
  return `${API_BASE}/review/${sessionId}/stream`;
}

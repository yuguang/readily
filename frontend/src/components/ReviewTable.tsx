import { useState } from 'react';
import { exportCsv } from '../api/client';
import { EvidenceCard } from './EvidenceCard';
import { useReview } from '../hooks/useReview';
import type { Evaluation, Requirement, UploadResponse } from '../types';

const USE_MOCK = import.meta.env.VITE_MOCK === 'true';

interface Props {
  uploadResponse: UploadResponse;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function confidenceClass(evaluation: Evaluation): {
  row: string;
  badge: string;
} {
  if (evaluation.needs_human_review || evaluation.confidence < 0.5) {
    return {
      row: 'bg-red-50 hover:bg-red-100',
      badge: 'bg-red-100 text-red-700',
    };
  }
  if (evaluation.confidence < 0.8) {
    return {
      row: 'bg-yellow-50 hover:bg-yellow-100',
      badge: 'bg-yellow-100 text-yellow-700',
    };
  }
  return {
    row: 'bg-green-50 hover:bg-green-100',
    badge: 'bg-green-100 text-green-700',
  };
}

const ANSWER_LABELS: Record<string, string> = {
  yes: '✅ Yes',
  no: '❌ No',
  partial: '⚠️ Partial',
};

const STATUS_PILL: Record<string, string> = {
  pending: 'bg-gray-100 text-gray-500',
  approved: 'bg-green-100 text-green-700',
  edited: 'bg-blue-100 text-blue-700',
  rejected: 'bg-red-100 text-red-700',
};

// ── Sub-components ────────────────────────────────────────────────────────────

function Spinner() {
  return (
    <svg
      className="animate-spin h-4 w-4 text-gray-400"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

function PendingRow({ req }: { req: Requirement }) {
  return (
    <tr className="border-b border-gray-100">
      <td className="px-4 py-3 text-sm text-gray-400">{req.id}</td>
      <td className="px-4 py-3 text-sm text-gray-500 max-w-xs">
        <span className="line-clamp-2">{req.text}</span>
      </td>
      <td className="px-4 py-3" colSpan={5}>
        <div className="flex items-center gap-2 text-gray-400 text-sm">
          <Spinner />
          Pending…
        </div>
      </td>
    </tr>
  );
}

interface ResultRowProps {
  req: Requirement;
  evaluation: Evaluation;
  onExpand: () => void;
  onApprove: () => void;
  onEdit: () => void;
  onReject: () => void;
}

function ResultRow({ req, evaluation, onExpand, onApprove }: ResultRowProps) {
  const { row, badge } = confidenceClass(evaluation);

  return (
    <tr
      className={`border-b border-gray-100 cursor-pointer transition-colors ${row}`}
      onClick={onExpand}
    >
      <td className="px-4 py-3 text-sm font-medium text-gray-600">{req.id}</td>
      <td className="px-4 py-3 text-sm text-gray-800 max-w-xs">
        <span className="line-clamp-2">{req.text}</span>
      </td>
      <td className="px-4 py-3 text-sm font-medium text-gray-800">
        {ANSWER_LABELS[evaluation.answer]}
      </td>
      <td className="px-4 py-3">
        <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold ${badge}`}>
          {(evaluation.confidence * 100).toFixed(0)}%
        </span>
      </td>
      <td className="px-4 py-3 text-sm text-gray-500 max-w-[200px]">
        <span className="line-clamp-2 italic text-xs">{evaluation.citation_text}</span>
      </td>
      <td className="px-4 py-3">
        <span
          className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium capitalize ${
            STATUS_PILL[evaluation.status]
          }`}
        >
          {evaluation.status}
        </span>
      </td>
      <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
        <div className="flex gap-1">
          <button
            onClick={onApprove}
            disabled={evaluation.status === 'approved'}
            title="Approve"
            className="p-1 rounded text-green-600 hover:bg-green-100 disabled:opacity-30 transition-colors"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </button>
          <button
            onClick={onExpand}
            title="Expand"
            className="p-1 rounded text-gray-500 hover:bg-gray-100 transition-colors"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </button>
        </div>
      </td>
    </tr>
  );
}

// ── Phase banner ──────────────────────────────────────────────────────────────

function PhaseBanner({ phase, docType }: { phase: string; docType: string }) {
  if (phase === 'idle' || phase === 'complete') return null;
  const isNarrative = docType === 'narrative';
  const messages: Record<string, string> = {
    reviewing: isNarrative
      ? 'AI agents are reviewing extracted obligations in parallel…'
      : 'AI agents are reviewing requirements in parallel…',
    critic: 'Critic pass in progress — reviewing flagged evaluations…',
  };
  return (
    <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg text-blue-700 text-sm">
      <Spinner />
      {messages[phase] ?? ''}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function ReviewTable({ uploadResponse }: Props) {
  const { session_id, requirements, doc_type } = uploadResponse;
  const {
    evaluations,
    progress,
    phase,
    error,
    startReview,
    updateEvaluation,
    bulkApprove,
  } = useReview(session_id);

  const [expandedId, setExpandedId] = useState<number | null>(null);

  // Start review automatically when the component first mounts
  const [started, setStarted] = useState(false);
  if (!started) {
    setStarted(true);
    void startReview();
  }

  // Bulk approve: all green (≥0.8 confidence, no human review flag) rows not yet approved
  const highConfidenceIds = requirements
    .filter((req) => {
      const ev = evaluations.get(req.id);
      return ev && ev.confidence >= 0.8 && !ev.needs_human_review && ev.status !== 'approved';
    })
    .map((r) => r.id);

  const handleBulkApprove = () => void bulkApprove(highConfidenceIds);

  const handleExportCsv = async () => {
    if (USE_MOCK) {
      // Build CSV client-side from current state
      const rows = requirements.map((req) => {
        const ev = evaluations.get(req.id);
        return [
          req.id,
          `"${req.text.replace(/"/g, '""')}"`,
          ev?.answer ?? '',
          ev ? (ev.confidence * 100).toFixed(0) + '%' : '',
          ev ? `"${ev.citation_text.replace(/"/g, '""')}"` : '',
          ev?.source_file ?? '',
          ev?.page_number ?? '',
          ev?.status ?? 'pending',
          ev ? `"${ev.reviewer_notes.replace(/"/g, '""')}"` : '',
        ].join(',');
      });
      const header = '#,Requirement,Answer,Confidence,Citation,Source,Page,Status,Notes';
      const csv = [header, ...rows].join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      triggerDownload(blob, `${uploadResponse.filename.replace('.pdf', '')}_review.csv`);
      return;
    }
    try {
      const blob = await exportCsv(session_id);
      triggerDownload(blob, `review_${session_id}.csv`);
    } catch (e) {
      console.error('Export failed', e);
    }
  };

  const progressPct =
    progress.total > 0 ? Math.round((progress.completed / progress.total) * 100) : 0;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top bar */}
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-gray-900">Readily</h1>
          <p className="text-xs text-gray-400 truncate max-w-sm">{uploadResponse.filename}</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500">
            {phase === 'complete'
              ? `${evaluations.size} of ${requirements.length} complete`
              : `${progress.completed} of ${progress.total > 0 ? progress.total : requirements.length} complete`}
          </span>
          <button
            onClick={handleBulkApprove}
            disabled={highConfidenceIds.length === 0}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:opacity-40 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Approve All High-Confidence ({highConfidenceIds.length})
          </button>
          <button
            onClick={() => void handleExportCsv()}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export CSV
          </button>
        </div>
      </header>

      <div className="px-6 py-4 space-y-3">
        {/* Progress bar */}
        {phase !== 'idle' && (
          <div>
            <div className="flex justify-between text-xs text-gray-400 mb-1">
              <span>
                {phase === 'complete' ? 'Review complete' : `${progressPct}% — ${progress.completed} of ${progress.total > 0 ? progress.total : requirements.length} evaluated`}
              </span>
              {phase === 'complete' && (
                <span className="text-green-600 font-medium">✓ Done</span>
              )}
            </div>
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  phase === 'complete' ? 'bg-green-500' : 'bg-blue-500'
                }`}
                style={{ width: `${phase === 'complete' ? 100 : progressPct}%` }}
              />
            </div>
          </div>
        )}

        {/* Phase banner */}
        <PhaseBanner phase={phase} docType={doc_type} />

        {/* Error */}
        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            {error}
          </div>
        )}

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs text-gray-400">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-green-400 inline-block" />
            High confidence (≥80%)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-yellow-400 inline-block" />
            Medium (50–80%)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-red-400 inline-block" />
            Needs attention (&lt;50% or flagged)
          </span>
        </div>

        {/* Table */}
        <div className="bg-white rounded-xl border border-gray-200 overflow-x-auto shadow-sm">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50">
                <th className="px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide w-10">#</th>
                <th className="px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">Requirement</th>
                <th className="px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">Answer</th>
                <th className="px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">Confidence</th>
                <th className="px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">Evidence</th>
                <th className="px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">Status</th>
                <th className="px-4 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide w-20">Actions</th>
              </tr>
            </thead>
            <tbody>
              {requirements.map((req) => {
                const ev = evaluations.get(req.id);
                if (!ev) {
                  return <PendingRow key={req.id} req={req} />;
                }
                return (
                  <ResultRow
                    key={req.id}
                    req={req}
                    evaluation={ev}
                    onExpand={() => setExpandedId(req.id)}
                    onApprove={() =>
                      void updateEvaluation(req.id, { status: 'approved' })
                    }
                    onEdit={() => setExpandedId(req.id)}
                    onReject={() => setExpandedId(req.id)}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* EvidenceCard modal */}
      {expandedId !== null && (() => {
        const req = requirements.find((r) => r.id === expandedId);
        const ev = evaluations.get(expandedId);
        if (!req || !ev) return null;
        return (
          <EvidenceCard
            requirement={req}
            evaluation={ev}
            onClose={() => setExpandedId(null)}
            onUpdate={(updates) => void updateEvaluation(expandedId, updates)}
          />
        );
      })()}
    </div>
  );
}

// ── Utils ─────────────────────────────────────────────────────────────────────

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

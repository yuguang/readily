import { useState } from 'react';
import type { AnswerType, Evaluation, Requirement } from '../types';

interface Props {
  requirement: Requirement;
  evaluation: Evaluation;
  onClose: () => void;
  onUpdate: (updates: Partial<Evaluation>) => void;
}

const ANSWER_LABELS: Record<AnswerType, string> = {
  yes: '✅ Yes',
  no: '❌ No',
  partial: '⚠️ Partial',
};

const STATUS_COLORS: Record<string, string> = {
  pending: 'text-gray-500',
  approved: 'text-green-600',
  edited: 'text-blue-600',
  rejected: 'text-red-600',
};

export function EvidenceCard({ requirement, evaluation, onClose, onUpdate }: Props) {
  const [editing, setEditing] = useState(false);
  const [editAnswer, setEditAnswer] = useState<AnswerType>(evaluation.answer);
  const [editCitation, setEditCitation] = useState(evaluation.citation_text);
  const [editNotes, setEditNotes] = useState(evaluation.reviewer_notes);
  const [rejectNote, setRejectNote] = useState('');
  const [showRejectInput, setShowRejectInput] = useState(false);

  const handleApprove = () => {
    onUpdate({ status: 'approved' });
    onClose();
  };

  const handleSaveEdit = () => {
    onUpdate({
      answer: editAnswer,
      citation_text: editCitation,
      reviewer_notes: editNotes,
      status: 'edited',
    });
    setEditing(false);
  };

  const handleReject = () => {
    if (!rejectNote.trim()) return;
    onUpdate({ status: 'rejected', reviewer_notes: rejectNote });
    setShowRejectInput(false);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-start justify-between p-5 border-b border-gray-100">
          <div>
            <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">
              Requirement #{requirement.id}
              {requirement.reference && ` · ${requirement.reference}`}
            </p>
            <p className="mt-1 text-gray-800 text-sm leading-relaxed font-medium">
              {requirement.text}
            </p>
          </div>
          <button
            onClick={onClose}
            className="ml-4 flex-shrink-0 p-1 rounded-full text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-5 space-y-4">
          {/* Answer + Confidence row */}
          <div className="flex items-center gap-6">
            <div>
              <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Answer</p>
              {editing ? (
                <select
                  value={editAnswer}
                  onChange={(e) => setEditAnswer(e.target.value as AnswerType)}
                  className="border border-gray-300 rounded-md px-2 py-1 text-sm text-gray-800 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                >
                  <option value="yes">✅ Yes</option>
                  <option value="no">❌ No</option>
                  <option value="partial">⚠️ Partial</option>
                </select>
              ) : (
                <p className="text-gray-900 font-semibold">{ANSWER_LABELS[evaluation.answer]}</p>
              )}
            </div>
            <div>
              <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Confidence</p>
              <p className="text-gray-900 font-semibold">
                {(evaluation.confidence * 100).toFixed(0)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Status</p>
              <p className={`font-semibold text-sm capitalize ${STATUS_COLORS[evaluation.status]}`}>
                {evaluation.status}
              </p>
            </div>
            {evaluation.needs_human_review && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-100 text-red-700 text-xs font-medium">
                ⚠ Needs Review
              </span>
            )}
          </div>

          {/* Citation */}
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-1.5">Citation</p>
            {editing ? (
              <textarea
                value={editCitation}
                onChange={(e) => setEditCitation(e.target.value)}
                rows={4}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-800 font-mono leading-relaxed focus:ring-2 focus:ring-blue-500 focus:outline-none resize-none"
              />
            ) : (
              <blockquote className="bg-gray-50 border border-gray-200 rounded-lg px-4 py-3 text-sm text-gray-700 italic leading-relaxed font-serif">
                {evaluation.citation_text}
              </blockquote>
            )}
          </div>

          {/* Source */}
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Source</p>
            <p className="text-sm text-gray-700">
              <span className="font-medium">{evaluation.source_file}</span>
              {', Page '}
              {evaluation.page_number}
            </p>
          </div>

          {/* Reasoning */}
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Reasoning</p>
            <p className="text-sm text-gray-700 leading-relaxed">{evaluation.reasoning}</p>
          </div>

          {/* Reviewer Notes */}
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wide mb-1.5">Reviewer Notes</p>
            {editing ? (
              <textarea
                value={editNotes}
                onChange={(e) => setEditNotes(e.target.value)}
                rows={3}
                placeholder="Add notes…"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-800 focus:ring-2 focus:ring-blue-500 focus:outline-none resize-none"
              />
            ) : (
              <p className="text-sm text-gray-500 italic min-h-[2rem]">
                {evaluation.reviewer_notes || 'No notes yet.'}
              </p>
            )}
          </div>

          {/* Reject input */}
          {showRejectInput && (
            <div className="rounded-lg bg-red-50 border border-red-200 p-3 space-y-2">
              <p className="text-sm text-red-700 font-medium">Rejection note (required)</p>
              <textarea
                value={rejectNote}
                onChange={(e) => setRejectNote(e.target.value)}
                rows={2}
                placeholder="Explain why this is being rejected…"
                className="w-full border border-red-300 rounded px-2 py-1.5 text-sm focus:ring-2 focus:ring-red-400 focus:outline-none resize-none"
              />
              <div className="flex gap-2">
                <button
                  onClick={handleReject}
                  disabled={!rejectNote.trim()}
                  className="px-3 py-1.5 bg-red-600 hover:bg-red-700 disabled:opacity-40 text-white text-sm font-medium rounded-md transition-colors"
                >
                  Confirm Reject
                </button>
                <button
                  onClick={() => setShowRejectInput(false)}
                  className="px-3 py-1.5 border border-gray-300 text-gray-600 text-sm rounded-md hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-end gap-2 px-5 py-4 border-t border-gray-100 bg-gray-50 rounded-b-xl">
          {editing ? (
            <>
              <button
                onClick={() => setEditing(false)}
                className="px-4 py-2 border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveEdit}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
              >
                Save Changes
              </button>
            </>
          ) : (
            <>
              <button
                onClick={handleApprove}
                disabled={evaluation.status === 'approved'}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:opacity-40 text-white text-sm font-semibold rounded-lg transition-colors"
              >
                Approve
              </button>
              <button
                onClick={() => setEditing(true)}
                className="px-4 py-2 border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-white transition-colors"
              >
                Edit
              </button>
              <button
                onClick={() => setShowRejectInput(true)}
                disabled={evaluation.status === 'rejected'}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:opacity-40 text-white text-sm font-medium rounded-lg transition-colors"
              >
                Reject
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

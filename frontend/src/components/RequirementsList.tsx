import type { Requirement, UploadResponse } from '../types';

interface Props {
  uploadResponse: UploadResponse;
  onStartReview: () => void;
}

function DocTypeBadge({ docType, count }: { docType: string; count: number }) {
  const isStructured = docType === 'structured';
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ${
        isStructured
          ? 'bg-blue-100 text-blue-700'
          : 'bg-purple-100 text-purple-700'
      }`}
    >
      {isStructured ? '📋' : '📄'}
      {isStructured ? `Structured (${count} questions)` : `Narrative (${count} requirements)`}
    </span>
  );
}

function RequirementRow({ req }: { req: Requirement }) {
  return (
    <div className="flex gap-4 py-3 border-b border-gray-100 last:border-0">
      <span className="flex-shrink-0 w-7 h-7 rounded-full bg-gray-100 text-gray-500 text-sm font-medium flex items-center justify-center">
        {req.id}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-gray-800 leading-relaxed">{req.text}</p>
        {req.reference && (
          <p className="mt-0.5 text-xs text-gray-400">{req.reference}</p>
        )}
      </div>
    </div>
  );
}

export function RequirementsList({ uploadResponse, onStartReview }: Props) {
  const { filename, doc_type, requirements } = uploadResponse;

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Readily</h1>
          <p className="text-sm text-gray-400 mt-0.5">Step 2 of 3 — Confirm Requirements</p>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          {/* File info */}
          <div className="flex items-start justify-between gap-4 mb-5">
            <div>
              <h2 className="text-lg font-semibold text-gray-900 truncate">{filename}</h2>
              <p className="text-sm text-gray-500 mt-1">
                Review the extracted requirements before starting AI analysis.
              </p>
            </div>
            <DocTypeBadge docType={doc_type} count={requirements.length} />
          </div>

          {/* Requirements list */}
          <div className="max-h-[50vh] overflow-y-auto rounded-lg border border-gray-100 bg-gray-50 px-4 py-2 mb-6">
            {requirements.map((req: Requirement) => (
              <RequirementRow key={req.id} req={req} />
            ))}
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-400">
              {requirements.length} requirement{requirements.length !== 1 ? 's' : ''} found
            </p>
            <button
              onClick={onStartReview}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors shadow-sm"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              Start Review
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

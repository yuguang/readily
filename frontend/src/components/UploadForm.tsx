import { useCallback, useEffect, useRef, useState } from 'react';
import { uploadDocument } from '../api/client';
import { useExtractionProgress } from '../hooks/useExtractionProgress';
import { mockUploadResponse } from '../mocks/mockData';
import type { Requirement, UploadResponse } from '../types';

const USE_MOCK = import.meta.env.VITE_MOCK === 'true';

// Pipeline step descriptors — match the SSE `step` field values from the backend.
const COMPLIANCE_STEPS: { key: string; label: string }[] = [
  { key: 'parsing', label: 'Parse' },
  { key: 'segmenting', label: 'Segment' },
  { key: 'filtering', label: 'Filter' },
  { key: 'extracting', label: 'Extract' },
  { key: 'deduplicating', label: 'Dedup' },
  { key: 'hierarchy', label: 'Hier.' },
  { key: 'finalizing', label: 'Done' },
];

const NARRATIVE_STEPS: { key: string; label: string }[] = [
  { key: 'reading', label: 'Read' },
  { key: 'extracting', label: 'Extract' },
  { key: 'parsing', label: 'Parse' },
  { key: 'segmenting', label: 'Segment' },
  { key: 'defining', label: 'Define' },
  { key: 'saving', label: 'Save' },
];

// ── Extraction progress panel ─────────────────────────────────────────────────

interface ExtractionProgressProps {
  filename: string;
  sessionId: string;
  docType: string;
  onComplete: (requirements: Requirement[]) => void;
  onError: (message: string) => void;
}

function ExtractionProgress({ filename, sessionId, docType, onComplete, onError }: ExtractionProgressProps) {
  const { currentStep, requirements, error } = useExtractionProgress(sessionId);

  useEffect(() => {
    if (requirements) onComplete(requirements);
  }, [requirements, onComplete]);

  useEffect(() => {
    if (error) onError(error);
  }, [error, onError]);

  const steps = docType === 'narrative' ? NARRATIVE_STEPS : COMPLIANCE_STEPS;
  const stepNum = currentStep?.step_number ?? 0;
  const totalSteps = currentStep?.total_steps ?? steps.length;
  const activeKey = currentStep?.step ?? '';

  return (
    <div className="flex flex-col gap-5">
      {/* File name + current step detail */}
      <div className="text-center">
        <p className="font-semibold text-gray-800 truncate">{filename}</p>
        <p className="mt-1 text-sm text-blue-600 font-medium">
          {currentStep
            ? `Step ${stepNum} of ${totalSteps}: ${currentStep.detail}`
            : 'Starting extraction…'}
        </p>
      </div>

      {/* Section-level sub-progress (only shown during the extraction step) */}
      {currentStep?.sections_total != null && currentStep.sections_total > 0 && (
        <div>
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Extracting requirements</span>
            <span>
              {currentStep.sections_completed ?? 0} / {currentStep.sections_total} sections
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="h-2 bg-blue-500 rounded-full transition-all duration-300"
              style={{
                width: `${Math.round(
                  ((currentStep.sections_completed ?? 0) / currentStep.sections_total) * 100,
                )}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Pipeline step indicators */}
      <div className="flex items-end justify-center gap-2">
        {steps.map((s, i) => {
          const stepIndex = i + 1;
          const isDone = stepIndex < stepNum;
          const isActive = s.key === activeKey;
          return (
            <div key={s.key} className="flex flex-col items-center gap-1.5">
              <div
                className={`w-9 h-9 rounded-lg border-2 flex items-center justify-center text-sm font-bold transition-all ${
                  isDone
                    ? 'border-green-500 bg-green-50 text-green-600'
                    : isActive
                      ? 'border-blue-500 bg-blue-50 text-blue-600 animate-pulse'
                      : 'border-gray-200 bg-white text-gray-300'
                }`}
              >
                {isDone ? '✓' : isActive ? '●' : ''}
              </div>
              <span className="text-[10px] text-gray-400 w-10 text-center leading-tight">
                {s.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main upload form ──────────────────────────────────────────────────────────

interface Props {
  onUploaded: (response: UploadResponse) => void;
}

export function UploadForm({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Set when the backend returns extraction_status: "processing" for a long doc.
  const [extractingSession, setExtractingSession] = useState<{
    sessionId: string;
    filename: string;
    docType: string;
  } | null>(null);

  const handleExtractionComplete = useCallback(
    (requirements: Requirement[]) => {
      if (!extractingSession) return;
      onUploaded({
        session_id: extractingSession.sessionId,
        filename: extractingSession.filename,
        doc_type: extractingSession.docType,
        extraction_status: 'complete',
        requirements,
      });
    },
    [extractingSession, onUploaded],
  );

  const handleExtractionError = useCallback((msg: string) => {
    setExtractingSession(null);
    setError(msg);
  }, []);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.endsWith('.pdf')) {
        setError('Please upload a PDF file.');
        return;
      }
      setError(null);
      setUploading(true);
      try {
        if (USE_MOCK) {
          // Simulate network delay
          await new Promise((r) => setTimeout(r, 1200));
          onUploaded(mockUploadResponse);
        } else {
          const response = await uploadDocument(file);
          if (response.extraction_status === 'processing') {
            // Long doc — switch to extraction progress view.
            setExtractingSession({
              sessionId: response.session_id,
              filename: response.filename,
              docType: response.doc_type,
            });
          } else {
            onUploaded(response);
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Upload failed. Please try again.');
      } finally {
        setUploading(false);
      }
    },
    [onUploaded],
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) void handleFile(file);
    },
    [handleFile],
  );

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) void handleFile(file);
    },
    [handleFile],
  );

  // ── Extraction progress view (long docs) ───────────────────────────────
  if (extractingSession) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
        <div className="w-full max-w-xl">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-bold text-gray-900">Readily</h1>
            <p className="mt-2 text-gray-500">
              {extractingSession.docType === 'narrative'
                ? 'Extracting requirements…'
                : 'Processing compliance document…'}
            </p>
          </div>
          <div className="bg-white rounded-xl border-2 border-blue-200 shadow-sm p-8">
            <ExtractionProgress
              filename={extractingSession.filename}
              sessionId={extractingSession.sessionId}
              docType={extractingSession.docType}
              onComplete={handleExtractionComplete}
              onError={handleExtractionError}
            />
          </div>
          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              {error}
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── Default upload drop-zone view ──────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
      <div className="w-full max-w-xl">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-900">Readily</h1>
          <p className="mt-2 text-gray-500">
            AI-assisted compliance review for regulatory policy documents
          </p>
        </div>

        {/* Drop zone */}
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload PDF file"
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`
            border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors
            ${dragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-white hover:border-blue-400 hover:bg-gray-50'}
          `}
        >
          {uploading ? (
            <div className="flex flex-col items-center gap-3">
              <svg
                className="animate-spin h-10 w-10 text-blue-500"
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
              <p className="text-gray-600 font-medium">Uploading & extracting requirements…</p>
            </div>
          ) : (
            <>
              <svg
                className="mx-auto h-12 w-12 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <p className="mt-4 text-gray-700 font-medium">
                Drag & drop your PDF here
              </p>
              <p className="mt-1 text-sm text-gray-400">or click to browse</p>
              <p className="mt-3 text-xs text-gray-400">PDF files only</p>
            </>
          )}
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            {error}
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          className="hidden"
          onChange={onInputChange}
        />

        {USE_MOCK && (
          <p className="mt-4 text-center text-xs text-amber-600 font-medium">
            Mock mode — no backend required
          </p>
        )}
      </div>
    </div>
  );
}

import { useCallback, useRef, useState } from 'react';
import { uploadDocument } from '../api/client';
import { mockUploadResponse } from '../mocks/mockData';
import type { UploadResponse } from '../types';

const USE_MOCK = import.meta.env.VITE_MOCK === 'true';

interface Props {
  onUploaded: (response: UploadResponse) => void;
}

export function UploadForm({ onUploaded }: Props) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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
          onUploaded(response);
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

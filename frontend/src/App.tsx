import { useState } from 'react';
import { RequirementsList } from './components/RequirementsList';
import { ReviewTable } from './components/ReviewTable';
import { UploadForm } from './components/UploadForm';
import type { UploadResponse } from './types';

type View = 'upload' | 'confirm' | 'review';

export default function App() {
  const [view, setView] = useState<View>('upload');
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);

  const handleUploaded = (response: UploadResponse) => {
    setUploadResponse(response);
    setView('confirm');
  };

  const handleStartReview = () => {
    setView('review');
  };

  if (view === 'upload') {
    return <UploadForm onUploaded={handleUploaded} />;
  }

  if (view === 'confirm' && uploadResponse) {
    return (
      <RequirementsList
        uploadResponse={uploadResponse}
        onStartReview={handleStartReview}
      />
    );
  }

  if (view === 'review' && uploadResponse) {
    return <ReviewTable uploadResponse={uploadResponse} />;
  }

  // Fallback
  return <UploadForm onUploaded={handleUploaded} />;
}

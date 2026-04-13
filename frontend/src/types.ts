export type AnswerType = 'yes' | 'no' | 'partial';

export type EvaluationStatus = 'pending' | 'approved' | 'edited' | 'rejected';

export type ReviewPhase = 'idle' | 'reviewing' | 'critic' | 'complete';

export interface Requirement {
  id: number;
  text: string;
  reference?: string;
  category?: string;
}

export interface Evaluation {
  requirement_id: number;
  answer: AnswerType;
  citation_text: string;
  source_file: string;
  page_number: number;
  confidence: number;
  reasoning: string;
  needs_human_review: boolean;
  status: EvaluationStatus;
  reviewer_notes: string;
}

export interface ReviewSession {
  id: string;
  filename: string;
  doc_type: string;
  requirements: Requirement[];
  evaluations: Evaluation[];
  created_at: string;
  status: string;
  progress: number;
}

export interface UploadResponse {
  session_id: string;
  filename: string;
  doc_type: string;
  requirements: Requirement[];
}

export interface UpdateEvaluationRequest {
  answer?: AnswerType;
  citation_text?: string;
  reviewer_notes?: string;
  status?: EvaluationStatus;
}

export interface SSEEvaluationPayload {
  evaluation: Evaluation;
  progress: number;
  total: number;
}

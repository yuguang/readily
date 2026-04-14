import type { Evaluation, Requirement, UploadResponse } from '../types';

export const mockRequirements: Requirement[] = [
  {
    id: 1,
    text: 'Does the P&P state that under existing Contract requirements, only general inpatient care is subject to Prior Authorization regardless of whether services are rendered by an in-network or out-of-network provider?',
    reference: 'APL 25-008, page 1',
  },
  {
    id: 2,
    text: 'Does the P&P state that the Plan must not require Prior Authorization for emergency services, including post-stabilization services?',
    reference: 'APL 25-008, page 1',
  },
  {
    id: 3,
    text: 'Does the P&P describe the process for submitting Prior Authorization requests, including required documentation and timeframes?',
    reference: 'APL 25-008, page 2',
  },
  {
    id: 4,
    text: 'Does the P&P specify that decisions for standard Prior Authorization requests must be made within 5 business days?',
    reference: 'APL 25-008, page 3',
  },
  {
    id: 5,
    text: 'Does the P&P state that expedited Prior Authorization decisions must be made within 72 hours when the standard timeframe would seriously jeopardize the member\'s health?',
    reference: 'APL 25-008, page 3',
  },
  {
    id: 6,
    text: 'Does the P&P include procedures for notifying members of Prior Authorization decisions, including denials and partial approvals?',
    reference: 'APL 25-008, page 4',
  },
  {
    id: 7,
    text: 'Does the P&P describe the appeals and grievance process available to members when a Prior Authorization request is denied?',
    reference: 'APL 25-008, page 5',
  },
  {
    id: 8,
    text: 'Does the P&P require that clinical criteria used for Prior Authorization decisions are evidence-based and consistently applied?',
    reference: 'APL 25-008, page 6',
  },
  {
    id: 9,
    text: 'Does the P&P specify requirements for continuity of care when a member\'s provider leaves the network mid-treatment?',
    reference: 'APL 25-008, page 7',
  },
  {
    id: 10,
    text: 'Does the P&P include provisions for members with complex or chronic conditions who require ongoing Prior Authorization approvals?',
    reference: 'APL 25-008, page 8',
  },
  {
    id: 11,
    text: 'Does the P&P describe how the Plan monitors and audits Prior Authorization decision-making for consistency and accuracy?',
    reference: 'APL 25-008, page 9',
  },
  {
    id: 12,
    text: 'Does the P&P address requirements for providing language assistance and translated materials related to Prior Authorization decisions?',
    reference: 'APL 25-008, page 10',
  },
];

export const mockEvaluations: Evaluation[] = [
  {
    requirement_id: 1,
    answer: 'yes',
    citation_text:
      '"Only general inpatient care shall be subject to prior authorization regardless of whether services are rendered by an in-network or out-of-network provider, consistent with the terms of the Contract."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 8,
    confidence: 0.95,
    reasoning:
      'The policy explicitly states that only general inpatient care requires prior auth, directly matching the requirement.',
    needs_human_review: false,
    status: 'pending',
    reviewer_notes: '',
  },
  {
    requirement_id: 2,
    answer: 'yes',
    citation_text:
      '"The Plan shall not require Prior Authorization for emergency services, including emergency transportation, and post-stabilization services as defined in Title 28, CCR Section 1300.67.2.2."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 8,
    confidence: 0.92,
    reasoning:
      'The policy explicitly exempts emergency services and post-stabilization services from prior authorization requirements.',
    needs_human_review: false,
    status: 'pending',
    reviewer_notes: '',
  },
  {
    requirement_id: 3,
    answer: 'yes',
    citation_text:
      '"Providers must submit prior authorization requests via the secure provider portal or by fax, including clinical documentation, diagnosis codes, and proposed treatment plan."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 12,
    confidence: 0.88,
    reasoning:
      'The policy describes submission methods and required documentation, satisfying this requirement.',
    needs_human_review: false,
    status: 'pending',
    reviewer_notes: '',
  },
  {
    requirement_id: 4,
    answer: 'yes',
    citation_text:
      '"Standard prior authorization decisions shall be rendered within five (5) business days of receipt of all necessary information."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 15,
    confidence: 0.97,
    reasoning:
      'Exact 5 business day standard is stated verbatim in the policy.',
    needs_human_review: false,
    status: 'pending',
    reviewer_notes: '',
  },
  {
    requirement_id: 5,
    answer: 'partial',
    citation_text:
      '"Expedited decisions will be processed within 72 hours when clinically warranted based on provider attestation."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 15,
    confidence: 0.68,
    reasoning:
      'The 72-hour timeframe is present but the policy does not explicitly mention "seriously jeopardize the member\'s health" language as required by the APL.',
    needs_human_review: true,
    status: 'pending',
    reviewer_notes: '',
  },
  {
    requirement_id: 6,
    answer: 'yes',
    citation_text:
      '"Members shall be notified in writing of prior authorization decisions within one (1) business day of the decision, including the basis for any denial or partial approval and information regarding the appeals process."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 18,
    confidence: 0.91,
    reasoning:
      'The policy covers all required elements: notification, denial basis, and appeals information.',
    needs_human_review: false,
    status: 'pending',
    reviewer_notes: '',
  },
  {
    requirement_id: 7,
    answer: 'yes',
    citation_text:
      '"Members denied prior authorization have the right to file a grievance or appeal within 180 days of the notice of action. Appeals will be reviewed by a physician not involved in the original decision."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 22,
    confidence: 0.89,
    reasoning:
      'The appeals and grievance process is clearly described including timeframes and independence of reviewer.',
    needs_human_review: false,
    status: 'pending',
    reviewer_notes: '',
  },
  {
    requirement_id: 8,
    answer: 'yes',
    citation_text:
      '"All prior authorization criteria shall be evidence-based, developed using nationally recognized clinical guidelines, and applied consistently across all members."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 25,
    confidence: 0.93,
    reasoning:
      'Policy explicitly requires evidence-based criteria and consistent application.',
    needs_human_review: false,
    status: 'pending',
    reviewer_notes: '',
  },
  {
    requirement_id: 9,
    answer: 'no',
    citation_text:
      '"Members are encouraged to transition to in-network providers when possible. The Plan provides a directory of contracted providers."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 30,
    confidence: 0.42,
    reasoning:
      'The policy does not address continuity of care requirements when a provider leaves the network mid-treatment. The cited text only describes general provider directory information.',
    needs_human_review: true,
    status: 'pending',
    reviewer_notes: '',
  },
  {
    requirement_id: 10,
    answer: 'partial',
    citation_text:
      '"Members with ongoing treatment needs may request extended authorization periods of up to 90 days."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 32,
    confidence: 0.55,
    reasoning:
      'The policy mentions extended authorization for ongoing treatment but does not specifically address complex or chronic conditions requiring continuous oversight.',
    needs_human_review: true,
    status: 'pending',
    reviewer_notes: '',
  },
  {
    requirement_id: 11,
    answer: 'yes',
    citation_text:
      '"The Plan will conduct quarterly audits of prior authorization decisions to ensure consistency, accuracy, and compliance with clinical criteria. Results will be reported to the Quality Improvement Committee."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 40,
    confidence: 0.86,
    reasoning:
      'Monitoring and auditing requirements are clearly described with reporting requirements.',
    needs_human_review: false,
    status: 'pending',
    reviewer_notes: '',
  },
  {
    requirement_id: 12,
    answer: 'yes',
    citation_text:
      '"All prior authorization notices and decision letters shall be available in threshold languages and upon request in any language spoken by the member. Language assistance services are available at no cost."',
    source_file: 'GG/GG.4521_CEO20250129.pdf',
    page_number: 44,
    confidence: 0.90,
    reasoning:
      'Policy covers language assistance and translated materials for prior authorization communications.',
    needs_human_review: false,
    status: 'pending',
    reviewer_notes: '',
  },
];

export const mockUploadResponse: UploadResponse = {
  session_id: 'mock-session-001',
  filename: 'Example_Input_Doc_Easy.pdf',
  doc_type: 'structured',
  extraction_status: 'complete',
  requirements: mockRequirements,
};

const BASE = '/api/v1';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

/* ── Types ─────────────────────────────────────────────────────────── */

export interface DocumentListItem {
  doc_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  uploaded_at: string;
}

export interface DocumentDetail {
  doc_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  uploaded_at: string;
  metadata: Record<string, unknown> | null;
  summary: string | null;
}

export interface DocumentProgress {
  doc_id: string;
  status: string;
  stage: string;
  stage_num: number;
  total_stages: number;
  chunks_embedded: number;
  total_chunks: number;
  percent: number;
}

export interface VectorInfo {
  point_id: string;
  chunk_index: number;
  vector_dim: number;
  vector_preview: number[];
}

export interface SourceDoc {
  content: string;
  metadata: Record<string, unknown>;
  score: number;
}

export interface ChatResponse {
  answer: string;
  sources: SourceDoc[];
  confidence: number;
  latency_ms: number;
}

export interface SearchResult {
  doc_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  patient_name: string | null;
  mrn: string | null;
  dob: string | null;
  document_type: string | null;
}

/* ── API calls ─────────────────────────────────────────────────────── */

export async function uploadDocument(file: File): Promise<{ doc_id: string; filename: string; status: string }> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/documents`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function listDocuments(): Promise<DocumentListItem[]> {
  return request<DocumentListItem[]>(`${BASE}/documents`);
}

export async function getDocument(docId: string): Promise<DocumentDetail> {
  return request<DocumentDetail>(`${BASE}/documents/${docId}`);
}

export async function getDocumentStatus(docId: string): Promise<DocumentProgress> {
  return request<DocumentProgress>(`${BASE}/documents/${docId}/status`);
}

export function getDocumentFileUrl(docId: string): string {
  return `${BASE}/documents/${docId}/file`;
}

export async function getDocumentVectors(docId: string): Promise<VectorInfo[]> {
  return request<VectorInfo[]>(`${BASE}/documents/${docId}/vectors`);
}

export async function chatWithDocument(docId: string, question: string, topK = 5): Promise<ChatResponse> {
  return request<ChatResponse>(`${BASE}/documents/${docId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  });
}

export async function searchDocuments(params: Record<string, string>): Promise<SearchResult[]> {
  const qs = new URLSearchParams(params).toString();
  return request<SearchResult[]>(`${BASE}/search?${qs}`);
}

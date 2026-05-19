export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  const payload = text ? JSON.parse(text) : undefined;

  if (!response.ok) {
    const message = payload && typeof payload === 'object' && 'detail' in payload ? String(payload.detail) : response.statusText;
    throw new ApiError(message, response.status);
  }

  return payload as T;
}

export function buildQuery(params: Record<string, string | number | boolean | undefined>): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) query.set(key, String(value));
  }
  const text = query.toString();
  return text ? '?' + text : '';
}

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
    const message = payload && typeof payload === 'object' && 'detail' in payload
      ? formatApiDetail((payload as { detail: unknown }).detail)
      : response.statusText;
    throw new ApiError(message, response.status);
  }

  return payload as T;
}

function formatApiDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (!item || typeof item !== 'object') return String(item);
      const record = item as Record<string, unknown>;
      const path = Array.isArray(record.loc) ? record.loc.join('.') : '';
      const message = typeof record.msg === 'string' ? record.msg : JSON.stringify(record);
      return path ? `${path}: ${message}` : message;
    });
    return messages.join('; ');
  }
  return JSON.stringify(detail);
}

export function buildQuery(params: Record<string, string | number | boolean | undefined>): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) query.set(key, String(value));
  }
  const text = query.toString();
  return text ? '?' + text : '';
}

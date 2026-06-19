import { JsonCopyButton } from './JsonCopyButton';

export function JsonDetails({ title, value }: { title: string; value: unknown }) {
  const json = JSON.stringify(value, null, 2);

  return (
    <details className="json-details">
      <summary>
        <span>{title}</span>
        <JsonCopyButton text={json} stopPropagation />
      </summary>
      <pre className="json-code-window">{json}</pre>
    </details>
  );
}

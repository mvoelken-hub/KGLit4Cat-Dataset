export function JsonDetails({ title, value }: { title: string; value: unknown }) {
  return (
    <details className="json-details">
      <summary>{title}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

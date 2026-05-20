import { useState } from 'react';

export type JsonValue = string | number | boolean | null | JsonObject | JsonArray;
export type JsonObject = { [key: string]: JsonValue };
export type JsonArray = JsonValue[];

export type JsonPatchMarker = {
  id: string;
  path: string;
  status: 'accepted' | 'needs_review' | 'unmapped';
  label: string;
  detail?: string;
  confidence?: number;
  fileName?: string;
  patch?: unknown;
  evidence?: string[];
  issues?: string[];
  resolved?: boolean;
};

export function getValueAtPath(obj: JsonObject, path: string): JsonValue | undefined {
  if (!path) return obj;
  const parts = path.split('.');
  let current: JsonValue = obj;
  for (const part of parts) {
    if (current === null || typeof current !== 'object') return undefined;
    if (Array.isArray(current)) {
      const index = parseInt(part, 10);
      current = current[index];
    } else {
      current = (current as JsonObject)[part];
    }
  }
  return current;
}

export function setValueAtPath(obj: JsonObject, path: string, value: JsonValue): JsonObject {
  if (!path) return value as JsonObject;
  const result: JsonObject = JSON.parse(JSON.stringify(obj));
  const parts = path.split('.');
  let current: JsonObject = result;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    const next = current[part];
    if (Array.isArray(next)) {
      current[part] = [...next];
    } else if (typeof next === 'object' && next !== null) {
      current[part] = { ...next };
    }
    current = current[part] as JsonObject;
  }
  current[parts[parts.length - 1]] = value;
  return result;
}

function isTopLevelPath(path: string): boolean {
  return path.length > 0 && !path.includes('.');
}

function isPathProtected(path: string, protectedPaths: string[]): boolean {
  if (!path) return false;
  const topLevelPath = path.split('.', 1)[0];
  return protectedPaths.includes(topLevelPath);
}

function formatPrimitivePreview(value: JsonValue): string {
  if (value === null) return 'null';
  const text = String(value);
  return text.length > 34 ? text.slice(0, 31) + '...' : text;
}

function TreeNode({
  data,
  path,
  selectedPath,
  onSelect,
  protectedPaths,
  onToggleProtected,
  patchMarkers,
  depth = 0,
}: {
  data: JsonValue;
  path: string;
  selectedPath: string;
  onSelect: (path: string) => void;
  protectedPaths: string[];
  onToggleProtected: (path: string) => void;
  patchMarkers: JsonPatchMarker[];
  depth?: number;
}) {
  if (data === null || data === undefined) return null;

  const isPrimitive = typeof data !== 'object';
  const isArray = Array.isArray(data);
  const label = path.split('.').pop() || 'root';
  const isSelected = path === selectedPath;
  const isLockable = isTopLevelPath(path);
  const isProtected = isPathProtected(path, protectedPaths);
  const fieldMarkers = patchMarkers.filter((marker) => marker.path === path);
  const childCount = isPrimitive ? 0 : isArray ? data.length : Object.keys(data).length;
  const displayLabel = isPrimitive
    ? `${label}: ${formatPrimitivePreview(data)}`
    : isArray
      ? `${label} [${childCount}]`
      : `${label} (${childCount})`;

  return (
    <>
      <li
        className={`${isSelected ? 'active' : ''} ${isProtected ? 'protected' : ''}`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={(e) => { e.stopPropagation(); onSelect(path); }}
      >
        <span className="tree-icon">{isPrimitive ? '•' : '▸'}</span>
        {displayLabel}
        {fieldMarkers.length > 0 && (
          <span className="patch-marker-count" title="Patch markers on this field">
            {fieldMarkers.length}
          </span>
        )}
        {isLockable && (
          <button
            className={`lock-toggle ${isProtected ? 'locked' : ''}`}
            title={isProtected ? 'Unlock field' : 'Lock field'}
            onClick={(e) => { e.stopPropagation(); onToggleProtected(path); }}
          >
            {isProtected ? '🔒' : '🔓'}
          </button>
        )}
      </li>
      {isArray && data.map((item, index) => {
        const itemPath = path ? `${path}.${index}` : `${index}`;
        // Only recurse if the item is an object (not primitive)
        if (typeof item !== 'object' || item === null) return null;
        return (
          <TreeNode
            key={itemPath}
            data={item}
            path={itemPath}
            selectedPath={selectedPath}
            onSelect={onSelect}
            protectedPaths={protectedPaths}
            onToggleProtected={onToggleProtected}
            patchMarkers={patchMarkers}
            depth={depth + 1}
          />
        );
      })}
      {!isArray && Object.entries(data).map(([key, value]) => {
        const childPath = path ? `${path}.${key}` : key;
        if (path && (typeof value !== 'object' || value === null)) return null;
        return (
          <TreeNode
            key={key}
            data={value}
            path={childPath}
            selectedPath={selectedPath}
            onSelect={onSelect}
            protectedPaths={protectedPaths}
            onToggleProtected={onToggleProtected}
            patchMarkers={patchMarkers}
            depth={depth + 1}
          />
        );
      })}
    </>
  );
}

function ValueEditor({
  value,
  path,
  onChange,
  protectedPaths,
  onToggleProtected,
  patchMarkers,
  onApplyPatch,
}: {
  value: JsonValue;
  path: string;
  onChange: (path: string, newValue: JsonValue) => void;
  protectedPaths: string[];
  onToggleProtected: (path: string) => void;
  patchMarkers: JsonPatchMarker[];
  onApplyPatch?: (itemId: string, value: unknown) => void;
}) {
  const isProtected = isPathProtected(path, protectedPaths);

  if (value === null || value === undefined) {
    return (
      <select
        className="json-editor-input"
        value="null"
        onChange={(e) => {
          const v = e.target.value;
          if (v === 'null') onChange(path, null);
          else if (v === 'true') onChange(path, true);
          else if (v === 'false') onChange(path, false);
          else if (v === '') onChange(path, '');
          else if (!Number.isNaN(Number(v))) onChange(path, Number(v));
          else onChange(path, v);
        }}
        disabled={isProtected}
      >
        <option value="null">null</option>
        <option value="true">true</option>
        <option value="false">false</option>
        <option value="">empty string</option>
      </select>
    );
  }

  if (typeof value === 'string') {
    if (value.length > 80) {
      return (
        <textarea
          className="json-editor-input"
          value={value}
          onChange={(e) => onChange(path, e.target.value)}
          disabled={isProtected}
          rows={4}
        />
      );
    }
    return (
      <input
        type="text"
        className="json-editor-input"
        value={value}
        onChange={(e) => onChange(path, e.target.value)}
        disabled={isProtected}
      />
    );
  }

  if (typeof value === 'number') {
    return (
      <input
        type="number"
        className="json-editor-input"
        value={value}
        onChange={(e) => onChange(path, Number(e.target.value))}
        disabled={isProtected}
      />
    );
  }

  if (typeof value === 'boolean') {
    return (
      <select
        className="json-editor-input"
        value={String(value)}
        onChange={(e) => onChange(path, e.target.value === 'true')}
        disabled={isProtected}
      >
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    );
  }

  if (Array.isArray(value)) {
    const fieldName = path.split('.').pop();
    const isStringArray = value.length > 0 && value.every((item) => typeof item === 'string');
    if ((fieldName === 'title' || fieldName === 'description') && isStringArray) {
      const firstValue = value[0] as string;
      if (firstValue.length > 80) {
        return (
          <textarea
            className="json-editor-input"
            value={firstValue}
            onChange={(e) => onChange(path, [e.target.value])}
            disabled={isProtected}
            rows={4}
          />
        );
      }
      return (
        <input
          type="text"
          className="json-editor-input"
          value={firstValue}
          onChange={(e) => onChange(path, [e.target.value])}
          disabled={isProtected}
        />
      );
    }

    return (
      <div className="json-editor-array">
        <p className="json-editor-array-info">Array with {value.length} items</p>
        <div className="json-editor-cards">
          {value.map((item, index) => {
            const itemPath = `${path}.${index}`;
            const itemLabel = typeof item === 'object' && item !== null
              ? String((item as JsonObject).title || (item as JsonObject).id || (item as JsonObject).name || `Item ${index + 1}`)
              : String(item).substring(0, 50);
            return (
              <div
                key={index}
                className="json-editor-card"
                onClick={() => onChange('__select__', itemPath)}
                aria-disabled={isProtected}
              >
                <strong>{itemLabel}</strong>
                <small>{typeof item === 'object' ? 'Object' : typeof item}</small>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div className="json-editor-object">
      {Object.entries(value).map(([key, val]) => {
        const childPath = path ? `${path}.${key}` : key;
        const childMarkers = patchMarkers.filter((marker) => marker.path === childPath);
        return (
        <div
          key={key}
          className={`json-editor-field ${isPathProtected(childPath, protectedPaths) ? 'protected' : ''}`}
        >
          <div className="json-editor-field-label">
            <label>{key}</label>
            <PatchMarkerBadges markers={childMarkers} />
            {isTopLevelPath(childPath) && (
              <button
                className={`lock-toggle ${isPathProtected(key, protectedPaths) ? 'locked' : ''}`}
                title={isPathProtected(key, protectedPaths) ? 'Unlock field' : 'Lock field'}
                onClick={() => onToggleProtected(key)}
              >
                {isPathProtected(key, protectedPaths) ? '🔒' : '🔓'}
              </button>
            )}
          </div>
          <SelectedPatchMarkerReview markers={childMarkers} compact onApplyPatch={onApplyPatch} />
          <ValueEditor
            value={val}
            path={childPath}
            onChange={onChange}
            protectedPaths={protectedPaths}
            onToggleProtected={onToggleProtected}
            patchMarkers={patchMarkers}
            onApplyPatch={onApplyPatch}
          />
        </div>
        );
      })}
    </div>
  );
}

function formatPatchValue(value: unknown): string {
  if (value === undefined) return '';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function extractPatchInnerValue(patch: unknown): unknown {
  if (patch && typeof patch === 'object' && !Array.isArray(patch)) {
    const values = Object.values(patch as Record<string, unknown>);
    if (values.length === 1) return values[0];
  }
  return patch;
}

export function PatchValueEditor({ value, onChange }: { value: unknown; onChange: (value: unknown) => void }) {
  if (value === null || value === undefined) {
    return (
      <select
        className="json-editor-input"
        value={value === null ? 'null' : 'undefined'}
        onChange={(e) => {
          const v = e.target.value;
          if (v === 'null') onChange(null);
          else if (v === 'true') onChange(true);
          else if (v === 'false') onChange(false);
          else if (v === '') onChange('');
          else if (!Number.isNaN(Number(v))) onChange(Number(v));
          else onChange(v);
        }}
      >
        <option value="null">null</option>
        <option value="true">true</option>
        <option value="false">false</option>
        <option value="">empty string</option>
      </select>
    );
  }

  if (typeof value === 'string') {
    if (value.length > 80) {
      return (
        <textarea
          className="json-editor-input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          rows={4}
        />
      );
    }
    return (
      <input
        type="text"
        className="json-editor-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }

  if (typeof value === 'number') {
    return (
      <input
        type="number"
        className="json-editor-input"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    );
  }

  if (typeof value === 'boolean') {
    return (
      <select
        className="json-editor-input"
        value={String(value)}
        onChange={(e) => onChange(e.target.value === 'true')}
      >
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    );
  }

  if (Array.isArray(value)) {
    const isPrimitiveArray = value.every((item) =>
      item === null ||
      ['string', 'number', 'boolean'].includes(typeof item)
    );

    if (isPrimitiveArray) {
      return (
        <div className="patch-array-editor">
          {value.map((item, index) => (
            <div key={index} className="patch-array-item">
              <PatchValueEditor
                value={item}
                onChange={(newItem) => {
                  const next = [...value];
                  next[index] = newItem;
                  onChange(next);
                }}
              />
              <button
                className="ghost"
                onClick={() => {
                  const next = value.filter((_, i) => i !== index);
                  onChange(next);
                }}
              >
                Remove
              </button>
            </div>
          ))}
          <button
            className="ghost"
            onClick={() => {
              const first = value[0];
              let newItem: unknown = '';
              if (typeof first === 'number') newItem = 0;
              else if (typeof first === 'boolean') newItem = false;
              else if (first === null) newItem = null;
              onChange([...value, newItem]);
            }}
          >
            Add item
          </button>
        </div>
      );
    }
  }

  // Fallback: raw JSON textarea for objects and mixed arrays
  const [jsonText, setJsonText] = useState(() => JSON.stringify(value, null, 2));
  const [jsonError, setJsonError] = useState<string | null>(null);

  return (
    <div>
      <textarea
        className="json-editor-input"
        style={{ fontFamily: "'Courier New', monospace", minHeight: '120px' }}
        value={jsonText}
        onChange={(e) => {
          setJsonText(e.target.value);
          try {
            const parsed = JSON.parse(e.target.value);
            setJsonError(null);
            onChange(parsed);
          } catch (err) {
            setJsonError(err instanceof Error ? err.message : 'Invalid JSON');
          }
        }}
        rows={6}
      />
      {jsonError && <small className="warning" style={{ display: 'block', marginTop: '6px' }}>{jsonError}</small>}
    </div>
  );
}

function PatchMarkerBadges({ markers }: { markers: JsonPatchMarker[] }) {
  if (!markers.length) return null;
  return (
    <div className="patch-marker-badges">
      {markers.map((marker, index) => (
        <span
          key={`${marker.path}-${marker.status}-${index}`}
          className={`patch-marker-badge ${marker.status}`}
          title={[marker.detail, marker.confidence !== undefined ? `Confidence ${Math.round(marker.confidence * 100)}%` : '']
            .filter(Boolean)
            .join(' - ')}
        >
          {marker.label}
        </span>
      ))}
    </div>
  );
}

function SelectedPatchMarkerReview({ markers, compact = false, onApplyPatch }: { markers: JsonPatchMarker[]; compact?: boolean; onApplyPatch?: (itemId: string, value: unknown) => void }) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editedValue, setEditedValue] = useState<unknown>(null);

  if (!markers.length) return null;

  const startEditing = (marker: JsonPatchMarker) => {
    const innerValue = extractPatchInnerValue(marker.patch);
    setEditingId(marker.id);
    setEditedValue(innerValue);
  };

  const cancelEditing = () => {
    setEditingId(null);
    setEditedValue(null);
  };

  return (
    <div className={compact ? 'selected-patch-review compact' : 'selected-patch-review'}>
      <div className="selected-patch-review-heading">
        <strong>Patch review</strong>
        <span>{markers.length} marker{markers.length === 1 ? '' : 's'}</span>
      </div>
      {markers.map((marker) => (
        <article key={marker.id} className={`selected-patch-card ${marker.status}`}>
          <div className="selected-patch-card-heading">
            <span className={`patch-marker-badge ${marker.status}`}>{marker.label}</span>
            {marker.confidence !== undefined && <small>{Math.round(marker.confidence * 100)}% confidence</small>}
            {marker.fileName && <small>{marker.fileName}</small>}
          </div>
          {marker.detail && <p>{marker.detail}</p>}
          {marker.issues && marker.issues.length > 0 && (
            <div className="selected-patch-section">
              <span>Issues</span>
              <ul>
                {marker.issues.map((issue, issueIndex) => <li key={`${issue}-${issueIndex}`}>{issue}</li>)}
              </ul>
            </div>
          )}
          {marker.evidence && marker.evidence.length > 0 && (
            <div className="selected-patch-section">
              <span>Evidence</span>
              <ul>
                {marker.evidence.map((evidence, evidenceIndex) => <li key={`${evidence}-${evidenceIndex}`}>{evidence}</li>)}
              </ul>
            </div>
          )}
          {marker.patch !== undefined && (
            <div className="selected-patch-section">
              <span>Proposed change</span>
              {editingId === marker.id ? (
                <>
                  <PatchValueEditor value={editedValue} onChange={setEditedValue} />
                  <div className="patch-edit-actions">
                    <button className="ghost" onClick={cancelEditing}>Cancel</button>
                    <button onClick={() => { onApplyPatch?.(marker.id, editedValue); cancelEditing(); }}>Apply patch</button>
                  </div>
                </>
              ) : (
                <>
                  <pre>{formatPatchValue(marker.patch)}</pre>
                  {onApplyPatch && (
                    <button className="ghost" onClick={() => startEditing(marker)}>Edit patch</button>
                  )}
                </>
              )}
            </div>
          )}
        </article>
      ))}
    </div>
  );
}

export function JsonEditor({
  value,
  onChange,
  protectedPaths,
  onProtectedPathsChange,
  patchMarkers,
  onApplyPatch,
}: {
  value: Record<string, unknown>;
  onChange?: (value: Record<string, unknown>) => void;
  protectedPaths?: string[];
  onProtectedPathsChange?: (paths: string[]) => void;
  patchMarkers?: JsonPatchMarker[];
  onApplyPatch?: (itemId: string, value: unknown) => void;
}) {
  const [selectedPath, setSelectedPath] = useState<string>('');
  const [showRaw, setShowRaw] = useState(false);

  const currentValue = getValueAtPath(value as JsonObject, selectedPath);
  const selectedPatchMarkers = (patchMarkers || []).filter((marker) => marker.path === selectedPath);

  const handleChange = (path: string, newValue: JsonValue) => {
    if (path === '__select__') {
      setSelectedPath(newValue as string);
      return;
    }
    if (isPathProtected(path, protectedPaths || [])) {
      return;
    }
    if (onChange) {
      const updated = setValueAtPath(value as JsonObject, path, newValue);
      onChange(updated as Record<string, unknown>);
    }
  };

  const handleToggleProtected = (path: string) => {
    if (!onProtectedPathsChange) return;
    const current = protectedPaths || [];
    if (current.includes(path)) {
      onProtectedPathsChange(current.filter((p) => p !== path));
    } else {
      onProtectedPathsChange([...current, path]);
    }
  };

  return (
    <div className="json-editor">
      <div className="json-editor-sidebar">
        <div className="json-editor-tree">
          <ul>
            <TreeNode
              data={value as JsonObject}
              path=""
              selectedPath={selectedPath}
              onSelect={setSelectedPath}
              protectedPaths={protectedPaths || []}
              onToggleProtected={handleToggleProtected}
              patchMarkers={patchMarkers || []}
            />
          </ul>
        </div>
      </div>
      <div className="json-editor-main">
        <div className="json-editor-breadcrumb">{selectedPath || 'root'}</div>
        <div className="json-editor-panel">
          <SelectedPatchMarkerReview markers={selectedPatchMarkers} onApplyPatch={onApplyPatch} />
          {currentValue !== undefined ? (
            <ValueEditor
              value={currentValue}
              path={selectedPath}
              onChange={handleChange}
              protectedPaths={protectedPaths || []}
              onToggleProtected={handleToggleProtected}
              patchMarkers={patchMarkers || []}
              onApplyPatch={onApplyPatch}
            />
          ) : (
            <p className="muted">Select a node from the tree to edit.</p>
          )}
        </div>
        <div className="json-editor-actions">
          <button className="ghost" onClick={() => setShowRaw(!showRaw)}>
            {showRaw ? 'Hide raw JSON' : 'Show raw JSON'}
          </button>
        </div>
        {showRaw && (
          <pre className="json-editor-raw">{JSON.stringify(value, null, 2)}</pre>
        )}
      </div>
    </div>
  );
}

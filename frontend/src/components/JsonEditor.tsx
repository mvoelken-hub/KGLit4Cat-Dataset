import { useState } from 'react';

type JsonValue = string | number | boolean | null | JsonObject | JsonArray;
type JsonObject = { [key: string]: JsonValue };
type JsonArray = JsonValue[];

function getValueAtPath(obj: JsonObject, path: string): JsonValue {
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

function setValueAtPath(obj: JsonObject, path: string, value: JsonValue): JsonObject {
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

function TreeNode({
  data,
  path,
  selectedPath,
  onSelect,
  depth = 0,
}: {
  data: JsonValue;
  path: string;
  selectedPath: string;
  onSelect: (path: string) => void;
  depth?: number;
}) {
  if (data === null || data === undefined) return null;

  const isPrimitive = typeof data !== 'object';
  const isArray = Array.isArray(data);
  const label = path.split('.').pop() || 'root';
  const isSelected = path === selectedPath;

  // Don't render primitives in the tree
  if (isPrimitive) return null;

  const childCount = isArray ? data.length : Object.keys(data).length;
  const displayLabel = isArray ? `${label} [${childCount}]` : `${label} (${childCount})`;

  return (
    <>
      <li
        className={isSelected ? 'active' : ''}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={(e) => { e.stopPropagation(); onSelect(path); }}
      >
        <span className="tree-icon">{isArray ? '▸' : '▸'}</span>
        {displayLabel}
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
            depth={depth + 1}
          />
        );
      })}
      {!isArray && Object.entries(data).map(([key, value]) => {
        const childPath = path ? `${path}.${key}` : key;
        // Only recurse if the value is an object or array (not primitive)
        if (typeof value !== 'object' || value === null) return null;
        return (
          <TreeNode
            key={key}
            data={value}
            path={childPath}
            selectedPath={selectedPath}
            onSelect={onSelect}
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
}: {
  value: JsonValue;
  path: string;
  onChange: (path: string, newValue: JsonValue) => void;
}) {
  if (value === null || value === undefined) {
    return <div className="json-editor-null">null</div>;
  }

  if (typeof value === 'string') {
    if (value.length > 80) {
      return (
        <textarea
          className="json-editor-input"
          value={value}
          onChange={(e) => onChange(path, e.target.value)}
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
      />
    );
  }

  if (typeof value === 'boolean') {
    return (
      <select
        className="json-editor-input"
        value={String(value)}
        onChange={(e) => onChange(path, e.target.value === 'true')}
      >
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    );
  }

  if (Array.isArray(value)) {
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
      {Object.entries(value).map(([key, val]) => (
        <div key={key} className="json-editor-field">
          <label>{key}</label>
          <ValueEditor value={val} path={path ? `${path}.${key}` : key} onChange={onChange} />
        </div>
      ))}
    </div>
  );
}

export function JsonEditor({ value, onChange }: { value: JsonObject; onChange?: (value: JsonObject) => void }) {
  const [selectedPath, setSelectedPath] = useState<string>('');
  const [showRaw, setShowRaw] = useState(false);

  const currentValue = getValueAtPath(value, selectedPath);

  const handleChange = (path: string, newValue: JsonValue) => {
    if (path === '__select__') {
      setSelectedPath(newValue as string);
      return;
    }
    if (onChange) {
      const updated = setValueAtPath(value, path, newValue);
      onChange(updated);
    }
  };

  return (
    <div className="json-editor">
      <div className="json-editor-sidebar">
        <div className="json-editor-tree">
          <ul>
            <TreeNode data={value} path="" selectedPath={selectedPath} onSelect={setSelectedPath} />
          </ul>
        </div>
      </div>
      <div className="json-editor-main">
        <div className="json-editor-breadcrumb">{selectedPath || 'root'}</div>
        <div className="json-editor-panel">
          {currentValue !== undefined ? (
            <ValueEditor value={currentValue} path={selectedPath} onChange={handleChange} />
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

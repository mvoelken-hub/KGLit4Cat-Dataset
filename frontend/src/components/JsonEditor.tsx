import { useState } from 'react';

type JsonValue = string | number | boolean | null | JsonObject | JsonArray;
type JsonObject = { [key: string]: JsonValue };
type JsonArray = JsonValue[];

function getValueAtPath(obj: JsonObject, path: string): JsonValue | undefined {
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
  depth = 0,
}: {
  data: JsonValue;
  path: string;
  selectedPath: string;
  onSelect: (path: string) => void;
  protectedPaths: string[];
  onToggleProtected: (path: string) => void;
  depth?: number;
}) {
  if (data === null || data === undefined) return null;

  const isPrimitive = typeof data !== 'object';
  const isArray = Array.isArray(data);
  const label = path.split('.').pop() || 'root';
  const isSelected = path === selectedPath;
  const isLockable = isTopLevelPath(path);
  const isProtected = isPathProtected(path, protectedPaths);
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
}: {
  value: JsonValue;
  path: string;
  onChange: (path: string, newValue: JsonValue) => void;
  protectedPaths: string[];
  onToggleProtected: (path: string) => void;
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
      {Object.entries(value).map(([key, val]) => (
        <div
          key={key}
          className={`json-editor-field ${isPathProtected(path ? `${path}.${key}` : key, protectedPaths) ? 'protected' : ''}`}
        >
          <div className="json-editor-field-label">
            <label>{key}</label>
            {isTopLevelPath(path ? `${path}.${key}` : key) && (
              <button
                className={`lock-toggle ${isPathProtected(key, protectedPaths) ? 'locked' : ''}`}
                title={isPathProtected(key, protectedPaths) ? 'Unlock field' : 'Lock field'}
                onClick={() => onToggleProtected(key)}
              >
                {isPathProtected(key, protectedPaths) ? '🔒' : '🔓'}
              </button>
            )}
          </div>
          <ValueEditor
            value={val}
            path={path ? `${path}.${key}` : key}
            onChange={onChange}
            protectedPaths={protectedPaths}
            onToggleProtected={onToggleProtected}
          />
        </div>
      ))}
    </div>
  );
}

export function JsonEditor({
  value,
  onChange,
  protectedPaths,
  onProtectedPathsChange,
}: {
  value: Record<string, unknown>;
  onChange?: (value: Record<string, unknown>) => void;
  protectedPaths?: string[];
  onProtectedPathsChange?: (paths: string[]) => void;
}) {
  const [selectedPath, setSelectedPath] = useState<string>('');
  const [showRaw, setShowRaw] = useState(false);

  const currentValue = getValueAtPath(value as JsonObject, selectedPath);

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
            />
          </ul>
        </div>
      </div>
      <div className="json-editor-main">
        <div className="json-editor-breadcrumb">{selectedPath || 'root'}</div>
        <div className="json-editor-panel">
          {currentValue !== undefined ? (
            <ValueEditor
              value={currentValue}
              path={selectedPath}
              onChange={handleChange}
              protectedPaths={protectedPaths || []}
              onToggleProtected={handleToggleProtected}
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

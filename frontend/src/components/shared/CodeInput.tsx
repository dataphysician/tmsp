interface CodeInputProps {
  value: string;
  onChange: (value: string) => void;
  onAdd: () => void;
  disabled?: boolean;
  isLoading?: boolean;
  placeholder?: string;
  label?: string;
}

export function CodeInput({
  value,
  onChange,
  onAdd,
  disabled,
  isLoading,
  placeholder = 'e.g., I25.10, E11.9',
  label = 'Add ICD-10-CM Codes',
}: CodeInputProps) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onAdd();
    }
  };

  return (
    <div className="input-group">
      <label className="input-label">{label}</label>
      <div className="input-row">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="code-input"
          rows={1}
          disabled={disabled}
        />
        <button
          onClick={onAdd}
          disabled={!value.trim() || disabled || isLoading}
          className="add-btn"
        >
          {isLoading ? '...' : 'Add'}
        </button>
      </div>
    </div>
  );
}
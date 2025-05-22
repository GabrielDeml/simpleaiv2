import React, { useState } from 'react';
import './ModelBuilder.css';

const LAYER_TYPES = {
  DENSE: 'dense',
  DROPOUT: 'dropout',
  INPUT: 'input'
};

const ACTIVATION_FUNCTIONS = [
  'relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'elu', 'selu'
];

function LayerCard({ 
  layer, 
  onUpdate, 
  onDelete, 
  index, 
  isDragging = false,
  onDragStart,
  onDragEnd,
  isConnectable = true 
}) {
  const [showDetails, setShowDetails] = useState(false);

  const handleParameterChange = (param, value) => {
    onUpdate(index, { ...layer, [param]: value });
  };

  const getLayerIcon = () => {
    switch (layer.type) {
      case LAYER_TYPES.INPUT:
        return '📥';
      case LAYER_TYPES.DENSE:
        return '🧠';
      case LAYER_TYPES.DROPOUT:
        return '🎯';
      default:
        return '⚙️';
    }
  };

  const getLayerDescription = () => {
    switch (layer.type) {
      case LAYER_TYPES.INPUT:
        return `Input: ${layer.units} features`;
      case LAYER_TYPES.DENSE:
        return `Dense: ${layer.units} units, ${layer.activation}`;
      case LAYER_TYPES.DROPOUT:
        return `Dropout: ${(layer.rate * 100).toFixed(0)}% rate`;
      default:
        return 'Unknown layer';
    }
  };

  const renderParameterControls = () => {
    switch (layer.type) {
      case LAYER_TYPES.INPUT:
        return (
          <div className="parameter-group">
            <label>Input Size:</label>
            <input
              type="number"
              min="1"
              max="10000"
              value={layer.units}
              onChange={(e) => handleParameterChange('units', parseInt(e.target.value))}
            />
          </div>
        );

      case LAYER_TYPES.DENSE:
        return (
          <>
            <div className="parameter-group">
              <label>Units:</label>
              <input
                type="number"
                min="1"
                max="1000"
                value={layer.units}
                onChange={(e) => handleParameterChange('units', parseInt(e.target.value))}
              />
            </div>
            <div className="parameter-group">
              <label>Activation:</label>
              <select
                value={layer.activation}
                onChange={(e) => handleParameterChange('activation', e.target.value)}
              >
                {ACTIVATION_FUNCTIONS.map(func => (
                  <option key={func} value={func}>
                    {func.charAt(0).toUpperCase() + func.slice(1)}
                  </option>
                ))}
              </select>
            </div>
          </>
        );

      case LAYER_TYPES.DROPOUT:
        return (
          <div className="parameter-group">
            <label>Dropout Rate:</label>
            <input
              type="range"
              min="0"
              max="0.8"
              step="0.1"
              value={layer.rate}
              onChange={(e) => handleParameterChange('rate', parseFloat(e.target.value))}
            />
            <span className="parameter-value">{(layer.rate * 100).toFixed(0)}%</span>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div 
      className={`layer-card ${layer.type} ${isDragging ? 'dragging' : ''} ${showDetails ? 'expanded' : ''}`}
      draggable={isConnectable}
      onDragStart={(e) => {
        e.dataTransfer.setData('text/plain', JSON.stringify({ type: 'move', index }));
        onDragStart && onDragStart(index);
      }}
      onDragEnd={() => onDragEnd && onDragEnd()}
    >
      {/* Connection Points */}
      {isConnectable && (
        <>
          <div className="connection-point input-point"></div>
          <div className="connection-point output-point"></div>
        </>
      )}

      {/* Layer Header */}
      <div className="layer-header">
        <span className="layer-icon">{getLayerIcon()}</span>
        <div className="layer-info">
          <div className="layer-title">{layer.type.charAt(0).toUpperCase() + layer.type.slice(1)}</div>
          <div className="layer-description">{getLayerDescription()}</div>
        </div>
        <div className="layer-actions">
          <button
            className="action-button details-button"
            onClick={() => setShowDetails(!showDetails)}
            title="Toggle details"
          >
            {showDetails ? '🔼' : '🔽'}
          </button>
          {layer.type !== LAYER_TYPES.INPUT && (
            <button
              className="action-button delete-button"
              onClick={() => onDelete(index)}
              title="Delete layer"
            >
              🗑️
            </button>
          )}
        </div>
      </div>

      {/* Parameter Controls */}
      {showDetails && (
        <div className="layer-details">
          <div className="parameter-controls">
            {renderParameterControls()}
          </div>
          
          {/* Layer Information */}
          <div className="layer-info-panel">
            <h4>Layer Information</h4>
            <div className="info-item">
              <strong>Parameters:</strong> {layer.paramCount || 'Calculated at build'}
            </div>
            <div className="info-item">
              <strong>Output Shape:</strong> {layer.outputShape || 'Dynamic'}
            </div>
            {layer.type === LAYER_TYPES.DENSE && (
              <div className="info-item">
                <strong>Trainable:</strong> Yes
              </div>
            )}
            {layer.type === LAYER_TYPES.DROPOUT && (
              <div className="info-item">
                <strong>Training Only:</strong> Yes
              </div>
            )}
          </div>
        </div>
      )}

      {/* Visual Representation */}
      <div className="layer-visualization">
        {layer.type === LAYER_TYPES.DENSE && (
          <div className="dense-viz">
            {Array.from({ length: Math.min(layer.units, 6) }).map((_, i) => (
              <div key={i} className="neuron-dot"></div>
            ))}
            {layer.units > 6 && <div className="neuron-ellipsis">...</div>}
          </div>
        )}
        {layer.type === LAYER_TYPES.DROPOUT && (
          <div className="dropout-viz">
            <div className="dropout-mask" style={{ opacity: layer.rate }}></div>
          </div>
        )}
      </div>
    </div>
  );
}

export default LayerCard;
export { LAYER_TYPES, ACTIVATION_FUNCTIONS };
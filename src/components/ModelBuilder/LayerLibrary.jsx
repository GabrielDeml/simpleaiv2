import React from 'react';
import { LAYER_TYPES } from './LayerCard.jsx';
import './ModelBuilder.css';

const LAYER_TEMPLATES = [
  {
    type: LAYER_TYPES.DENSE,
    name: 'Dense Layer',
    description: 'Fully connected layer with adjustable neurons',
    icon: '🧠',
    defaultProps: {
      units: 128,
      activation: 'relu'
    },
    category: 'Core'
  },
  {
    type: LAYER_TYPES.DROPOUT,
    name: 'Dropout Layer',
    description: 'Regularization layer to prevent overfitting',
    icon: '🎯',
    defaultProps: {
      rate: 0.2
    },
    category: 'Regularization'
  }
];

const ARCHITECTURE_PRESETS = [
  {
    name: 'Simple Network',
    description: 'Basic 2-layer network for beginners',
    layers: [
      { type: LAYER_TYPES.INPUT, units: 784 },
      { type: LAYER_TYPES.DENSE, units: 128, activation: 'relu' },
      { type: LAYER_TYPES.DENSE, units: 10, activation: 'softmax' }
    ]
  },
  {
    name: 'Deep Network',
    description: 'Deeper network with dropout regularization',
    layers: [
      { type: LAYER_TYPES.INPUT, units: 784 },
      { type: LAYER_TYPES.DENSE, units: 256, activation: 'relu' },
      { type: LAYER_TYPES.DROPOUT, rate: 0.3 },
      { type: LAYER_TYPES.DENSE, units: 128, activation: 'relu' },
      { type: LAYER_TYPES.DROPOUT, rate: 0.2 },
      { type: LAYER_TYPES.DENSE, units: 64, activation: 'relu' },
      { type: LAYER_TYPES.DENSE, units: 10, activation: 'softmax' }
    ]
  },
  {
    name: 'Wide Network',
    description: 'Wider layers for learning complex patterns',
    layers: [
      { type: LAYER_TYPES.INPUT, units: 784 },
      { type: LAYER_TYPES.DENSE, units: 512, activation: 'relu' },
      { type: LAYER_TYPES.DROPOUT, rate: 0.2 },
      { type: LAYER_TYPES.DENSE, units: 256, activation: 'relu' },
      { type: LAYER_TYPES.DENSE, units: 10, activation: 'softmax' }
    ]
  }
];

function LayerLibrary({ onAddLayer, onLoadPreset, isVisible = true }) {
  const handleDragStart = (e, layerTemplate) => {
    e.dataTransfer.setData('text/plain', JSON.stringify({
      type: 'add',
      layerType: layerTemplate.type,
      defaultProps: layerTemplate.defaultProps
    }));
  };

  const handleAddLayer = (layerTemplate) => {
    const newLayer = {
      id: Date.now(),
      type: layerTemplate.type,
      ...layerTemplate.defaultProps
    };
    onAddLayer(newLayer);
  };

  if (!isVisible) return null;

  return (
    <div className="layer-library">
      <div className="library-section">
        <h3>Layer Library</h3>
        <div className="layer-templates">
          {LAYER_TEMPLATES.map((template, index) => (
            <div
              key={index}
              className="layer-template"
              draggable
              onDragStart={(e) => handleDragStart(e, template)}
              onClick={() => handleAddLayer(template)}
            >
              <div className="template-header">
                <span className="template-icon">{template.icon}</span>
                <div className="template-info">
                  <div className="template-name">{template.name}</div>
                  <div className="template-category">{template.category}</div>
                </div>
              </div>
              <div className="template-description">
                {template.description}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="library-section">
        <h3>Architecture Presets</h3>
        <div className="architecture-presets">
          {ARCHITECTURE_PRESETS.map((preset, index) => (
            <div
              key={index}
              className="preset-card"
              onClick={() => onLoadPreset(preset.layers)}
            >
              <div className="preset-header">
                <div className="preset-name">{preset.name}</div>
                <div className="preset-layers-count">{preset.layers.length} layers</div>
              </div>
              <div className="preset-description">
                {preset.description}
              </div>
              <div className="preset-architecture">
                {preset.layers.map((layer, layerIndex) => (
                  <div key={layerIndex} className="preset-layer">
                    {layer.type === LAYER_TYPES.INPUT ? '📥' : 
                     layer.type === LAYER_TYPES.DENSE ? '🧠' : '🎯'}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="library-section">
        <h3>Quick Actions</h3>
        <div className="quick-actions">
          <button 
            className="quick-action-btn"
            onClick={() => onLoadPreset(ARCHITECTURE_PRESETS[0].layers)}
          >
            🚀 Start Simple
          </button>
          <button 
            className="quick-action-btn"
            onClick={() => onLoadPreset([])}
          >
            🧹 Clear All
          </button>
        </div>
      </div>

      <div className="library-help">
        <h4>💡 Tips</h4>
        <ul>
          <li>Drag layers from the library to add them</li>
          <li>Click preset architectures to load them</li>
          <li>Input layer is automatically added</li>
          <li>Add softmax output layer for classification</li>
        </ul>
      </div>
    </div>
  );
}

export default LayerLibrary;
export { ARCHITECTURE_PRESETS, LAYER_TEMPLATES };
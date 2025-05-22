import React, { useState, useCallback, useRef } from 'react';
import LayerCard, { LAYER_TYPES } from './LayerCard.jsx';
import LayerLibrary from './LayerLibrary.jsx';
import './ModelBuilder.css';

function ModelBuilder({ onModelChange, initialLayers = null }) {
  const [layers, setLayers] = useState(initialLayers || [
    { id: 'input', type: LAYER_TYPES.INPUT, units: 784 }
  ]);
  const [draggedLayer, setDraggedLayer] = useState(null);
  const [showLibrary, setShowLibrary] = useState(true);
  const [validationErrors, setValidationErrors] = useState([]);
  const builderRef = useRef(null);

  // Validate the current architecture
  const validateArchitecture = useCallback((layerList) => {
    const errors = [];
    
    // Check for input layer
    if (layerList.length === 0 || layerList[0].type !== LAYER_TYPES.INPUT) {
      errors.push('Architecture must start with an input layer');
    }
    
    // Check for output layer
    const lastLayer = layerList[layerList.length - 1];
    if (lastLayer && lastLayer.type === LAYER_TYPES.DENSE && lastLayer.units !== 10) {
      errors.push('For MNIST classification, output layer should have 10 units');
    }
    
    if (lastLayer && lastLayer.type === LAYER_TYPES.DENSE && lastLayer.activation !== 'softmax' && layerList.length > 1) {
      errors.push('Output layer should typically use softmax activation for classification');
    }
    
    // Check for consecutive dropout layers
    for (let i = 0; i < layerList.length - 1; i++) {
      if (layerList[i].type === LAYER_TYPES.DROPOUT && layerList[i + 1].type === LAYER_TYPES.DROPOUT) {
        errors.push('Avoid consecutive dropout layers');
      }
    }
    
    // Check minimum architecture
    const denseLayerCount = layerList.filter(l => l.type === LAYER_TYPES.DENSE).length;
    if (denseLayerCount < 1) {
      errors.push('Architecture needs at least one dense layer');
    }
    
    setValidationErrors(errors);
    return errors.length === 0;
  }, []);

  // Update layers and notify parent
  const updateLayers = useCallback((newLayers) => {
    setLayers(newLayers);
    const isValid = validateArchitecture(newLayers);
    onModelChange && onModelChange(newLayers, isValid);
  }, [validateArchitecture, onModelChange]);

  // Add new layer
  const addLayer = useCallback((newLayer, insertIndex = null) => {
    const layerWithId = {
      ...newLayer,
      id: newLayer.id || `layer_${Date.now()}_${Math.random()}`
    };
    
    const newLayers = [...layers];
    if (insertIndex !== null) {
      newLayers.splice(insertIndex, 0, layerWithId);
    } else {
      newLayers.push(layerWithId);
    }
    
    updateLayers(newLayers);
  }, [layers, updateLayers]);

  // Update existing layer
  const updateLayer = useCallback((index, updatedLayer) => {
    const newLayers = [...layers];
    newLayers[index] = { ...updatedLayer, id: layers[index].id };
    updateLayers(newLayers);
  }, [layers, updateLayers]);

  // Delete layer
  const deleteLayer = useCallback((index) => {
    if (layers[index].type === LAYER_TYPES.INPUT) return; // Can't delete input layer
    
    const newLayers = layers.filter((_, i) => i !== index);
    updateLayers(newLayers);
  }, [layers, updateLayers]);

  // Move layer
  const moveLayer = useCallback((fromIndex, toIndex) => {
    if (fromIndex === toIndex) return;
    if (layers[fromIndex].type === LAYER_TYPES.INPUT && toIndex !== 0) return; // Input layer must stay first
    if (toIndex === 0 && layers[fromIndex].type !== LAYER_TYPES.INPUT) return; // Only input layer can be first
    
    const newLayers = [...layers];
    const [movedLayer] = newLayers.splice(fromIndex, 1);
    newLayers.splice(toIndex, 0, movedLayer);
    updateLayers(newLayers);
  }, [layers, updateLayers]);

  // Load preset architecture
  const loadPreset = useCallback((presetLayers) => {
    const layersWithIds = presetLayers.map((layer, index) => ({
      ...layer,
      id: `preset_${index}_${Date.now()}`
    }));
    updateLayers(layersWithIds);
  }, [updateLayers]);

  // Calculate architecture stats
  const getArchitectureStats = useCallback(() => {
    let totalParams = 0;
    let trainableParams = 0;
    
    for (let i = 1; i < layers.length; i++) {
      const currentLayer = layers[i];
      const prevLayer = layers[i - 1];
      
      if (currentLayer.type === LAYER_TYPES.DENSE) {
        const params = (prevLayer.units || 0) * currentLayer.units + currentLayer.units; // weights + biases
        totalParams += params;
        trainableParams += params;
      }
    }
    
    return {
      totalLayers: layers.length,
      denseLayers: layers.filter(l => l.type === LAYER_TYPES.DENSE).length,
      dropoutLayers: layers.filter(l => l.type === LAYER_TYPES.DROPOUT).length,
      totalParams,
      trainableParams
    };
  }, [layers]);

  // Drag and drop handlers
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const data = JSON.parse(e.dataTransfer.getData('text/plain'));
    
    if (data.type === 'add') {
      // Adding new layer from library
      const newLayer = {
        type: data.layerType,
        ...data.defaultProps
      };
      addLayer(newLayer);
    } else if (data.type === 'move') {
      // Moving existing layer
      const rect = builderRef.current.getBoundingClientRect();
      const dropY = e.clientY - rect.top;
      const layerHeight = 120; // Approximate layer card height
      const targetIndex = Math.floor(dropY / layerHeight);
      moveLayer(data.index, Math.max(0, Math.min(targetIndex, layers.length - 1)));
    }
  }, [addLayer, moveLayer]);

  const stats = getArchitectureStats();

  return (
    <div className="model-builder">
      <div className="builder-header">
        <h2>Neural Network Builder</h2>
        <div className="builder-controls">
          <button
            className={`toggle-library ${showLibrary ? 'active' : ''}`}
            onClick={() => setShowLibrary(!showLibrary)}
          >
            {showLibrary ? 'Hide' : 'Show'} Library
          </button>
          <div className="architecture-stats">
            <span className="stat">
              <strong>{stats.totalLayers}</strong> layers
            </span>
            <span className="stat">
              <strong>{stats.totalParams.toLocaleString()}</strong> parameters
            </span>
          </div>
        </div>
      </div>

      <div className="builder-content">
        {/* Layer Library */}
        {showLibrary && (
          <div className="library-panel">
            <LayerLibrary
              onAddLayer={addLayer}
              onLoadPreset={loadPreset}
              isVisible={showLibrary}
            />
          </div>
        )}

        {/* Model Architecture */}
        <div 
          className="architecture-panel"
          ref={builderRef}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <div className="architecture-header">
            <h3>Current Architecture</h3>
            {validationErrors.length > 0 && (
              <div className="validation-status error">
                ⚠️ {validationErrors.length} issue{validationErrors.length > 1 ? 's' : ''}
              </div>
            )}
            {validationErrors.length === 0 && layers.length > 1 && (
              <div className="validation-status valid">
                ✅ Architecture looks good
              </div>
            )}
          </div>

          <div className="layers-container">
            {layers.map((layer, index) => (
              <div key={layer.id} className="layer-container">
                <LayerCard
                  layer={layer}
                  index={index}
                  onUpdate={updateLayer}
                  onDelete={deleteLayer}
                  onDragStart={(idx) => setDraggedLayer(idx)}
                  onDragEnd={() => setDraggedLayer(null)}
                  isDragging={draggedLayer === index}
                />
                {index < layers.length - 1 && (
                  <div className="layer-connection">
                    <div className="connection-arrow">↓</div>
                  </div>
                )}
              </div>
            ))}
            
            {layers.length === 1 && (
              <div className="empty-architecture">
                <div className="empty-message">
                  <h4>🏗️ Start Building!</h4>
                  <p>Drag layers from the library or use a preset to begin</p>
                </div>
              </div>
            )}
          </div>

          {/* Validation Errors */}
          {validationErrors.length > 0 && (
            <div className="validation-errors">
              <h4>⚠️ Architecture Issues:</h4>
              <ul>
                {validationErrors.map((error, index) => (
                  <li key={index}>{error}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Architecture Summary */}
          {layers.length > 1 && (
            <div className="architecture-summary">
              <h4>📊 Architecture Summary</h4>
              <div className="summary-grid">
                <div className="summary-item">
                  <strong>Total Layers:</strong> {stats.totalLayers}
                </div>
                <div className="summary-item">
                  <strong>Dense Layers:</strong> {stats.denseLayers}
                </div>
                <div className="summary-item">
                  <strong>Dropout Layers:</strong> {stats.dropoutLayers}
                </div>
                <div className="summary-item">
                  <strong>Parameters:</strong> {stats.totalParams.toLocaleString()}
                </div>
                <div className="summary-item">
                  <strong>Trainable:</strong> {stats.trainableParams.toLocaleString()}
                </div>
                <div className="summary-item">
                  <strong>Memory Est:</strong> ~{Math.round(stats.totalParams * 4 / 1024)}KB
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ModelBuilder;
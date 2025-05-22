import * as tf from '@tensorflow/tfjs';
import { LAYER_TYPES } from '../components/ModelBuilder/LayerCard.jsx';
import { DATASET_CONFIG } from '../constants/mnistConfig.js';
import { Logger } from './logger.js';

/**
 * Dynamic Model Generation Utilities
 * Convert visual model builder designs into TensorFlow.js models
 */

/**
 * Create a TensorFlow.js model from layer configuration
 * @param {Array} layers - Array of layer configurations from ModelBuilder
 * @returns {tf.Sequential} Compiled TensorFlow.js model
 */
export function createDynamicModel(layers) {
  if (!layers || layers.length === 0) {
    throw new Error('No layers provided for model creation');
  }

  // Validate architecture
  validateArchitecture(layers);

  const model = tf.sequential();
  
  // Skip input layer (it's just for visualization)
  const actualLayers = layers.slice(1);
  
  Logger.info('Creating dynamic model with layers:', actualLayers.map(l => l.type));

  actualLayers.forEach((layer, index) => {
    const isFirstLayer = index === 0;
    const tfLayer = createTensorFlowLayer(layer, layers, isFirstLayer);
    
    if (tfLayer) {
      model.add(tfLayer);
      Logger.debug(`Added ${layer.type} layer:`, getLayerSummary(layer));
    }
  });

  // Compile the model
  const compileConfig = getCompileConfiguration(layers);
  model.compile(compileConfig);
  
  Logger.info('Dynamic model created and compiled successfully');
  return model;
}

/**
 * Validate the architecture before creating the model
 * @param {Array} layers - Layer configurations
 */
function validateArchitecture(layers) {
  const errors = [];

  // Check for input layer
  if (layers[0].type !== LAYER_TYPES.INPUT) {
    errors.push('First layer must be an input layer');
  }

  // Check for at least one dense layer
  const denseLayerCount = layers.filter(l => l.type === LAYER_TYPES.DENSE).length;
  if (denseLayerCount === 0) {
    errors.push('Architecture must contain at least one dense layer');
  }

  // Check output layer for MNIST (should be 10 units for classification)
  const lastLayer = layers[layers.length - 1];
  if (lastLayer.type === LAYER_TYPES.DENSE && lastLayer.units !== DATASET_CONFIG.NUM_CLASSES) {
    errors.push(`Output layer should have ${DATASET_CONFIG.NUM_CLASSES} units for MNIST classification`);
  }

  // Check for invalid layer sequences
  for (let i = 0; i < layers.length - 1; i++) {
    if (layers[i].type === LAYER_TYPES.DROPOUT && layers[i + 1].type === LAYER_TYPES.DROPOUT) {
      errors.push('Consecutive dropout layers are not recommended');
    }
  }

  if (errors.length > 0) {
    throw new Error(`Architecture validation failed: ${errors.join(', ')}`);
  }
}

/**
 * Create a TensorFlow.js layer from configuration
 * @param {Object} layerConfig - Layer configuration
 * @param {Array} allLayers - All layers for context
 * @param {boolean} isFirstLayer - Whether this is the first actual layer
 * @returns {tf.Layer} TensorFlow.js layer
 */
function createTensorFlowLayer(layerConfig, allLayers, isFirstLayer) {
  switch (layerConfig.type) {
    case LAYER_TYPES.DENSE:
      return createDenseLayer(layerConfig, allLayers, isFirstLayer);
      
    case LAYER_TYPES.DROPOUT:
      return createDropoutLayer(layerConfig);
      
    default:
      Logger.warn(`Unknown layer type: ${layerConfig.type}`);
      return null;
  }
}

/**
 * Create a dense (fully connected) layer
 * @param {Object} config - Layer configuration
 * @param {Array} allLayers - All layers for input shape inference
 * @param {boolean} isFirstLayer - Whether this is the first layer
 * @returns {tf.layers.Dense} Dense layer
 */
function createDenseLayer(config, allLayers, isFirstLayer) {
  const layerConfig = {
    units: config.units || 128,
    activation: config.activation || 'relu'
  };

  // Add input shape for the first layer
  if (isFirstLayer) {
    const inputLayer = allLayers[0];
    layerConfig.inputShape = [inputLayer.units || DATASET_CONFIG.IMAGE_SIZE];
  }

  return tf.layers.dense(layerConfig);
}

/**
 * Create a dropout layer
 * @param {Object} config - Layer configuration
 * @returns {tf.layers.Dropout} Dropout layer
 */
function createDropoutLayer(config) {
  return tf.layers.dropout({
    rate: config.rate || 0.2
  });
}

/**
 * Get compilation configuration based on architecture
 * @param {Array} layers - All layer configurations
 * @returns {Object} Compilation configuration
 */
function getCompileConfiguration(layers) {
  const lastLayer = layers[layers.length - 1];
  
  // Default configuration
  let config = {
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  };

  // Adjust based on output layer
  if (lastLayer.type === LAYER_TYPES.DENSE) {
    if (lastLayer.activation === 'sigmoid' && lastLayer.units === 1) {
      // Binary classification
      config.loss = 'binaryCrossentropy';
    } else if (lastLayer.activation === 'linear') {
      // Regression
      config.loss = 'meanSquaredError';
      config.metrics = ['mae'];
    }
  }

  return config;
}

/**
 * Get a summary of layer configuration for logging
 * @param {Object} layer - Layer configuration
 * @returns {Object} Layer summary
 */
function getLayerSummary(layer) {
  const summary = { type: layer.type };
  
  switch (layer.type) {
    case LAYER_TYPES.DENSE:
      summary.units = layer.units;
      summary.activation = layer.activation;
      break;
      
    case LAYER_TYPES.DROPOUT:
      summary.rate = layer.rate;
      break;
  }
  
  return summary;
}

/**
 * Calculate estimated parameters for the architecture
 * @param {Array} layers - Layer configurations
 * @returns {Object} Parameter statistics
 */
export function calculateModelParameters(layers) {
  let totalParams = 0;
  let trainableParams = 0;
  
  for (let i = 1; i < layers.length; i++) {
    const currentLayer = layers[i];
    const prevLayer = layers[i - 1];
    
    if (currentLayer.type === LAYER_TYPES.DENSE) {
      // Calculate weights and biases
      const prevUnits = prevLayer.units || DATASET_CONFIG.IMAGE_SIZE;
      const weights = prevUnits * currentLayer.units;
      const biases = currentLayer.units;
      const layerParams = weights + biases;
      
      totalParams += layerParams;
      trainableParams += layerParams;
    }
    // Dropout layers don't add parameters
  }
  
  return {
    total: totalParams,
    trainable: trainableParams,
    nonTrainable: totalParams - trainableParams
  };
}

/**
 * Estimate memory usage for the model
 * @param {Array} layers - Layer configurations
 * @returns {Object} Memory estimates in bytes
 */
export function estimateModelMemory(layers) {
  const params = calculateModelParameters(layers);
  
  // Each parameter is typically 4 bytes (float32)
  const modelMemory = params.total * 4;
  
  // Estimate activation memory (rough approximation)
  let maxActivationSize = 0;
  for (const layer of layers) {
    if (layer.type === LAYER_TYPES.DENSE) {
      const activationSize = layer.units * 4; // 4 bytes per activation
      maxActivationSize = Math.max(maxActivationSize, activationSize);
    }
  }
  
  return {
    model: modelMemory,
    activations: maxActivationSize,
    total: modelMemory + maxActivationSize,
    formatted: {
      model: formatBytes(modelMemory),
      activations: formatBytes(maxActivationSize),
      total: formatBytes(modelMemory + maxActivationSize)
    }
  };
}

/**
 * Format bytes to human readable format
 * @param {number} bytes - Number of bytes
 * @returns {string} Formatted string
 */
function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Generate a text summary of the architecture
 * @param {Array} layers - Layer configurations
 * @returns {string} Architecture summary
 */
export function generateArchitectureSummary(layers) {
  const params = calculateModelParameters(layers);
  const memory = estimateModelMemory(layers);
  
  let summary = 'Neural Network Architecture:\n';
  summary += `- Total Layers: ${layers.length}\n`;
  summary += `- Parameters: ${params.total.toLocaleString()}\n`;
  summary += `- Memory: ${memory.formatted.total}\n\n`;
  
  summary += 'Layer Details:\n';
  layers.forEach((layer, index) => {
    summary += `${index + 1}. ${layer.type.charAt(0).toUpperCase() + layer.type.slice(1)}`;
    
    if (layer.type === LAYER_TYPES.INPUT) {
      summary += ` (${layer.units} features)`;
    } else if (layer.type === LAYER_TYPES.DENSE) {
      summary += ` (${layer.units} units, ${layer.activation})`;
    } else if (layer.type === LAYER_TYPES.DROPOUT) {
      summary += ` (${(layer.rate * 100).toFixed(0)}% rate)`;
    }
    
    summary += '\n';
  });
  
  return summary;
}

/**
 * Validate if the current architecture is suitable for MNIST
 * @param {Array} layers - Layer configurations
 * @returns {Object} Validation result with suggestions
 */
export function validateMNISTArchitecture(layers) {
  const result = {
    isValid: true,
    warnings: [],
    suggestions: [],
    errors: []
  };
  
  // Check input layer
  if (layers[0].type !== LAYER_TYPES.INPUT || layers[0].units !== DATASET_CONFIG.IMAGE_SIZE) {
    result.errors.push(`Input layer should have ${DATASET_CONFIG.IMAGE_SIZE} units for MNIST`);
    result.isValid = false;
  }
  
  // Check output layer
  const lastLayer = layers[layers.length - 1];
  if (lastLayer.type === LAYER_TYPES.DENSE) {
    if (lastLayer.units !== DATASET_CONFIG.NUM_CLASSES) {
      result.errors.push(`Output layer should have ${DATASET_CONFIG.NUM_CLASSES} units for MNIST classification`);
      result.isValid = false;
    }
    
    if (lastLayer.activation !== 'softmax') {
      result.warnings.push('Output layer should typically use softmax activation for classification');
    }
  }
  
  // Check architecture complexity
  const params = calculateModelParameters(layers);
  if (params.total > 1000000) {
    result.warnings.push('Very large model - may be slow to train');
  } else if (params.total < 1000) {
    result.warnings.push('Very small model - may underfit the data');
  }
  
  // Architecture suggestions
  const denseLayerCount = layers.filter(l => l.type === LAYER_TYPES.DENSE).length;
  if (denseLayerCount === 1) {
    result.suggestions.push('Consider adding hidden layers for better performance');
  }
  
  const dropoutLayerCount = layers.filter(l => l.type === LAYER_TYPES.DROPOUT).length;
  if (dropoutLayerCount === 0 && denseLayerCount > 2) {
    result.suggestions.push('Consider adding dropout layers to prevent overfitting');
  }
  
  return result;
}
<script lang="ts">
  /**
   * ModelSummary Component
   * 
   * Purpose: Displays real-time statistics about the neural network model,
   * including total parameters, trainable parameters, layer count, and
   * output shape. Automatically recalculates when layers change.
   * 
   * Key features:
   * - Dynamic parameter counting using TensorFlow.js
   * - Temporary model building for accurate calculations
   * - Smart number formatting (K/M suffixes)
   * - Error handling for invalid layer configurations
   * - Memory-efficient (disposes temporary models)
   */
  
  import { layers } from '$lib/nn-designer/stores';
  import * as tf from '@tensorflow/tfjs';
  
  // Summary statistics displayed to user
  let modelSummary = {
    totalParams: 0,
    trainableParams: 0,
    layerCount: 0,
    outputShape: [] as number[]
  };
  
  // Reactive: Recalculate summary whenever layers change
  $: if ($layers.length > 0) {
    calculateModelSummary();
  }
  
  /**
   * Calculates model statistics by building a temporary TensorFlow.js model
   * - Builds the model from current layer configuration
   * - Counts parameters using TF.js built-in methods
   * - Extracts output shape for display
   * - Properly disposes model to prevent memory leaks
   */
  async function calculateModelSummary() {
    try {
      const model = await buildModel();
      if (model) {
        const totalParams = model.countParams();
        const outputShape = model.outputs[0].shape;
        
        modelSummary = {
          totalParams,
          trainableParams: totalParams, // All params are trainable in our case
          layerCount: $layers.length,
          outputShape: outputShape.slice(1) // Remove batch dimension (first dim)
        };
        
        // Clean up to prevent memory leaks
        model.dispose();
      }
    } catch (error) {
      console.error('Error calculating model summary:', error);
    }
  }
  
  /**
   * Builds a temporary TensorFlow.js model from the current layer configuration
   * @returns A TF.js Sequential model or null if no valid layers
   * 
   * - Handles input shape propagation from input layer
   * - Adds each layer with its configured parameters
   * - Skips invalid layers and continues building
   * - Input shape is only specified for the first actual layer
   */
  async function buildModel() {
    if ($layers.length === 0) return null;
    
    const model = tf.sequential();
    let inputShape: number[] | undefined;
    
    // Extract input shape from input layer if present
    const inputLayer = $layers.find(l => l.type === 'input');
    if (inputLayer) {
      inputShape = inputLayer.params.shape;
    }
    
    // Build model layer by layer
    for (const layer of $layers) {
      try {
        switch (layer.type) {
          case 'input':
            // Input layer doesn't add a TF.js layer, just defines shape
            break;
            
          case 'dense':
            model.add(tf.layers.dense({
              units: layer.params.units,
              activation: layer.params.activation,
              useBias: layer.params.useBias,
              kernelInitializer: layer.params.kernelInitializer,
              // Input shape only needed for first layer
              inputShape: model.layers.length === 0 ? inputShape || [784] : undefined
            }));
            break;
            
          case 'conv2d':
            model.add(tf.layers.conv2d({
              filters: layer.params.filters,
              kernelSize: layer.params.kernelSize,
              strides: layer.params.strides,
              padding: layer.params.padding,
              activation: layer.params.activation,
              useBias: layer.params.useBias,
              // Conv2d might be first layer, needs proper shape
              inputShape: model.layers.length === 0 ? layer.params.shape || [28, 28, 1] : undefined
            }));
            break;
            
          case 'maxpooling2d':
            model.add(tf.layers.maxPooling2d({
              poolSize: layer.params.poolSize,
              strides: layer.params.strides,
              padding: layer.params.padding
            }));
            break;
            
          case 'dropout':
            model.add(tf.layers.dropout({
              rate: layer.params.rate
            }));
            break;
            
          case 'flatten':
            model.add(tf.layers.flatten());
            break;
        }
      } catch (error) {
        // Log but continue building to show partial stats
        console.error(`Error adding layer ${layer.type}:`, error);
      }
    }
    
    return model.layers.length > 0 ? model : null;
  }
  
  /**
   * Formats large numbers with K/M suffixes for readability
   * @param num - The number to format
   * @returns Formatted string (e.g., "1.2M", "15.3K", "999")
   * 
   * - Numbers >= 1M show as "X.XM"
   * - Numbers >= 1K show as "X.XK"
   * - Smaller numbers use locale formatting with commas
   */
  function formatNumber(num: number): string {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toLocaleString();
  }
</script>

<!-- Model summary card container -->
<div class="model-summary">
  <h2>Model Summary</h2>
  
  <!-- Statistics grid -->
  <div class="summary-stats">
    <!-- Total parameter count -->
    <div class="stat">
      <span class="label">Total Parameters:</span>
      <span class="value">{formatNumber(modelSummary.totalParams)}</span>
    </div>
    
    <!-- Trainable parameter count (currently same as total) -->
    <div class="stat">
      <span class="label">Trainable:</span>
      <span class="value">{formatNumber(modelSummary.trainableParams)}</span>
    </div>
    
    <!-- Number of layers in the network -->
    <div class="stat">
      <span class="label">Layers:</span>
      <span class="value">{modelSummary.layerCount}</span>
    </div>
    
    <!-- Output tensor shape (batch dimension shown as 'None') -->
    <div class="stat">
      <span class="label">Output Shape:</span>
      <span class="value">
        {modelSummary.outputShape.length > 0 
          ? `(None, ${modelSummary.outputShape.join(', ')})`
          : 'N/A'}
      </span>
    </div>
  </div>
</div>

<style>
  .model-summary {
    background: #0f0f0f;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }
  
  h2 {
    font-size: 16px;
    font-weight: 500;
    margin: 0 0 16px 0;
  }
  
  .summary-stats {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
  }
  
  .label {
    color: #737373;
  }
  
  .value {
    color: #ffffff;
    font-weight: 500;
  }
</style>
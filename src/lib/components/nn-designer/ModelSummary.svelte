<script lang="ts">
  import { layers } from '$lib/nn-designer/stores';
  import * as tf from '@tensorflow/tfjs';
  
  let modelSummary = {
    totalParams: 0,
    trainableParams: 0,
    layerCount: 0,
    outputShape: [] as number[]
  };
  
  $: if ($layers.length > 0) {
    calculateModelSummary();
  }
  
  async function calculateModelSummary() {
    try {
      const model = await buildModel();
      if (model) {
        const totalParams = model.countParams();
        const outputShape = model.outputs[0].shape;
        
        modelSummary = {
          totalParams,
          trainableParams: totalParams,
          layerCount: $layers.length,
          outputShape: outputShape.slice(1) // Remove batch dimension
        };
        
        model.dispose();
      }
    } catch (error) {
      console.error('Error calculating model summary:', error);
    }
  }
  
  async function buildModel() {
    if ($layers.length === 0) return null;
    
    const model = tf.sequential();
    let inputShape: number[] | undefined;
    
    // Find input shape from input layer
    const inputLayer = $layers.find(l => l.type === 'input');
    if (inputLayer) {
      inputShape = inputLayer.params.shape;
    }
    
    for (const layer of $layers) {
      try {
        switch (layer.type) {
          case 'input':
            // Input layer is implicit in the first layer
            break;
            
          case 'dense':
            model.add(tf.layers.dense({
              units: layer.params.units,
              activation: layer.params.activation,
              useBias: layer.params.useBias,
              kernelInitializer: layer.params.kernelInitializer,
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
        console.error(`Error adding layer ${layer.type}:`, error);
      }
    }
    
    return model.layers.length > 0 ? model : null;
  }
  
  function formatNumber(num: number): string {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toLocaleString();
  }
</script>

<div class="model-summary">
  <h2>Model Summary</h2>
  
  <div class="summary-stats">
    <div class="stat">
      <span class="label">Total Parameters:</span>
      <span class="value">{formatNumber(modelSummary.totalParams)}</span>
    </div>
    
    <div class="stat">
      <span class="label">Trainable:</span>
      <span class="value">{formatNumber(modelSummary.trainableParams)}</span>
    </div>
    
    <div class="stat">
      <span class="label">Layers:</span>
      <span class="value">{modelSummary.layerCount}</span>
    </div>
    
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
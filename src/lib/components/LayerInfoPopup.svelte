<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { LayerType } from '$lib/nn-designer/types';
  import { layerDefinitions } from '$lib/nn-designer/layerDefinitions';
  
  export let layerType: LayerType | null = null;
  export let isOpen = false;
  
  const dispatch = createEventDispatcher();
  
  $: layerDef = layerType ? layerDefinitions[layerType] : null;
  
  function close() {
    dispatch('close');
  }
  
  function handleBackdropClick(e: MouseEvent) {
    if (e.target === e.currentTarget) {
      close();
    }
  }
</script>

{#if isOpen && layerDef}
  <div 
    class="popup-backdrop" 
    on:click={handleBackdropClick}
    on:keydown={(e) => e.key === 'Escape' && close()}
    role="dialog"
    aria-modal="true"
    aria-labelledby="popup-title"
    tabindex="-1"
  >
    <div class="popup-content">
      <div class="popup-header" style="background: linear-gradient(135deg, {layerDef.color}22, {layerDef.color}11)">
        <div class="header-left">
          <div class="layer-icon" style="background: {layerDef.color}22; color: {layerDef.color}">
            {layerDef.icon}
          </div>
          <h2 id="popup-title">{layerDef.displayName}</h2>
        </div>
        <button class="close-button" on:click={close} aria-label="Close dialog">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M5 5L15 15M5 15L15 5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </button>
      </div>
      
      <div class="popup-body">
        <section class="info-section">
          <h3>Overview</h3>
          <p>{layerDef.description}</p>
          {#if layerDef.learnMore}
            <p class="learn-more">{layerDef.learnMore}</p>
          {/if}
        </section>
        
        <section class="info-section">
          <h3>Parameters</h3>
          <div class="params-grid">
            {#each Object.entries(layerDef.defaultParams) as [param, value]}
              <div class="param-item">
                <span class="param-name">{param}:</span>
                <span class="param-value">{JSON.stringify(value)}</span>
              </div>
            {/each}
          </div>
        </section>
        
        {#if layerType === 'dense'}
          <section class="info-section">
            <h3>How It Works</h3>
            <p>Each neuron in a dense layer:</p>
            <ul>
              <li>Receives input from every neuron in the previous layer</li>
              <li>Multiplies each input by a learned weight</li>
              <li>Adds a bias term</li>
              <li>Applies an activation function</li>
            </ul>
            <div class="formula">
              output = activation(inputs × weights + bias)
            </div>
          </section>
          
          <section class="info-section">
            <h3>Best Practices</h3>
            <ul>
              <li>Start with 64-128 units for hidden layers</li>
              <li>Use ReLU activation for hidden layers</li>
              <li>Match output units to your number of classes</li>
              <li>Consider dropout after dense layers to prevent overfitting</li>
            </ul>
          </section>
        {/if}
        
        {#if layerType === 'conv2d'}
          <section class="info-section">
            <h3>How It Works</h3>
            <p>Convolutional layers scan across images with small filters to detect features:</p>
            <ul>
              <li>Each filter slides across the input image</li>
              <li>Performs element-wise multiplication and sums the results</li>
              <li>Creates feature maps showing where patterns are detected</li>
              <li>Multiple filters learn different features (edges, textures, shapes)</li>
            </ul>
          </section>
          
          <section class="info-section">
            <h3>Common Configurations</h3>
            <ul>
              <li><strong>32-64 filters</strong> for early layers (simple features)</li>
              <li><strong>128-256 filters</strong> for deeper layers (complex features)</li>
              <li><strong>3×3 kernel</strong> is most common (good balance)</li>
              <li><strong>Padding 'same'</strong> preserves spatial dimensions</li>
              <li><strong>Stride 1</strong> for normal convolution, 2 for downsampling</li>
            </ul>
          </section>
        {/if}
        
        {#if layerType === 'maxpooling2d'}
          <section class="info-section">
            <h3>How It Works</h3>
            <p>Max pooling reduces spatial dimensions while keeping important features:</p>
            <ul>
              <li>Divides input into pooling windows (e.g., 2×2)</li>
              <li>Takes the maximum value from each window</li>
              <li>Reduces computation and parameters</li>
              <li>Provides translation invariance</li>
            </ul>
          </section>
          
          <section class="info-section">
            <h3>Usage Tips</h3>
            <ul>
              <li>Place after Conv2D layers to downsample</li>
              <li>2×2 pooling with stride 2 halves dimensions</li>
              <li>Helps prevent overfitting</li>
              <li>Don't use too many - you'll lose spatial information</li>
            </ul>
          </section>
        {/if}
        
        {#if layerType === 'dropout'}
          <section class="info-section">
            <h3>How It Works</h3>
            <p>Dropout randomly "turns off" neurons during training:</p>
            <ul>
              <li>Each training step randomly disables neurons</li>
              <li>Forces network to learn redundant representations</li>
              <li>Prevents co-adaptation of neurons</li>
              <li>Disabled during inference (all neurons active)</li>
            </ul>
          </section>
          
          <section class="info-section">
            <h3>When to Use</h3>
            <ul>
              <li>After Dense or Conv2D layers</li>
              <li>When model overfits (high train accuracy, low validation)</li>
              <li>Rate 0.2-0.3 for Conv layers</li>
              <li>Rate 0.4-0.5 for Dense layers</li>
              <li>Never use after the output layer</li>
            </ul>
          </section>
        {/if}
        
        {#if layerType === 'flatten'}
          <section class="info-section">
            <h3>How It Works</h3>
            <p>Flatten reshapes multi-dimensional data into a 1D vector:</p>
            <ul>
              <li>Preserves all data, just changes the shape</li>
              <li>Required between Conv2D and Dense layers</li>
              <li>Example: 28×28×32 → 25,088 elements</li>
            </ul>
          </section>
          
          <section class="info-section">
            <h3>Usage</h3>
            <p>Place between convolutional blocks and dense layers. No parameters to configure!</p>
          </section>
        {/if}
        
        {#if layerType === 'input'}
          <section class="info-section">
            <h3>Data Formats</h3>
            <ul>
              <li><strong>[28, 28]</strong> - Grayscale images (MNIST)</li>
              <li><strong>[32, 32, 3]</strong> - Color images (CIFAR-10)</li>
              <li><strong>[784]</strong> - Flattened MNIST</li>
              <li><strong>[10]</strong> - Simple numeric features</li>
            </ul>
          </section>
          
          <section class="info-section">
            <h3>Important Notes</h3>
            <ul>
              <li>Must match your dataset exactly</li>
              <li>Batch dimension is handled automatically</li>
              <li>For images: [height, width, channels]</li>
              <li>For tabular data: [number_of_features]</li>
            </ul>
          </section>
        {/if}
        
        {#if layerType === 'output'}
          <section class="info-section">
            <h3>Activation Functions</h3>
            <ul>
              <li><strong>Softmax</strong> - Multi-class classification (probabilities sum to 1)</li>
              <li><strong>Sigmoid</strong> - Binary classification (0-1 probability)</li>
              <li><strong>Linear</strong> - Regression (continuous values)</li>
            </ul>
          </section>
          
          <section class="info-section">
            <h3>Units Configuration</h3>
            <ul>
              <li>Classification: units = number of classes</li>
              <li>Binary classification: 1 unit with sigmoid</li>
              <li>Regression: units = number of outputs</li>
            </ul>
          </section>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .popup-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    animation: fadeIn 0.2s ease-out;
  }
  
  .popup-content {
    background: #0f0f0f;
    border: 1px solid #262626;
    border-radius: 12px;
    width: 90%;
    max-width: 600px;
    max-height: 85vh;
    display: flex;
    flex-direction: column;
    animation: slideIn 0.3s ease-out;
  }
  
  .popup-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    border-bottom: 1px solid #262626;
    border-radius: 12px 12px 0 0;
  }
  
  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  
  .layer-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 14px;
  }
  
  .popup-header h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 500;
  }
  
  .close-button {
    background: none;
    border: none;
    color: #737373;
    cursor: pointer;
    padding: 4px;
    transition: color 0.2s;
  }
  
  .close-button:hover {
    color: #ffffff;
  }
  
  .popup-body {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }
  
  .info-section {
    margin-bottom: 24px;
  }
  
  .info-section:last-child {
    margin-bottom: 0;
  }
  
  .info-section h3 {
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
    margin: 0 0 12px 0;
  }
  
  .info-section p {
    color: #a3a3a3;
    line-height: 1.6;
    margin: 0 0 12px 0;
  }
  
  .info-section p:last-child {
    margin-bottom: 0;
  }
  
  .learn-more {
    color: #737373;
    font-style: italic;
    font-size: 14px;
  }
  
  .info-section ul {
    color: #a3a3a3;
    margin: 0;
    padding-left: 20px;
  }
  
  .info-section li {
    margin-bottom: 8px;
    line-height: 1.5;
  }
  
  .info-section li strong {
    color: #e5e5e5;
  }
  
  .params-grid {
    background: #171717;
    border: 1px solid #262626;
    border-radius: 6px;
    padding: 12px;
  }
  
  .param-item {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 13px;
  }
  
  .param-name {
    color: #737373;
  }
  
  .param-value {
    color: #22c55e;
    font-family: 'Monaco', 'Consolas', monospace;
  }
  
  .formula {
    background: #171717;
    border: 1px solid #262626;
    border-radius: 6px;
    padding: 12px;
    font-family: 'Monaco', 'Consolas', monospace;
    color: #22c55e;
    text-align: center;
    margin-top: 12px;
  }
  
  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
  
  @keyframes slideIn {
    from {
      transform: translateY(20px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }
</style>
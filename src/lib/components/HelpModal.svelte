<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  
  export let isOpen = false;
  
  const dispatch = createEventDispatcher();
  
  let activeTab = 'quickstart';
  let searchQuery = '';
  
  const tabs = [
    { id: 'quickstart', label: 'Quick Start' },
    { id: 'basics', label: 'AI Basics' },
    { id: 'layers', label: 'Layer Types' },
    { id: 'training', label: 'Training' },
    { id: 'interface', label: 'Interface' },
    { id: 'tips', label: 'Tips & Tricks' }
  ];
  
  function close() {
    dispatch('close');
  }
</script>

{#if isOpen}
  <div 
    class="modal-backdrop" 
    on:click={close}
    on:keydown={(e) => e.key === 'Escape' && close()}
    role="dialog"
    aria-modal="true"
    aria-labelledby="help-modal-title"
    tabindex="-1"
  >
    <div class="modal">
      <div class="modal-header">
        <div class="header-content">
          <h2 id="help-modal-title">Neural Network Designer Help</h2>
          <p class="subtitle">Everything you need to build and train neural networks</p>
        </div>
        <button class="close-button" on:click={close} aria-label="Close help">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M6 6L18 18M6 18L18 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </button>
      </div>
      
      <div class="tabs">
        {#each tabs as tab}
          <button 
            class="tab" 
            class:active={activeTab === tab.id}
            on:click={() => activeTab = tab.id}
          >
            {tab.label}
          </button>
        {/each}
      </div>
      
      <div class="content">
        {#if activeTab === 'quickstart'}
          <div class="section">
            <h3>🚀 Getting Started in 5 Minutes</h3>
            
            <div class="quickstart-step">
              <div class="step-number">1</div>
              <div class="step-content">
                <h4>Choose a Dataset</h4>
                <p>Select MNIST from the dataset dropdown (bottom left). This is a collection of handwritten digits (0-9) perfect for beginners.</p>
              </div>
            </div>
            
            <div class="quickstart-step">
              <div class="step-number">2</div>
              <div class="step-content">
                <h4>Build Your Network</h4>
                <p>Drag layers from the left panel to the canvas:</p>
                <ul>
                  <li>Start with an <strong>Input Layer</strong> (already there for MNIST)</li>
                  <li>Add a <strong>Flatten</strong> layer to convert 2D images to 1D</li>
                  <li>Add 2-3 <strong>Dense</strong> layers (try 128, 64, then 10 neurons)</li>
                  <li>Add <strong>Dropout</strong> layers between Dense layers (0.2 rate)</li>
                </ul>
              </div>
            </div>
            
            <div class="quickstart-step">
              <div class="step-number">3</div>
              <div class="step-content">
                <h4>Configure Training</h4>
                <p>In the bottom-right panel:</p>
                <ul>
                  <li>Epochs: 10</li>
                  <li>Batch Size: 32</li>
                  <li>Learning Rate: 0.001</li>
                  <li>Optimizer: Adam</li>
                </ul>
              </div>
            </div>
            
            <div class="quickstart-step">
              <div class="step-number">4</div>
              <div class="step-content">
                <h4>Train Your Model</h4>
                <p>Click the green <strong>Run</strong> button and watch your model learn! The training progress window will show accuracy improving over time.</p>
              </div>
            </div>
            
            <div class="info-box">
              <h4>💡 Quick Tips</h4>
              <ul>
                <li>Click on any layer to edit its properties</li>
                <li>Drag layers to reorder them</li>
                <li>Use the Templates dropdown for pre-built architectures</li>
                <li>Save your model to browser storage with the Save button</li>
              </ul>
            </div>
          </div>
        {/if}
        
        {#if activeTab === 'basics'}
          <div class="section">
            <h3>What is a Neural Network?</h3>
            <p>A neural network is a computer system inspired by the human brain. It learns patterns from data by adjusting connections between artificial neurons.</p>
            
            <h3>Key Concepts</h3>
            <div class="concept">
              <h4>🧠 Neurons</h4>
              <p>Basic units that receive inputs, process them, and produce outputs. Each neuron has weights that determine how much each input matters.</p>
            </div>
            
            <div class="concept">
              <h4>🔗 Layers</h4>
              <p>Neurons are organized in layers. Data flows from input layer → hidden layers → output layer.</p>
            </div>
            
            <div class="concept">
              <h4>📊 Training</h4>
              <p>The network learns by seeing examples and adjusting weights to minimize prediction errors.</p>
            </div>
            
            <div class="concept">
              <h4>🎯 Loss Function</h4>
              <p>Measures how wrong the predictions are. The goal is to minimize this value.</p>
            </div>
          </div>
        {/if}
        
        {#if activeTab === 'layers'}
          <div class="section">
            <h3>Understanding Layer Types</h3>
            
            <div class="layer-guide">
              <h4>Input Layer</h4>
              <p>Where data enters the network. Must match your data shape exactly.</p>
              <div class="example">Example: 28×28 for MNIST digit images</div>
            </div>
            
            <div class="layer-guide">
              <h4>Dense (Fully Connected)</h4>
              <p>Every neuron connects to all neurons in the previous layer. Good for learning general patterns.</p>
              <div class="example">Use for: Classification, regression, feature extraction</div>
            </div>
            
            <div class="layer-guide">
              <h4>Conv2D (Convolutional)</h4>
              <p>Scans images with small filters to detect features like edges and shapes. Essential for computer vision.</p>
              <div class="example">Use for: Image recognition, object detection</div>
            </div>
            
            <div class="layer-guide">
              <h4>MaxPooling2D</h4>
              <p>Reduces image size by taking maximum values in regions. Makes the network faster and more robust.</p>
              <div class="example">Use after: Conv2D layers to downsample</div>
            </div>
            
            <div class="layer-guide">
              <h4>Dropout</h4>
              <p>Randomly disables neurons during training to prevent overfitting (memorizing instead of learning).</p>
              <div class="example">Use when: Model performs well on training but poorly on test data</div>
            </div>
            
            <div class="layer-guide">
              <h4>Flatten</h4>
              <p>Converts multi-dimensional data to 1D. Required between Conv2D and Dense layers.</p>
              <div class="example">Use when: Transitioning from image processing to classification</div>
            </div>
          </div>
        {/if}
        
        {#if activeTab === 'training'}
          <div class="section">
            <h3>Training Your Network</h3>
            
            <div class="training-guide">
              <h4>📈 Epochs</h4>
              <p>One epoch = one pass through the entire dataset. More epochs allow more learning but risk overfitting.</p>
              <div class="tip">Start with 10-20, increase if loss keeps decreasing</div>
            </div>
            
            <div class="training-guide">
              <h4>📦 Batch Size</h4>
              <p>Number of examples processed together. Smaller = more accurate gradients but slower. Larger = faster but noisier.</p>
              <div class="tip">Try 32 or 64 for a good balance</div>
            </div>
            
            <div class="training-guide">
              <h4>🎚️ Learning Rate</h4>
              <p>How big steps to take when updating weights. Too high = unstable, too low = slow learning.</p>
              <div class="tip">0.001 is a safe starting point</div>
            </div>
            
            <div class="training-guide">
              <h4>🎯 Loss Functions</h4>
              <p><strong>Categorical Crossentropy:</strong> For multi-class classification (e.g., digits 0-9)</p>
              <p><strong>Binary Crossentropy:</strong> For yes/no classification</p>
              <p><strong>Mean Squared Error:</strong> For predicting continuous values</p>
            </div>
            
            <div class="training-guide">
              <h4>⚡ Optimizers</h4>
              <p><strong>Adam:</strong> Adaptive learning rate, works well in most cases</p>
              <p><strong>SGD:</strong> Simple and reliable, good for fine-tuning</p>
              <p><strong>RMSprop:</strong> Good for recurrent networks</p>
            </div>
          </div>
        {/if}
        
        {#if activeTab === 'interface'}
          <div class="section">
            <h3>Using the Interface</h3>
            
            <div class="interface-section">
              <h4>🎨 Layer Palette (Left Panel)</h4>
              <p>Available layer types you can drag onto the canvas. Each layer has specific use cases:</p>
              <ul>
                <li><strong>Input:</strong> Define your data shape</li>
                <li><strong>Dense:</strong> General-purpose processing</li>
                <li><strong>Conv2D:</strong> Image feature detection</li>
                <li><strong>Pooling:</strong> Reduce dimensions</li>
                <li><strong>Dropout:</strong> Prevent overfitting</li>
                <li><strong>Flatten:</strong> Convert to 1D</li>
              </ul>
            </div>
            
            <div class="interface-section">
              <h4>🔗 Network Canvas (Center)</h4>
              <p>Your visual network builder:</p>
              <ul>
                <li>Drag layers from the palette to add them</li>
                <li>Click a layer to select and edit properties</li>
                <li>Drag layers up/down to reorder</li>
                <li>Click the × button to remove a layer</li>
                <li>Green connections show data flow</li>
              </ul>
            </div>
            
            <div class="interface-section">
              <h4>⚙️ Properties Panel (Right)</h4>
              <p>Configure selected layers and training:</p>
              <ul>
                <li><strong>Layer Properties:</strong> Edit parameters like neurons, filters, activation</li>
                <li><strong>Model Summary:</strong> See total parameters and output shapes</li>
                <li><strong>Training Config:</strong> Set hyperparameters for learning</li>
              </ul>
            </div>
            
            <div class="interface-section">
              <h4>🎛️ Toolbar (Top)</h4>
              <div class="toolbar-items">
                <div class="toolbar-item">
                  <span class="button-preview">▶ Run</span>
                  <span>Start/stop training</span>
                </div>
                <div class="toolbar-item">
                  <span class="button-preview">📊 Show Training</span>
                  <span>View live metrics</span>
                </div>
                <div class="toolbar-item">
                  <span class="button-preview">Save</span>
                  <span>Save to browser</span>
                </div>
                <div class="toolbar-item">
                  <span class="button-preview">Export</span>
                  <span>Download as JSON or Colab</span>
                </div>
                <div class="toolbar-item">
                  <span class="button-preview">Clear</span>
                  <span>Reset to empty model</span>
                </div>
              </div>
            </div>
            
            <div class="interface-section">
              <h4>⌨️ Keyboard Shortcuts</h4>
              <div class="shortcuts">
                <div class="shortcut">
                  <kbd>Delete</kbd> <span>Remove selected layer</span>
                </div>
                <div class="shortcut">
                  <kbd>Esc</kbd> <span>Deselect layer</span>
                </div>
                <div class="shortcut">
                  <kbd>Ctrl/Cmd + S</kbd> <span>Save model</span>
                </div>
              </div>
            </div>
          </div>
        {/if}
        
        {#if activeTab === 'tips'}
          <div class="section">
            <h3>Tips for Success</h3>
            
            <div class="tip-card">
              <h4>🏗️ Start Simple</h4>
              <p>Begin with a basic architecture and add complexity only if needed. A simple model that works is better than a complex one that doesn't.</p>
            </div>
            
            <div class="tip-card">
              <h4>📊 Watch the Metrics</h4>
              <p>If training loss decreases but validation loss increases, you're overfitting. Add dropout or reduce model size.</p>
            </div>
            
            <div class="tip-card">
              <h4>🔄 Experiment</h4>
              <p>Try different architectures, learning rates, and batch sizes. Neural networks often require experimentation.</p>
            </div>
            
            <div class="tip-card">
              <h4>📈 Common Architectures</h4>
              <p><strong>For MNIST:</strong> Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Output</p>
              <p><strong>For simple classification:</strong> Input → Dense → Dropout → Dense → Output</p>
            </div>
            
            <div class="tip-card">
              <h4>⚠️ Common Issues</h4>
              <p><strong>Loss = NaN:</strong> Learning rate too high</p>
              <p><strong>Loss not decreasing:</strong> Learning rate too low or model too simple</p>
              <p><strong>Very low accuracy:</strong> Check data preprocessing and output activation</p>
              <p><strong>Overfitting:</strong> Add dropout layers or reduce model complexity</p>
              <p><strong>Slow training:</strong> Reduce batch size or model complexity</p>
            </div>
            
            <div class="tip-card">
              <h4>🎓 Advanced Tips</h4>
              <p><strong>Batch Normalization:</strong> Add after Conv2D layers for faster, more stable training</p>
              <p><strong>Data Augmentation:</strong> Improve generalization by transforming training images</p>
              <p><strong>Transfer Learning:</strong> Start with pre-trained models for better results</p>
              <p><strong>Ensemble Methods:</strong> Combine multiple models for higher accuracy</p>
            </div>
            
            <div class="tip-card">
              <h4>📚 Resources</h4>
              <p><strong>Export to Colab:</strong> Use Google's free GPUs for faster training</p>
              <p><strong>Model Templates:</strong> Start with proven architectures</p>
              <p><strong>Save Frequently:</strong> Browser storage persists your work</p>
              <p><strong>Experiment:</strong> Try different architectures - there's no single best solution!</p>
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-backdrop {
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
  
  .modal {
    background: #0f0f0f;
    border: 1px solid #262626;
    border-radius: 12px;
    width: 90%;
    max-width: 900px;
    max-height: 85vh;
    display: flex;
    flex-direction: column;
    animation: slideIn 0.3s ease-out;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  }
  
  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 24px;
    border-bottom: 1px solid #262626;
  }
  
  .header-content {
    flex: 1;
  }
  
  .modal-header h2 {
    margin: 0;
    font-size: 24px;
    font-weight: 600;
    color: #ffffff;
  }
  
  .subtitle {
    margin: 4px 0 0 0;
    font-size: 14px;
    color: #737373;
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
  
  .tabs {
    display: flex;
    gap: 8px;
    padding: 0 24px;
    background: #0a0a0a;
    border-bottom: 1px solid #262626;
  }
  
  .tab {
    background: none;
    border: none;
    color: #737373;
    padding: 16px 24px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
    border-bottom: 2px solid transparent;
  }
  
  .tab:hover {
    color: #a3a3a3;
  }
  
  .tab.active {
    color: #ffffff;
    border-bottom-color: #22c55e;
  }
  
  .content {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
  }
  
  .section h3 {
    font-size: 18px;
    margin: 0 0 16px 0;
    color: #ffffff;
  }
  
  .section h4 {
    font-size: 14px;
    margin: 0 0 8px 0;
    color: #e5e5e5;
  }
  
  .section p {
    margin: 0 0 16px 0;
    line-height: 1.6;
    color: #a3a3a3;
  }
  
  .concept {
    background: #171717;
    border: 1px solid #262626;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
  }
  
  .concept h4 {
    font-size: 16px;
    margin-bottom: 8px;
  }
  
  .concept p {
    margin: 0;
    font-size: 14px;
  }
  
  .layer-guide,
  .training-guide {
    background: #171717;
    border-left: 3px solid #22c55e;
    padding: 16px;
    margin-bottom: 16px;
  }
  
  .layer-guide h4,
  .training-guide h4 {
    color: #22c55e;
    margin-bottom: 8px;
  }
  
  .example,
  .tip {
    font-size: 13px;
    color: #737373;
    font-style: italic;
    margin-top: 8px;
  }
  
  .tip-card {
    background: #171717;
    border: 1px solid #262626;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }
  
  .tip-card h4 {
    font-size: 16px;
    margin-bottom: 8px;
  }
  
  .tip-card p {
    margin: 8px 0;
    font-size: 14px;
  }
  
  .tip-card p strong {
    color: #e5e5e5;
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
  
  /* Quick Start styles */
  .quickstart-step {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    align-items: flex-start;
  }
  
  .quickstart-step .step-number {
    width: 32px;
    height: 32px;
    background: #22c55e;
    color: #000;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    flex-shrink: 0;
  }
  
  .step-content {
    flex: 1;
  }
  
  .step-content h4 {
    margin-top: 0;
  }
  
  .step-content ul {
    margin: 8px 0;
    padding-left: 20px;
    color: #a3a3a3;
  }
  
  .step-content li {
    margin: 4px 0;
  }
  
  .info-box {
    background: #171717;
    border: 1px solid #22c55e;
    border-radius: 8px;
    padding: 16px;
    margin-top: 24px;
  }
  
  .info-box h4 {
    color: #22c55e;
    margin-top: 0;
  }
  
  .info-box ul {
    margin: 8px 0;
    padding-left: 20px;
    color: #a3a3a3;
  }
  
  /* Interface styles */
  .interface-section {
    background: #171717;
    border: 1px solid #262626;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }
  
  .interface-section ul {
    margin: 8px 0;
    padding-left: 20px;
    color: #a3a3a3;
  }
  
  .interface-section li {
    margin: 4px 0;
  }
  
  .toolbar-items {
    display: grid;
    gap: 8px;
    margin-top: 12px;
  }
  
  .toolbar-item {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
  }
  
  .button-preview {
    background: #262626;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 13px;
    color: #e5e5e5;
    font-family: monospace;
    white-space: nowrap;
  }
  
  .shortcuts {
    display: grid;
    gap: 8px;
    margin-top: 12px;
  }
  
  .shortcut {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
  }
  
  kbd {
    background: #262626;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    font-family: monospace;
    color: #e5e5e5;
  }
</style>
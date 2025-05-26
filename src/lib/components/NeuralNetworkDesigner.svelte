<script lang="ts">
  // UI Components for the neural network designer
  import LayerPalette from './nn-designer/LayerPalette.svelte';
  import NetworkCanvas from './nn-designer/NetworkCanvas.svelte';
  import LayerProperties from './nn-designer/LayerProperties.svelte';
  import ModelSummary from './nn-designer/ModelSummary.svelte';
  import TrainingConfig from './nn-designer/TrainingConfig.svelte';
  import DatasetSelector from './nn-designer/DatasetSelector.svelte';
  import TrainingProgress from './nn-designer/TrainingProgress.svelte';
  import ModelTemplates from './nn-designer/ModelTemplates.svelte';
  
  // State management and core functionality
  import { isTraining, layers, resetTraining } from '$lib/nn-designer/stores';
  import { trainingManager } from '$lib/nn-designer/trainingManager';
  import { modelBuilder } from '$lib/nn-designer/modelBuilder';
  import { showSuccess, showError } from '$lib/stores/toastStore';
  import ConfirmDialog from './ConfirmDialog.svelte';
  import InstructionModal from './InstructionModal.svelte';
  import ToastContainer from './ToastContainer.svelte';
  
  // Control visibility of training progress modal
  let showTrainingProgress = false;
  
  // Control visibility of dialogs
  let showClearConfirm = false;
  let showColabInstructions = false;
  
  // Start or stop training the neural network model
  async function handleRunStop() {
    if ($isTraining) {
      trainingManager.stopTraining();
    } else {
      try {
        showTrainingProgress = true; // Show progress modal
        await trainingManager.startTraining(); // Delegate to training manager
      } catch (error) {
        showError(`Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        showTrainingProgress = false; // Hide modal on error
      }
    }
  }
  
  // Save current model configuration to browser's localStorage
  function handleSave() {
    const modelData = {
      layers: $layers, // Current layer configuration from store
      timestamp: new Date().toISOString() // Add timestamp for versioning
    };
    localStorage.setItem('nn-designer-model', JSON.stringify(modelData));
    showSuccess('Model saved to browser storage!');
  }
  
  // Control visibility of export dropdown
  let showExportDropdown = false;

  // Close dropdown when clicking outside
  function handleClickOutside(event: MouseEvent) {
    const target = event.target as Element;
    if (!target.closest('.export-dropdown')) {
      showExportDropdown = false;
    }
  }

  // Export trained model to downloadable files
  async function handleExport() {
    try {
      await modelBuilder.exportModel(); // Export as TensorFlow.js format
      showSuccess('Model exported successfully!');
    } catch (error) {
      showError(`Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  // Export model configuration as JSON
  function handleExportJSON() {
    const modelData = {
      layers: $layers,
      trainingConfig: {
        epochs: 10,
        batchSize: 32,
        learningRate: 0.001,
        optimizer: 'adam',
        validationSplit: 0.2,
        loss: 'categoricalCrossentropy'
      },
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(modelData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = 'neural-network-config.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    URL.revokeObjectURL(url);
    showExportDropdown = false;
  }

  // Export to Google Colab with instructions
  function handleExportColab() {
    const colabCode = `# Neural Network Training in Google Colab
# Generated from NN Designer

# Install TensorFlow
!pip install tensorflow

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Model Configuration
model_config = ${JSON.stringify($layers, null, 2)}

# Build the model
model = keras.Sequential()

# Add layers based on configuration
for layer_config in model_config:
    layer_type = layer_config['type']
    params = layer_config['params']
    
    if layer_type == 'input':
        # Input layer is implicit in Sequential model
        pass
    elif layer_type == 'dense':
        model.add(layers.Dense(
            units=params['units'],
            activation=params.get('activation', 'relu')
        ))
    elif layer_type == 'conv2d':
        model.add(layers.Conv2D(
            filters=params['filters'],
            kernel_size=params['kernelSize'],
            activation=params.get('activation', 'relu')
        ))
    elif layer_type == 'flatten':
        model.add(layers.Flatten())
    elif layer_type == 'dropout':
        model.add(layers.Dropout(params['rate']))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Load your dataset here
# For MNIST example:
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
# x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)

# Train the model
# history = model.fit(
#     x_train, y_train,
#     batch_size=32,
#     epochs=10,
#     validation_data=(x_test, y_test),
#     verbose=1
# )

print("Model configuration exported successfully!")
print("Next steps:")
print("1. Upload your dataset or use a built-in dataset like MNIST")
print("2. Uncomment and modify the data loading section")
print("3. Uncomment the training section")
print("4. Run the notebook to train your model with GPU acceleration!")
`;

    const blob = new Blob([colabCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = 'neural-network-colab.py';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    URL.revokeObjectURL(url);
    
    // Show instructions modal
    showColabInstructions = true;
    showExportDropdown = false;
  }
  
  // Clear current model and reset to default state
  function handleClear() {
    showClearConfirm = true;
  }
  
  function confirmClear() {
    // Reset to single input layer with MNIST dimensions
    layers.set([
      {
        id: 'input-1',
        type: 'input',
        name: 'Input Layer',
        params: { shape: [28, 28] } // Default MNIST input shape
      }
    ]);
    resetTraining(); // Clear training history and metrics
    showClearConfirm = false;
    showSuccess('Model cleared successfully!');
  }
</script>

<!-- Main container for the neural network designer application -->
<div class="designer-container" on:click={handleClickOutside}>
  <!-- Top toolbar with action buttons -->
  <div class="top-bar">
    <div class="toolbar">
      <button class="btn {$isTraining ? 'btn-danger' : 'btn-primary'}" on:click={handleRunStop}>
        <span class="icon">{$isTraining ? '⏹' : '▶'}</span> {$isTraining ? 'Stop' : 'Run'}
      </button>
      <button class="btn btn-training" on:click={() => showTrainingProgress = true}>
        <span class="icon">📊</span> Show Training
      </button>
      <button class="btn btn-secondary" on:click={handleSave}>Save</button>
      <div class="export-dropdown" class:open={showExportDropdown}>
        <button class="btn btn-secondary export-trigger" on:click={() => showExportDropdown = !showExportDropdown}>
          Export ▼
        </button>
        {#if showExportDropdown}
          <div class="export-menu">
            <button class="export-option" on:click={handleExportJSON}>
              📄 Download JSON
            </button>
            <button class="export-option" on:click={handleExportColab}>
              📓 Download Colab
            </button>
          </div>
        {/if}
      </div>
      <button class="btn btn-danger" on:click={handleClear}>Clear</button>
    </div>
    <div class="model-name">Model: my_neural_network</div>
  </div>
  
  <!-- Main content area with three-panel layout -->
  <div class="main-content">
    <!-- Left sidebar: Model types, layer palette, and dataset selector -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1>NN Designer</h1>
      </div>
      
      <div class="sidebar-section">
        <h3>MODEL TYPES</h3>
        <div class="model-type-selector">
          <label class="model-type active">
            <input type="radio" name="modelType" value="sequential" checked />
            <span class="icon">●</span>
            <span>Sequential</span>
            <span class="badge">Active</span>
          </label>
          <label class="model-type disabled">
            <input type="radio" name="modelType" value="functional" disabled />
            <span class="icon">○</span>
            <span>Functional API</span>
          </label>
        </div>
      </div>
      
      <ModelTemplates />
      <LayerPalette />
      <DatasetSelector />
    </aside>
    
    <!-- Central canvas: Visual network editor -->
    <main class="canvas-area">
      <NetworkCanvas /> <!-- Drag-and-drop network visualization -->
    </main>
    
    <!-- Right panel: Layer properties editor, model summary, and training configuration -->
    <aside class="right-panel">
      <LayerProperties /> <!-- Edit selected layer parameters -->
      <ModelSummary /> <!-- Display parameter count and shapes -->
      <TrainingConfig /> <!-- Hyperparameter controls -->
    </aside>
  </div>
  
  <!-- Training progress modal (shown during training) -->
  {#if showTrainingProgress}
    <TrainingProgress on:close={() => showTrainingProgress = false} />
  {/if}
  
  <!-- Confirmation dialogs -->
  {#if showClearConfirm}
    <ConfirmDialog
      title="Clear Model"
      message="Are you sure you want to clear the model? This will remove all layers and reset to default."
      confirmText="Clear"
      cancelText="Cancel"
      type="danger"
      on:confirm={confirmClear}
      on:cancel={() => showClearConfirm = false}
    />
  {/if}
  
  <!-- Colab instructions modal -->
  {#if showColabInstructions}
    <InstructionModal
      title="Colab Export Complete!"
      instructions={[
        "Go to https://colab.research.google.com/",
        "Click 'Upload' and select the downloaded .py file",
        "Load your dataset (or use built-in datasets like MNIST)",
        "Train your model with GPU acceleration",
        "Evaluate performance and visualize results"
      ]}
      on:close={() => showColabInstructions = false}
    />
  {/if}
</div>

<!-- Toast notifications -->
<ToastContainer />

<style>
  .designer-container {
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: #0a0a0a;
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  
  .top-bar {
    height: 60px;
    background: #0a0a0a;
    border-bottom: 1px solid #262626;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
  }
  
  .toolbar {
    display: flex;
    gap: 16px;
  }
  
  .btn {
    padding: 8px 20px;
    border-radius: 6px;
    border: none;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .btn-primary {
    background: linear-gradient(to right, #22c55e, #16a34a);
    color: white;
  }
  
  .btn-primary:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(34, 197, 94, 0.3);
  }
  
  .btn-secondary {
    background: #1a1a1a;
    color: #d4d4d4;
    border: 1px solid #333333;
  }
  
  .btn-secondary:hover:not(:disabled) {
    background: #262626;
  }
  
  .btn-danger {
    background: #1a1a1a;
    color: #ef4444;
    border: 1px solid #333333;
  }
  
  .btn-training {
    background: linear-gradient(to right, #3b82f6, #1d4ed8);
    color: white;
  }
  
  .btn-training:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
  }
  
  .model-name {
    color: #737373;
    font-size: 13px;
  }
  
  .main-content {
    flex: 1;
    display: flex;
    overflow: hidden;
  }
  
  .sidebar {
    width: 280px;
    background: #0f0f0f;
    border-right: 1px solid #262626;
    overflow-y: auto;
  }
  
  .sidebar-header {
    height: 60px;
    background: #171717;
    display: flex;
    align-items: center;
    padding: 0 24px;
    border-bottom: 1px solid #262626;
  }
  
  .sidebar-header h1 {
    font-size: 20px;
    font-weight: 600;
    margin: 0;
  }
  
  .sidebar-section {
    padding: 20px 24px;
  }
  
  .sidebar-section h3 {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
    color: #737373;
    margin: 0 0 16px 0;
  }
  
  .model-type-selector {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .model-type {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 20px;
    border-radius: 8px;
    border: 1px solid #262626;
    background: #171717;
    cursor: pointer;
    position: relative;
  }
  
  .model-type.active {
    background: #052e16;
    border-color: #16a34a;
  }
  
  .model-type.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .model-type input {
    display: none;
  }
  
  .model-type .icon {
    font-size: 12px;
    color: #525252;
  }
  
  .model-type.active .icon {
    color: #22c55e;
  }
  
  .model-type .badge {
    margin-left: auto;
    font-size: 11px;
    padding: 2px 10px;
    border-radius: 10px;
    background: rgba(22, 163, 74, 0.2);
    color: #22c55e;
  }
  
  .canvas-area {
    flex: 1;
    background: #000000;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
  
  .right-panel {
    width: 300px;
    background: #0a0a0a;
    border-left: 1px solid #262626;
    padding: 30px 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .export-dropdown {
    position: relative;
    display: inline-block;
  }

  .export-trigger {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .export-menu {
    position: absolute;
    top: 100%;
    left: 0;
    background: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    z-index: 1000;
    min-width: 160px;
    margin-top: 4px;
  }

  .export-option {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 12px 16px;
    background: transparent;
    border: none;
    color: #d4d4d4;
    font-size: 14px;
    cursor: pointer;
    text-align: left;
    transition: background-color 0.2s;
  }

  .export-option:hover {
    background: #262626;
  }

  .export-option:first-child {
    border-radius: 6px 6px 0 0;
  }

  .export-option:last-child {
    border-radius: 0 0 6px 6px;
  }
</style>
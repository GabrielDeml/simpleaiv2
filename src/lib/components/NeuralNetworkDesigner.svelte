<script lang="ts">
  // UI Components for the neural network designer
  import LayerPalette from './nn-designer/LayerPalette.svelte';
  import NetworkCanvas from './nn-designer/NetworkCanvas.svelte';
  import LayerProperties from './nn-designer/LayerProperties.svelte';
  import ModelSummary from './nn-designer/ModelSummary.svelte';
  import TrainingConfig from './nn-designer/TrainingConfig.svelte';
  import DatasetSelector from './nn-designer/DatasetSelector.svelte';
  import TrainingProgress from './nn-designer/TrainingProgress.svelte';
  
  // State management and core functionality
  import { isTraining, layers, resetTraining } from '$lib/nn-designer/stores';
  import { trainingManager } from '$lib/nn-designer/trainingManager';
  import { modelBuilder } from '$lib/nn-designer/modelBuilder';
  
  // Control visibility of training progress modal
  let showTrainingProgress = false;
  
  // Start training the neural network model
  async function handleRun() {
    try {
      showTrainingProgress = true; // Show progress modal
      await trainingManager.startTraining(); // Delegate to training manager
    } catch (error) {
      alert(`Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      showTrainingProgress = false; // Hide modal on error
    }
  }
  
  // Save current model configuration to browser's localStorage
  function handleSave() {
    const modelData = {
      layers: $layers, // Current layer configuration from store
      timestamp: new Date().toISOString() // Add timestamp for versioning
    };
    localStorage.setItem('nn-designer-model', JSON.stringify(modelData));
    alert('Model saved to browser storage!');
  }
  
  // Export trained model to downloadable files
  async function handleExport() {
    try {
      await modelBuilder.exportModel(); // Export as TensorFlow.js format
      alert('Model exported successfully!');
    } catch (error) {
      alert(`Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
  
  // Clear current model and reset to default state
  function handleClear() {
    if (confirm('Are you sure you want to clear the model?')) {
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
    }
  }
</script>

<!-- Main container for the neural network designer application -->
<div class="designer-container">
  <!-- Top toolbar with action buttons -->
  <div class="top-bar">
    <div class="toolbar">
      <button class="btn btn-primary" on:click={handleRun} disabled={$isTraining}>
        <span class="icon">▶</span> Run
      </button>
      <button class="btn btn-secondary" on:click={handleSave}>Save</button>
      <button class="btn btn-secondary" on:click={handleExport}>Export</button>
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
      
      <LayerPalette />
      <DatasetSelector />
    </aside>
    
    <!-- Central canvas: Visual network editor and training configuration -->
    <main class="canvas-area">
      <NetworkCanvas /> <!-- Drag-and-drop network visualization -->
      <TrainingConfig /> <!-- Hyperparameter controls -->
    </main>
    
    <!-- Right panel: Layer properties editor and model summary -->
    <aside class="right-panel">
      <LayerProperties /> <!-- Edit selected layer parameters -->
      <ModelSummary /> <!-- Display parameter count and shapes -->
    </aside>
  </div>
  
  <!-- Training progress modal (shown during training) -->
  {#if showTrainingProgress}
    <TrainingProgress on:close={() => showTrainingProgress = false} />
  {/if}
</div>

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
</style>
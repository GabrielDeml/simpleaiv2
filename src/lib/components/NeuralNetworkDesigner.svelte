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
  import { isTraining, layers, resetTraining, trainingConfig, selectedDataset } from '$lib/nn-designer/stores';
  import { trainingManager } from '$lib/nn-designer/trainingManager';
  import { modelBuilder } from '$lib/nn-designer/modelBuilder';
  import { colabExporter } from '$lib/nn-designer/colabExporter';
  import { showSuccess, showError } from '$lib/stores/toastStore';
  import ConfirmDialog from './ConfirmDialog.svelte';
  import InstructionModal from './InstructionModal.svelte';
  import ToastContainer from './ToastContainer.svelte';
  import HelpModal from './HelpModal.svelte';
  import { onMount } from 'svelte';
  
  // Control visibility of training progress modal
  let showTrainingProgress = false;
  
  // Control visibility of dialogs
  let showClearConfirm = false;
  let showColabInstructions = false;
  let showHelp = false;
  
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
  
  // Set up click outside listener when dropdown is shown
  onMount(() => {
    const handleDocumentClick = (event: MouseEvent) => {
      if (showExportDropdown) {
        handleClickOutside(event);
      }
    };
    
    document.addEventListener('click', handleDocumentClick);
    
    return () => {
      document.removeEventListener('click', handleDocumentClick);
    };
  });

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
      trainingConfig: $trainingConfig,
      selectedDataset: $selectedDataset,
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
    try {
      const downloadUrl = colabExporter.generateDownloadLink(
        $layers,
        $trainingConfig,
        $selectedDataset
      );
      
      const filename = colabExporter.generateFilename($selectedDataset);
      
      // Create download link and trigger download
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up blob URL
      URL.revokeObjectURL(downloadUrl);
      showExportDropdown = false;
      showSuccess('Colab notebook exported successfully!');
    } catch (error) {
      console.error('Failed to export to Colab:', error);
      showError('Failed to export notebook. Please check your model configuration.');
    }
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
<div class="designer-container">
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
      <div class="toolbar-spacer"></div>
      <button class="btn btn-help" on:click={() => showHelp = true}>
        <span class="icon">❓</span> Help
      </button>
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
  
  <!-- Help modal -->
  <HelpModal isOpen={showHelp} on:close={() => showHelp = false} />
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
  
  .btn-help {
    background: #1a1a1a;
    color: #22c55e;
    border: 1px solid #333333;
  }
  
  .btn-help:hover {
    background: #262626;
    border-color: #22c55e;
  }
  
  .toolbar-spacer {
    flex: 1;
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
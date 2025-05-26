<script lang="ts">
  /**
   * TrainingConfig Component
   * 
   * Purpose: Provides a comprehensive form for configuring neural network
   * training hyperparameters. All settings are bound to a global store
   * for use during model training.
   * 
   * Key features:
   * - Grid layout for efficient space usage
   * - Input validation with min/max constraints
   * - Real-time binding to trainingConfig store
   * - Percentage display for validation split
   * - Grouped related settings (optimizer, loss, etc.)
   */
  
  import { trainingConfig, layers, selectedDataset } from '$lib/nn-designer/stores';
  import { colabExporter } from '$lib/nn-designer/colabExporter';

  /**
   * Exports the current model configuration to a Google Colab notebook.
   * Downloads a .ipynb file that can be opened directly in Colab.
   */
  function exportToColab() {
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
    } catch (error) {
      console.error('Failed to export to Colab:', error);
      alert('Failed to export notebook. Please check your model configuration.');
    }
  }
</script>

<!-- Training configuration card -->
<div class="training-config">
  <h2>Training Configuration</h2>
  
  <!-- Grid layout for training hyperparameters -->
  <div class="config-grid">
    <!-- Number of training epochs -->
    <div class="config-field">
      <label for="epochs">Epochs</label>
      <input
        id="epochs"
        type="number"
        bind:value={$trainingConfig.epochs}
        min="1"
        max="1000"
      />
    </div>
    
    <!-- Batch size for mini-batch gradient descent -->
    <div class="config-field">
      <label for="batch-size">Batch Size</label>
      <input
        id="batch-size"
        type="number"
        bind:value={$trainingConfig.batchSize}
        min="1"
        max="512"
      />
    </div>
    
    <!-- Learning rate for optimizer -->
    <div class="config-field">
      <label for="learning-rate">Learning Rate</label>
      <input
        id="learning-rate"
        type="number"
        bind:value={$trainingConfig.learningRate}
        min="0.00001"
        max="1"
        step="0.001"
      />
    </div>
    
    <!-- Optimization algorithm selection -->
    <div class="config-field">
      <label for="optimizer">Optimizer</label>
      <select id="optimizer" bind:value={$trainingConfig.optimizer}>
        <option value="adam">Adam</option>
        <option value="sgd">SGD</option>
        <option value="rmsprop">RMSprop</option>
      </select>
    </div>
    
    <!-- 
      Validation split percentage
      - Stored as decimal (0.2) but displayed as percentage (20%)
      - Custom input handler converts between formats
    -->
    <div class="config-field">
      <label for="validation-split">Validation Split</label>
      <div class="input-with-suffix">
        <input
          id="validation-split"
          type="number"
          value={$trainingConfig.validationSplit * 100}
          on:input={(e) => $trainingConfig.validationSplit = Number(e.currentTarget.value) / 100}
          min="0"
          max="50"
          step="5"
        />
        <span class="suffix">%</span>
      </div>
    </div>
    
    <!-- Loss function selection (spans 2 columns for wider dropdown) -->
    <div class="config-field wide">
      <label for="loss">Loss Function</label>
      <select id="loss" bind:value={$trainingConfig.loss}>
        <option value="categoricalCrossentropy">categorical_crossentropy</option>
        <option value="meanSquaredError">mean_squared_error</option>
        <option value="binaryCrossentropy">binary_crossentropy</option>
      </select>
    </div>
  </div>
  
  <!-- Export section -->
  <div class="export-section">
    <button class="export-btn" on:click={exportToColab}>
      <span class="export-icon">📓</span>
      Export to Google Colab
    </button>
    <p class="export-hint">
      Download a Jupyter notebook that you can open in Google Colab to train your model with GPU acceleration.
    </p>
  </div>
</div>

<style>
  .training-config {
    background: #0f0f0f;
    border-radius: 8px;
    padding: 20px;
    margin: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }
  
  h2 {
    font-size: 16px;
    font-weight: 500;
    margin: 0 0 20px 0;
  }
  
  .config-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
  }
  
  .config-field {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .config-field.wide {
    grid-column: span 2;
  }
  
  label {
    font-size: 12px;
    color: #737373;
  }
  
  input,
  select {
    height: 28px;
    padding: 0 12px;
    background: #000000;
    border: 1px solid #262626;
    border-radius: 4px;
    color: #ffffff;
    font-size: 13px;
  }
  
  select {
    cursor: pointer;
  }
  
  .input-with-suffix {
    position: relative;
    display: flex;
    align-items: center;
  }
  
  .input-with-suffix input {
    flex: 1;
    padding-right: 28px;
  }
  
  .suffix {
    position: absolute;
    right: 12px;
    color: #737373;
    font-size: 13px;
    pointer-events: none;
  }
  
  .export-section {
    margin-top: 24px;
    padding-top: 20px;
    border-top: 1px solid #262626;
  }
  
  .export-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 20px;
    background: #1a365d;
    border: 1px solid #2b77ad;
    border-radius: 6px;
    color: #ffffff;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    width: 100%;
    justify-content: center;
  }
  
  .export-btn:hover {
    background: #2c5282;
    border-color: #3182ce;
    transform: translateY(-1px);
  }
  
  .export-btn:active {
    transform: translateY(0);
  }
  
  .export-icon {
    font-size: 16px;
  }
  
  .export-hint {
    margin: 12px 0 0 0;
    font-size: 12px;
    color: #737373;
    line-height: 1.4;
  }
</style>
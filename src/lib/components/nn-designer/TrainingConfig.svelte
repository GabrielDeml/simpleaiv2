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
  
  import { trainingConfig } from '$lib/nn-designer/stores';

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
</style>
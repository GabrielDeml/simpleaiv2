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
  import { parameterHelp } from '$lib/nn-designer/parameterHelp';
  import Tooltip from '$lib/components/Tooltip.svelte';
</script>

<!-- Training configuration card -->
<div class="training-config">
  <h2>Training Configuration</h2>
  
  <!-- Grid layout for training hyperparameters -->
  <div class="config-grid">
    <!-- Number of training epochs -->
    <div class="config-field">
      <label for="epochs">
        <span>Epochs</span>
        <Tooltip content={parameterHelp.epochs.description} position="top" delay={200}>
          <svg class="help-icon" width="14" height="14" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="7.5" stroke="currentColor" stroke-opacity="0.5"/>
            <path d="M8 7V11" stroke="currentColor" stroke-linecap="round"/>
            <circle cx="8" cy="5" r="0.5" fill="currentColor"/>
          </svg>
        </Tooltip>
      </label>
      <input
        id="epochs"
        type="number"
        bind:value={$trainingConfig.epochs}
        min="1"
        max="1000"
      />
      <span class="help-text">{parameterHelp.epochs.example}</span>
    </div>
    
    <!-- Batch size for mini-batch gradient descent -->
    <div class="config-field">
      <label for="batch-size">
        <span>Batch Size</span>
        <Tooltip content={parameterHelp.batchSize.description} position="top" delay={200}>
          <svg class="help-icon" width="14" height="14" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="7.5" stroke="currentColor" stroke-opacity="0.5"/>
            <path d="M8 7V11" stroke="currentColor" stroke-linecap="round"/>
            <circle cx="8" cy="5" r="0.5" fill="currentColor"/>
          </svg>
        </Tooltip>
      </label>
      <input
        id="batch-size"
        type="number"
        bind:value={$trainingConfig.batchSize}
        min="1"
        max="512"
      />
      <span class="help-text">{parameterHelp.batchSize.example}</span>
    </div>
    
    <!-- Learning rate for optimizer -->
    <div class="config-field">
      <label for="learning-rate">
        <span>Learning Rate</span>
        <Tooltip content={parameterHelp.learningRate.description} position="top" delay={200}>
          <svg class="help-icon" width="14" height="14" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="7.5" stroke="currentColor" stroke-opacity="0.5"/>
            <path d="M8 7V11" stroke="currentColor" stroke-linecap="round"/>
            <circle cx="8" cy="5" r="0.5" fill="currentColor"/>
          </svg>
        </Tooltip>
      </label>
      <input
        id="learning-rate"
        type="number"
        bind:value={$trainingConfig.learningRate}
        min="0.00001"
        max="1"
        step="0.001"
      />
      <span class="help-text">{parameterHelp.learningRate.example}</span>
    </div>
    
    <!-- Optimization algorithm selection -->
    <div class="config-field">
      <label for="optimizer">
        <span>Optimizer</span>
        <Tooltip content={parameterHelp.optimizer.description} position="top" delay={200}>
          <svg class="help-icon" width="14" height="14" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="7.5" stroke="currentColor" stroke-opacity="0.5"/>
            <path d="M8 7V11" stroke="currentColor" stroke-linecap="round"/>
            <circle cx="8" cy="5" r="0.5" fill="currentColor"/>
          </svg>
        </Tooltip>
      </label>
      <select id="optimizer" bind:value={$trainingConfig.optimizer}>
        <option value="adam">Adam</option>
        <option value="sgd">SGD</option>
        <option value="rmsprop">RMSprop</option>
      </select>
      {#if $trainingConfig.optimizer && parameterHelp.optimizer.options[$trainingConfig.optimizer]}
        <span class="help-text">{parameterHelp.optimizer.options[$trainingConfig.optimizer]}</span>
      {/if}
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
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }
  
  h2 {
    font-size: 16px;
    font-weight: 500;
    margin: 0 0 20px 0;
  }
  
  .config-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
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
    display: flex;
    align-items: center;
    gap: 6px;
  }
  
  .help-icon {
    color: #525252;
    cursor: help;
    transition: color 0.2s;
  }
  
  .help-icon:hover {
    color: #737373;
  }
  
  .help-text {
    font-size: 11px;
    color: #525252;
    margin-top: -4px;
  }
  
  input,
  select {
    height: 32px;
    padding: 0 12px;
    background: #000000;
    border: 1px solid #262626;
    border-radius: 4px;
    color: #ffffff;
    font-size: 13px;
    transition: border-color 0.2s;
  }
  
  input:focus,
  select:focus {
    outline: none;
    border-color: #22c55e;
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
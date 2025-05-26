<script lang="ts">
  /**
   * LayerProperties Component
   * 
   * Purpose: Provides a dynamic form for editing the parameters of the
   * currently selected layer. Different layer types show different controls
   * based on their available parameters.
   * 
   * Key features:
   * - Dynamic form generation based on layer type
   * - Local editing state with Apply/Cancel functionality
   * - Type-specific input validation
   * - Custom controls (toggles, selects, number inputs)
   * - Real-time preview of changes before applying
   */
  
  import { layers, selectedLayerId, updateLayer } from '$lib/nn-designer/stores';
  import { layerDefinitions } from '$lib/nn-designer/layerDefinitions';
  import { parameterHelp } from '$lib/nn-designer/parameterHelp';
  import Tooltip from '$lib/components/Tooltip.svelte';
  
  // Reactive: Find the currently selected layer from the layers array
  $: selectedLayer = $layers.find(l => l.id === $selectedLayerId);
  
  // Reactive: Get the layer definition for the selected layer type
  $: layerDef = selectedLayer ? layerDefinitions[selectedLayer.type] : null;
  
  // Local state for edited parameters (allows cancel without saving)
  let editedParams = {};
  
  // Special handling for input layer shape (comma-separated string)
  let shapeString = '';
  
  /**
   * Reactive statement to update local edit state when selection changes
   * - Copies current parameters to local state
   * - Converts shape array to string for input layers
   * - Ensures UI always reflects current selection
   */
  $: if (selectedLayer) {
    editedParams = { ...selectedLayer.params };
    if (selectedLayer.type === 'input' && Array.isArray(selectedLayer.params.shape)) {
      shapeString = selectedLayer.params.shape.join(', ');
    }
  }
  
  /**
   * Applies the edited parameters to the selected layer
   * - Special handling for input shape (string to array conversion)
   * - Updates the global store with new parameters
   * - Validates and filters shape values to ensure they're numbers
   */
  function handleApply() {
    if (selectedLayer) {
      // Convert shape string back to array for input layers
      if (selectedLayer.type === 'input' && shapeString) {
        const shape = shapeString.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
        editedParams.shape = shape;
      }
      updateLayer(selectedLayer.id, editedParams);
    }
  }
  
  /**
   * Cancels editing and reverts to original parameters
   * - Resets local state to match current layer parameters
   * - Useful for undoing changes before they're applied
   */
  function handleCancel() {
    if (selectedLayer) {
      editedParams = { ...selectedLayer.params };
      if (selectedLayer.type === 'input' && Array.isArray(selectedLayer.params.shape)) {
        shapeString = selectedLayer.params.shape.join(', ');
      }
    }
  }
</script>

<!-- Main container for the properties panel -->
<div class="layer-properties">
  <h2>Layer Properties</h2>
  
  <!-- Show properties only when a layer is selected -->
  {#if selectedLayer && layerDef}
    <!-- Layer type indicator with dynamic color and position -->
    <p class="layer-type" style="color: {layerDef.color}">
      {layerDef.displayName} (Layer {$layers.findIndex(l => l.id === selectedLayer.id) + 1})
    </p>
    
    <!-- Dynamic form based on layer type -->
    <div class="properties-form">
      <!-- Input layer properties -->
      {#if selectedLayer.type === 'input'}
        <div class="form-group">
          <label>
            <span>Shape</span>
            <Tooltip content={parameterHelp.shape.description} position="right" delay={200}>
              <svg class="help-icon" width="14" height="14" viewBox="0 0 16 16" fill="none">
                <circle cx="8" cy="8" r="7.5" stroke="currentColor" stroke-opacity="0.5"/>
                <path d="M8 7V11" stroke="currentColor" stroke-linecap="round"/>
                <circle cx="8" cy="5" r="0.5" fill="currentColor"/>
              </svg>
            </Tooltip>
          </label>
          <!-- Shape as comma-separated values for easier editing -->
          <input
            type="text"
            bind:value={shapeString}
            placeholder="28, 28"
          />
          <span class="help-text">{parameterHelp.shape.example}</span>
        </div>
      {/if}
      
      <!-- Dense layer properties -->
      {#if selectedLayer.type === 'dense'}
        <!-- Number of neurons -->
        <div class="form-group">
          <label>
            <span>Units</span>
            <Tooltip content={parameterHelp.units.description} position="right" delay={200}>
              <svg class="help-icon" width="14" height="14" viewBox="0 0 16 16" fill="none">
                <circle cx="8" cy="8" r="7.5" stroke="currentColor" stroke-opacity="0.5"/>
                <path d="M8 7V11" stroke="currentColor" stroke-linecap="round"/>
                <circle cx="8" cy="5" r="0.5" fill="currentColor"/>
              </svg>
            </Tooltip>
          </label>
          <input
            type="number"
            bind:value={editedParams.units}
            min="1"
            max="10000"
          />
          <span class="help-text">{parameterHelp.units.example}</span>
        </div>
        
        <!-- Activation function selection -->
        <div class="form-group">
          <label>
            <span>Activation</span>
            <Tooltip content={parameterHelp.activation.description} position="right" delay={200}>
              <svg class="help-icon" width="14" height="14" viewBox="0 0 16 16" fill="none">
                <circle cx="8" cy="8" r="7.5" stroke="currentColor" stroke-opacity="0.5"/>
                <path d="M8 7V11" stroke="currentColor" stroke-linecap="round"/>
                <circle cx="8" cy="5" r="0.5" fill="currentColor"/>
              </svg>
            </Tooltip>
          </label>
          <select bind:value={editedParams.activation}>
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="tanh">Tanh</option>
            <option value="softmax">Softmax</option>
            <option value="linear">Linear</option>
          </select>
          {#if editedParams.activation && parameterHelp.activation.options[editedParams.activation]}
            <span class="help-text">{parameterHelp.activation.options[editedParams.activation]}</span>
          {/if}
        </div>
        
        <!-- Bias toggle with custom switch UI -->
        <div class="form-group">
          <label>Use Bias</label>
          <div class="toggle-switch">
            <input
              type="checkbox"
              id="use-bias"
              bind:checked={editedParams.useBias}
            />
            <label for="use-bias"></label>
          </div>
        </div>
        
        <!-- Weight initialization method -->
        <div class="form-group">
          <label>Kernel Initializer</label>
          <select bind:value={editedParams.kernelInitializer}>
            <option value="glorotUniform">glorot_uniform</option>
            <option value="glorotNormal">glorot_normal</option>
            <option value="heUniform">he_uniform</option>
            <option value="heNormal">he_normal</option>
            <option value="randomUniform">random_uniform</option>
            <option value="randomNormal">random_normal</option>
          </select>
        </div>
      {/if}
      
      <!-- Conv2D layer properties -->
      {#if selectedLayer.type === 'conv2d'}
        <!-- Number of convolutional filters -->
        <div class="form-group">
          <label>Filters</label>
          <input
            type="number"
            bind:value={editedParams.filters}
            min="1"
            max="512"
          />
        </div>
        
        <!-- Size of convolutional kernel -->
        <div class="form-group">
          <label>Kernel Size</label>
          <input
            type="number"
            bind:value={editedParams.kernelSize}
            min="1"
            max="11"
          />
        </div>
        
        <!-- Stride length for convolution -->
        <div class="form-group">
          <label>Strides</label>
          <input
            type="number"
            bind:value={editedParams.strides}
            min="1"
            max="5"
          />
        </div>
        
        <!-- Padding strategy -->
        <div class="form-group">
          <label>Padding</label>
          <select bind:value={editedParams.padding}>
            <option value="valid">Valid</option>
            <option value="same">Same</option>
          </select>
        </div>
        
        <!-- Activation function for conv layer -->
        <div class="form-group">
          <label>Activation</label>
          <select bind:value={editedParams.activation}>
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="tanh">Tanh</option>
            <option value="linear">Linear</option>
          </select>
        </div>
      {/if}
      
      <!-- MaxPooling2D layer properties -->
      {#if selectedLayer.type === 'maxpooling2d'}
        <!-- Size of pooling window -->
        <div class="form-group">
          <label>Pool Size</label>
          <input
            type="number"
            bind:value={editedParams.poolSize}
            min="2"
            max="5"
          />
        </div>
        
        <!-- Stride for pooling operation -->
        <div class="form-group">
          <label>Strides</label>
          <input
            type="number"
            bind:value={editedParams.strides}
            min="1"
            max="5"
          />
        </div>
        
        <!-- Padding for pooling -->
        <div class="form-group">
          <label>Padding</label>
          <select bind:value={editedParams.padding}>
            <option value="valid">Valid</option>
            <option value="same">Same</option>
          </select>
        </div>
      {/if}
      
      <!-- Dropout layer properties -->
      {#if selectedLayer.type === 'dropout'}
        <!-- Dropout rate (0-1) -->
        <div class="form-group">
          <label>
            <span>Rate</span>
            <Tooltip content={parameterHelp.rate.description} position="right" delay={200}>
              <svg class="help-icon" width="14" height="14" viewBox="0 0 16 16" fill="none">
                <circle cx="8" cy="8" r="7.5" stroke="currentColor" stroke-opacity="0.5"/>
                <path d="M8 7V11" stroke="currentColor" stroke-linecap="round"/>
                <circle cx="8" cy="5" r="0.5" fill="currentColor"/>
              </svg>
            </Tooltip>
          </label>
          <input
            type="number"
            bind:value={editedParams.rate}
            min="0"
            max="1"
            step="0.1"
          />
          <span class="help-text">{parameterHelp.rate.example}</span>
        </div>
      {/if}
      
      <!-- Action buttons -->
      <div class="form-actions">
        <button class="btn btn-primary" on:click={handleApply}>Apply</button>
        <button class="btn btn-secondary" on:click={handleCancel}>Cancel</button>
      </div>
    </div>
  {:else}
    <!-- Empty state when no layer is selected -->
    <p class="empty-state">Select a layer to edit its properties</p>
  {/if}
</div>

<style>
  .layer-properties {
    background: #0f0f0f;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }
  
  h2 {
    font-size: 16px;
    font-weight: 500;
    margin: 0 0 8px 0;
  }
  
  .layer-type {
    font-size: 14px;
    margin: 0 0 20px 0;
  }
  
  .empty-state {
    color: #737373;
    font-size: 14px;
    text-align: center;
    padding: 20px 0;
  }
  
  .properties-form {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  
  .form-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  
  .form-group label {
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
    margin-top: 4px;
  }
  
  input[type="text"],
  input[type="number"],
  select {
    height: 32px;
    padding: 0 12px;
    background: #000000;
    border: 1px solid #262626;
    border-radius: 6px;
    color: #ffffff;
    font-size: 14px;
  }
  
  select {
    cursor: pointer;
  }
  
  .toggle-switch {
    position: relative;
    width: 44px;
    height: 24px;
  }
  
  .toggle-switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }
  
  .toggle-switch label {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: #262626;
    border-radius: 12px;
    transition: 0.3s;
  }
  
  .toggle-switch label:before {
    position: absolute;
    content: "";
    height: 20px;
    width: 20px;
    left: 2px;
    bottom: 2px;
    background: white;
    border-radius: 50%;
    transition: 0.3s;
  }
  
  .toggle-switch input:checked + label {
    background: #22c55e;
  }
  
  .toggle-switch input:checked + label:before {
    transform: translateX(20px);
  }
  
  .form-actions {
    display: flex;
    gap: 10px;
    margin-top: 20px;
  }
  
  .btn {
    flex: 1;
    padding: 8px 16px;
    border-radius: 6px;
    border: none;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .btn-primary {
    background: linear-gradient(to right, #22c55e, #16a34a);
    color: white;
  }
  
  .btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(34, 197, 94, 0.3);
  }
  
  .btn-secondary {
    background: #1a1a1a;
    color: #d4d4d4;
    border: 1px solid #333333;
  }
  
  .btn-secondary:hover {
    background: #262626;
  }
</style>
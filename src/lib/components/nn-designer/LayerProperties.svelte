<script lang="ts">
  import { layers, selectedLayerId, updateLayer } from '$lib/nn-designer/stores';
  import { layerDefinitions } from '$lib/nn-designer/layerDefinitions';
  
  $: selectedLayer = $layers.find(l => l.id === $selectedLayerId);
  $: layerDef = selectedLayer ? layerDefinitions[selectedLayer.type] : null;
  
  let editedParams = {};
  
  $: if (selectedLayer) {
    editedParams = { ...selectedLayer.params };
  }
  
  function handleApply() {
    if (selectedLayer) {
      updateLayer(selectedLayer.id, editedParams);
    }
  }
  
  function handleCancel() {
    if (selectedLayer) {
      editedParams = { ...selectedLayer.params };
    }
  }
</script>

<div class="layer-properties">
  <h2>Layer Properties</h2>
  
  {#if selectedLayer && layerDef}
    <p class="layer-type" style="color: {layerDef.color}">
      {layerDef.displayName} (Layer {$layers.findIndex(l => l.id === selectedLayer.id) + 1})
    </p>
    
    <div class="properties-form">
      {#if selectedLayer.type === 'input'}
        <div class="form-group">
          <label>Shape</label>
          <input
            type="text"
            bind:value={editedParams.shape}
            placeholder="28, 28"
          />
        </div>
      {/if}
      
      {#if selectedLayer.type === 'dense'}
        <div class="form-group">
          <label>Units</label>
          <input
            type="number"
            bind:value={editedParams.units}
            min="1"
            max="10000"
          />
        </div>
        
        <div class="form-group">
          <label>Activation</label>
          <select bind:value={editedParams.activation}>
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="tanh">Tanh</option>
            <option value="softmax">Softmax</option>
            <option value="linear">Linear</option>
          </select>
        </div>
        
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
      
      {#if selectedLayer.type === 'conv2d'}
        <div class="form-group">
          <label>Filters</label>
          <input
            type="number"
            bind:value={editedParams.filters}
            min="1"
            max="512"
          />
        </div>
        
        <div class="form-group">
          <label>Kernel Size</label>
          <input
            type="number"
            bind:value={editedParams.kernelSize}
            min="1"
            max="11"
          />
        </div>
        
        <div class="form-group">
          <label>Strides</label>
          <input
            type="number"
            bind:value={editedParams.strides}
            min="1"
            max="5"
          />
        </div>
        
        <div class="form-group">
          <label>Padding</label>
          <select bind:value={editedParams.padding}>
            <option value="valid">Valid</option>
            <option value="same">Same</option>
          </select>
        </div>
        
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
      
      {#if selectedLayer.type === 'maxpooling2d'}
        <div class="form-group">
          <label>Pool Size</label>
          <input
            type="number"
            bind:value={editedParams.poolSize}
            min="2"
            max="5"
          />
        </div>
        
        <div class="form-group">
          <label>Strides</label>
          <input
            type="number"
            bind:value={editedParams.strides}
            min="1"
            max="5"
          />
        </div>
        
        <div class="form-group">
          <label>Padding</label>
          <select bind:value={editedParams.padding}>
            <option value="valid">Valid</option>
            <option value="same">Same</option>
          </select>
        </div>
      {/if}
      
      {#if selectedLayer.type === 'dropout'}
        <div class="form-group">
          <label>Rate</label>
          <input
            type="number"
            bind:value={editedParams.rate}
            min="0"
            max="1"
            step="0.1"
          />
        </div>
      {/if}
      
      <div class="form-actions">
        <button class="btn btn-primary" on:click={handleApply}>Apply</button>
        <button class="btn btn-secondary" on:click={handleCancel}>Cancel</button>
      </div>
    </div>
  {:else}
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
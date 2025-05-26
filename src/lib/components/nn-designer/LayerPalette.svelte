<script lang="ts">
  import { layerDefinitions } from '$lib/nn-designer/layerDefinitions';
  import { addLayer } from '$lib/nn-designer/stores';
  import type { LayerType } from '$lib/nn-designer/types';
  
  let draggedLayerType: LayerType | null = null;
  
  function handleDragStart(e: DragEvent, layerType: LayerType) {
    draggedLayerType = layerType;
    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = 'copy';
      e.dataTransfer.setData('layerType', layerType);
    }
  }
  
  function handleKeyDown(e: KeyboardEvent, layerType: LayerType) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick(layerType);
    }
  }
  
  function handleDragEnd() {
    draggedLayerType = null;
  }
  
  function generateLayerId(type: LayerType): string {
    return `${type}-${Date.now()}`;
  }
  
  function handleClick(layerType: LayerType) {
    const definition = layerDefinitions[layerType];
    const newLayer = {
      id: generateLayerId(layerType),
      type: layerType,
      name: definition.displayName,
      params: { ...definition.defaultParams }
    };
    
    addLayer(newLayer);
  }
</script>

<div class="layer-palette">
  <h3>LAYERS (Drag to add)</h3>
  
  <div class="layer-cards">
    {#each Object.entries(layerDefinitions) as [type, definition]}
      <div
        class="layer-card"
        style="--layer-color: {definition.color}"
        draggable="true"
        on:dragstart={(e) => handleDragStart(e, type as LayerType)}
        on:dragend={handleDragEnd}
        on:click={() => handleClick(type as LayerType)}
        on:keydown={(e) => handleKeyDown(e, type as LayerType)}
        role="button"
        tabindex="0"
      >
        <div class="layer-accent"></div>
        <div class="layer-icon">
          <span>{definition.icon}</span>
        </div>
        <div class="layer-name">{definition.displayName}</div>
        <div class="layer-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    {/each}
  </div>
</div>

<style>
  .layer-palette {
    padding: 20px 24px;
  }
  
  h3 {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
    color: #737373;
    margin: 0 0 16px 0;
  }
  
  .layer-cards {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .layer-card {
    position: relative;
    display: flex;
    align-items: center;
    gap: 16px;
    height: 48px;
    padding: 0 16px;
    background: #171717;
    border: 1px solid var(--layer-color);
    border-radius: 8px;
    cursor: move;
    transition: all 0.2s;
    user-select: none;
  }
  
  .layer-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }
  
  .layer-card:active {
    transform: translateY(0);
  }
  
  .layer-accent {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: var(--layer-color);
    border-radius: 8px 0 0 8px;
  }
  
  .layer-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: color-mix(in srgb, var(--layer-color) 20%, transparent);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: 12px;
  }
  
  .layer-icon span {
    font-size: 12px;
    font-weight: 600;
    color: var(--layer-color);
  }
  
  .layer-name {
    flex: 1;
    font-size: 14px;
    color: #ffffff;
  }
  
  .layer-dots {
    display: flex;
    gap: 4px;
  }
  
  .layer-dots span {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--layer-color);
    opacity: 0.5;
  }
</style>
<script lang="ts">
  /**
   * LayerPalette Component
   * 
   * Purpose: Provides a draggable palette of available neural network layers
   * that users can drag onto the NetworkCanvas or click to add to the model.
   * 
   * Key features:
   * - Displays all available layer types from layerDefinitions
   * - Supports drag-and-drop to NetworkCanvas
   * - Click-to-add functionality as an alternative to dragging
   * - Keyboard accessibility (Enter/Space to add layers)
   * - Visual feedback on hover and during drag operations
   */
  
  import { layerDefinitions } from '$lib/nn-designer/layerDefinitions';
  import { addLayer } from '$lib/nn-designer/stores';
  import type { LayerType } from '$lib/nn-designer/types';
  
  // Tracks which layer type is currently being dragged (for potential visual feedback)
  let draggedLayerType: LayerType | null = null;
  
  /**
   * Handles the start of a drag operation
   * @param e - The drag event containing dataTransfer object
   * @param layerType - The type of layer being dragged (input, dense, conv2d, etc.)
   * 
   * Sets up the drag data transfer with the layer type so NetworkCanvas
   * can receive it on drop. The 'copy' effect indicates we're creating
   * a new instance, not moving an existing layer.
   */
  function handleDragStart(e: DragEvent, layerType: LayerType) {
    draggedLayerType = layerType;
    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = 'copy';
      e.dataTransfer.setData('layerType', layerType);
    }
  }
  
  /**
   * Handles keyboard interaction for accessibility
   * @param e - Keyboard event
   * @param layerType - The layer type to add when activated
   * 
   * Allows users to add layers using keyboard navigation by pressing
   * Enter or Space when a layer card is focused.
   */
  function handleKeyDown(e: KeyboardEvent, layerType: LayerType) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick(layerType);
    }
  }
  
  /**
   * Cleans up after a drag operation ends
   * Resets the draggedLayerType to remove any visual drag indicators
   */
  function handleDragEnd() {
    draggedLayerType = null;
  }
  
  /**
   * Generates a unique ID for a new layer instance
   * @param type - The layer type
   * @returns A unique ID combining type and timestamp
   * 
   * Using timestamp ensures uniqueness for layers created in quick succession
   */
  function generateLayerId(type: LayerType): string {
    return `${type}-${Date.now()}`;
  }
  
  /**
   * Handles click events to add a layer directly (without dragging)
   * @param layerType - The type of layer to add
   * 
   * Creates a new layer instance with:
   * - Unique ID
   * - Layer type
   * - Display name from layer definition
   * - Default parameters copied from layer definition
   * 
   * The spread operator on defaultParams ensures each layer gets its own
   * parameter object instance, preventing shared state issues.
   */
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

<!-- Main container for the layer palette sidebar -->
<div class="layer-palette">
  <h3>LAYERS (Drag to add)</h3>
  
  <!-- Container for all layer cards -->
  <div class="layer-cards">
    <!-- Iterate through all available layer types from layerDefinitions -->
    {#each Object.entries(layerDefinitions) as [type, definition]}
      <!-- 
        Individual layer card that can be dragged or clicked
        - CSS custom property for dynamic coloring based on layer type
        - draggable attribute enables HTML5 drag functionality
        - Multiple event handlers for different interaction methods
        - ARIA attributes for accessibility (role="button" + tabindex)
      -->
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
        <!-- Colored accent bar on the left side of the card -->
        <div class="layer-accent"></div>
        
        <!-- Icon container with layer-specific icon/emoji -->
        <div class="layer-icon">
          <span>{definition.icon}</span>
        </div>
        
        <!-- Layer type display name -->
        <div class="layer-name">{definition.displayName}</div>
        
        <!-- Decorative dots on the right (visual hint for draggability) -->
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
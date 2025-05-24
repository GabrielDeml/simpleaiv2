<script lang="ts">
  import { availableLayers } from '../layers';
  import type { LayerType } from '../types';

  const categories = [
    { id: 'input', name: 'Input', color: 'bg-blue-500' },
    { id: 'core', name: 'Core', color: 'bg-green-500' },
    { id: 'convolutional', name: 'Convolutional', color: 'bg-purple-500' },
    { id: 'pooling', name: 'Pooling', color: 'bg-yellow-500' },
    { id: 'regularization', name: 'Regularization', color: 'bg-red-500' },
    { id: 'normalization', name: 'Normalization', color: 'bg-indigo-500' }
  ];

  function handleDragStart(event: DragEvent, layer: LayerType) {
    event.dataTransfer!.effectAllowed = 'copy';
    event.dataTransfer!.setData('layerType', layer.id);
  }

  function getCategoryColor(category: string): string {
    return categories.find(c => c.id === category)?.color || 'bg-gray-500';
  }
</script>

<div class="h-full bg-gray-100 dark:bg-gray-800 overflow-y-auto w-72">
  <h2 class="text-lg font-bold mb-4 px-4 pt-4">Neural Network Layers</h2>
  
  <div class="layer-categories">
    {#each categories as category}
      {@const categoryLayers = availableLayers.filter(l => l.category === category.id)}
      {#if categoryLayers.length > 0}
        <div class="mb-4">
          <h3 class="text-sm font-semibold px-4 py-2 text-gray-600 dark:text-gray-400">{category.name}</h3>
          <div class="px-2">
            {#each categoryLayers as layer}
              <div
                class="p-3 m-2 rounded-lg cursor-move text-white shadow-sm hover:shadow-md transition-shadow bg-opacity-90 hover:bg-opacity-100 {getCategoryColor(layer.category)}"
                draggable="true"
                on:dragstart={(e) => handleDragStart(e, layer)}
                role="button"
                tabindex="0"
              >
                <div class="font-semibold text-sm">{layer.name}</div>
                <div class="text-xs opacity-90 mt-1">{layer.description}</div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {/each}
  </div>

  <div class="p-4 mt-4 border-t border-gray-300 dark:border-gray-700 text-gray-600 dark:text-gray-400">
    <h3 class="text-sm font-semibold mb-2">Tips:</h3>
    <ul class="text-xs space-y-1">
      <li>• Drag layers to the canvas</li>
      <li>• Connect layers by clicking outputs to inputs</li>
      <li>• Double-click to configure</li>
      <li>• Right-click to delete</li>
    </ul>
  </div>
</div>
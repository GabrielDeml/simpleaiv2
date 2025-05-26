import { writable, derived } from 'svelte/store';
import type { LayerConfig, TrainingConfig, DatasetType, ModelSummary } from './types';

// Network architecture store
export const layers = writable<LayerConfig[]>([
  {
    id: 'input-1',
    type: 'input',
    name: 'Input Layer',
    params: { shape: [28, 28] }
  }
]);

// Selected layer for editing
export const selectedLayerId = writable<string | null>(null);

// Training configuration
export const trainingConfig = writable<TrainingConfig>({
  epochs: 10,
  batchSize: 32,
  learningRate: 0.001,
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  validationSplit: 0.2
});

// Selected dataset
export const selectedDataset = writable<DatasetType>('mnist');

// Training state
export const isTraining = writable(false);
export const currentEpoch = writable(0);
export const trainingHistory = writable<{
  loss: number[];
  valLoss: number[];
  accuracy: number[];
  valAccuracy: number[];
}>({
  loss: [],
  valLoss: [],
  accuracy: [],
  valAccuracy: []
});

// Model summary derived from layers
export const modelSummary = derived(layers, ($layers) => {
  // This will be calculated when we build the actual model
  // For now, return placeholder values
  const summary: ModelSummary = {
    totalParams: 0,
    trainableParams: 0,
    layerCount: $layers.length,
    outputShape: []
  };
  
  // Simple parameter calculation for demo
  if ($layers.length > 0) {
    const lastLayer = $layers[$layers.length - 1];
    if (lastLayer.type === 'dense') {
      summary.outputShape = [lastLayer.params.units];
    }
  }
  
  return summary;
});

// Helper functions
export function addLayer(layer: LayerConfig, afterId?: string) {
  layers.update(currentLayers => {
    if (!afterId) {
      return [...currentLayers, layer];
    }
    
    const index = currentLayers.findIndex(l => l.id === afterId);
    if (index === -1) {
      return [...currentLayers, layer];
    }
    
    const newLayers = [...currentLayers];
    newLayers.splice(index + 1, 0, layer);
    return newLayers;
  });
}

export function removeLayer(layerId: string) {
  layers.update(currentLayers => {
    return currentLayers.filter(l => l.id !== layerId);
  });
  
  // Clear selection if removed layer was selected
  selectedLayerId.update(selected => selected === layerId ? null : selected);
}

export function updateLayer(layerId: string, params: Record<string, any>) {
  layers.update(currentLayers => {
    return currentLayers.map(layer => {
      if (layer.id === layerId) {
        return { ...layer, params: { ...layer.params, ...params } };
      }
      return layer;
    });
  });
}

export function moveLayer(layerId: string, direction: 'up' | 'down') {
  layers.update(currentLayers => {
    const index = currentLayers.findIndex(l => l.id === layerId);
    if (index === -1) return currentLayers;
    
    const newIndex = direction === 'up' ? index - 1 : index + 1;
    
    // Check bounds
    if (newIndex < 0 || newIndex >= currentLayers.length) {
      return currentLayers;
    }
    
    // Don't allow moving before input layer
    if (newIndex === 0 && currentLayers[0].type === 'input') {
      return currentLayers;
    }
    
    const newLayers = [...currentLayers];
    const [movedLayer] = newLayers.splice(index, 1);
    newLayers.splice(newIndex, 0, movedLayer);
    
    return newLayers;
  });
}

export function resetTraining() {
  currentEpoch.set(0);
  trainingHistory.set({
    loss: [],
    valLoss: [],
    accuracy: [],
    valAccuracy: []
  });
}
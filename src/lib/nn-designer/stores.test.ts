import { describe, it, expect, beforeEach } from 'vitest';
import { get } from 'svelte/store';
import {
  layers,
  selectedLayerId,
  trainingConfig,
  selectedDataset,
  isTraining,
  currentEpoch,
  trainingHistory,
  modelSummary,
  addLayer,
  removeLayer,
  updateLayer,
  moveLayer,
  resetTraining,
  loadTemplate,
  getAvailableTemplates
} from './stores';
import type { LayerConfig, ModelTemplate } from './types';

describe('Neural Network Designer Stores', () => {
  beforeEach(() => {
    // Reset stores to default state
    layers.set([
      {
        id: 'input-1',
        type: 'input',
        name: 'Input Layer',
        params: { shape: [28, 28] }
      },
      {
        id: 'flatten-1',
        type: 'flatten',
        name: 'Flatten',
        params: {}
      },
      {
        id: 'dense-1',
        type: 'dense',
        name: 'Hidden Layer',
        params: { units: 128, activation: 'relu', useBias: true, kernelInitializer: 'glorotUniform' }
      },
      {
        id: 'dense-2',
        type: 'dense',
        name: 'Output Layer',
        params: { units: 10, activation: 'softmax', useBias: true, kernelInitializer: 'glorotUniform' }
      }
    ]);
    selectedLayerId.set(null);
    trainingConfig.set({
      epochs: 10,
      batchSize: 32,
      learningRate: 0.001,
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      validationSplit: 0.2
    });
    selectedDataset.set('mnist');
    isTraining.set(false);
    currentEpoch.set(0);
    trainingHistory.set({
      loss: [],
      valLoss: [],
      accuracy: [],
      valAccuracy: []
    });
  });

  describe('layers store', () => {
    it('initializes with default layers', () => {
      const currentLayers = get(layers);
      expect(currentLayers).toHaveLength(4);
      expect(currentLayers[0].type).toBe('input');
      expect(currentLayers[1].type).toBe('flatten');
      expect(currentLayers[2].type).toBe('dense');
      expect(currentLayers[3].type).toBe('dense');
    });
  });

  describe('addLayer', () => {
    it('adds layer to the end when no afterId provided', () => {
      const newLayer: LayerConfig = {
        id: 'dense-3',
        type: 'dense',
        name: 'New Dense',
        params: { units: 64, activation: 'relu' }
      };

      addLayer(newLayer);
      const currentLayers = get(layers);
      
      expect(currentLayers).toHaveLength(5);
      expect(currentLayers[4]).toEqual(newLayer);
    });

    it('adds layer after specified layer', () => {
      const newLayer: LayerConfig = {
        id: 'dropout-1',
        type: 'dropout',
        name: 'Dropout',
        params: { rate: 0.5 }
      };

      addLayer(newLayer, 'dense-1');
      const currentLayers = get(layers);
      
      expect(currentLayers).toHaveLength(5);
      expect(currentLayers[3]).toEqual(newLayer);
      expect(currentLayers[2].id).toBe('dense-1');
      expect(currentLayers[4].id).toBe('dense-2');
    });

    it('adds to end if afterId not found', () => {
      const newLayer: LayerConfig = {
        id: 'dense-3',
        type: 'dense',
        name: 'New Dense',
        params: { units: 64, activation: 'relu' }
      };

      addLayer(newLayer, 'non-existent');
      const currentLayers = get(layers);
      
      expect(currentLayers).toHaveLength(5);
      expect(currentLayers[4]).toEqual(newLayer);
    });
  });

  describe('removeLayer', () => {
    it('removes layer by id', () => {
      removeLayer('dense-1');
      const currentLayers = get(layers);
      
      expect(currentLayers).toHaveLength(3);
      expect(currentLayers.find(l => l.id === 'dense-1')).toBeUndefined();
    });

    it('clears selection if removed layer was selected', () => {
      selectedLayerId.set('dense-1');
      removeLayer('dense-1');
      
      expect(get(selectedLayerId)).toBeNull();
    });

    it('keeps selection if different layer was removed', () => {
      selectedLayerId.set('dense-2');
      removeLayer('dense-1');
      
      expect(get(selectedLayerId)).toBe('dense-2');
    });
  });

  describe('updateLayer', () => {
    it('updates layer parameters', () => {
      updateLayer('dense-1', { units: 256, activation: 'tanh' });
      
      const currentLayers = get(layers);
      const updatedLayer = currentLayers.find(l => l.id === 'dense-1');
      
      expect(updatedLayer?.params.units).toBe(256);
      expect(updatedLayer?.params.activation).toBe('tanh');
      expect(updatedLayer?.params.useBias).toBe(true); // Unchanged
    });

    it('does nothing if layer not found', () => {
      const beforeLayers = get(layers);
      updateLayer('non-existent', { units: 256 });
      const afterLayers = get(layers);
      
      expect(afterLayers).toEqual(beforeLayers);
    });
  });

  describe('moveLayer', () => {
    it('moves layer up', () => {
      moveLayer('dense-2', 'up');
      const currentLayers = get(layers);
      
      expect(currentLayers[2].id).toBe('dense-2');
      expect(currentLayers[3].id).toBe('dense-1');
    });

    it('moves layer down', () => {
      moveLayer('dense-1', 'down');
      const currentLayers = get(layers);
      
      expect(currentLayers[2].id).toBe('dense-2');
      expect(currentLayers[3].id).toBe('dense-1');
    });

    it('prevents moving before input layer', () => {
      moveLayer('flatten-1', 'up');
      const currentLayers = get(layers);
      
      expect(currentLayers[0].id).toBe('input-1');
      expect(currentLayers[1].id).toBe('flatten-1');
    });

    it('prevents moving beyond bounds', () => {
      moveLayer('dense-2', 'down');
      const currentLayers = get(layers);
      
      expect(currentLayers[3].id).toBe('dense-2');
    });

    it('handles non-existent layer', () => {
      const beforeLayers = get(layers);
      moveLayer('non-existent', 'up');
      const afterLayers = get(layers);
      
      expect(afterLayers).toEqual(beforeLayers);
    });
  });

  describe('resetTraining', () => {
    it('resets training state', () => {
      currentEpoch.set(5);
      trainingHistory.set({
        loss: [1, 0.9, 0.8],
        valLoss: [1.1, 1, 0.9],
        accuracy: [0.7, 0.8, 0.85],
        valAccuracy: [0.65, 0.75, 0.8]
      });

      resetTraining();

      expect(get(currentEpoch)).toBe(0);
      expect(get(trainingHistory)).toEqual({
        loss: [],
        valLoss: [],
        accuracy: [],
        valAccuracy: []
      });
    });
  });

  describe('loadTemplate', () => {
    it('loads template layers and dataset', () => {
      const template: ModelTemplate = {
        id: 'test-template',
        name: 'Test Template',
        description: 'Test',
        category: 'computer-vision',
        recommendedDataset: 'cifar10',
        layers: [
          {
            id: 'input-test',
            type: 'input',
            name: 'Input',
            params: { shape: [32, 32, 3] }
          },
          {
            id: 'conv-test',
            type: 'conv2d',
            name: 'Conv',
            params: { filters: 32, kernelSize: 3 }
          }
        ]
      };

      loadTemplate(template);

      expect(get(layers)).toEqual(template.layers);
      expect(get(selectedDataset)).toBe('cifar10');
      expect(get(selectedLayerId)).toBeNull();
    });

    it('resets training when loading template', () => {
      currentEpoch.set(5);
      trainingHistory.set({
        loss: [1, 0.9],
        valLoss: [1.1, 1],
        accuracy: [0.7, 0.8],
        valAccuracy: [0.65, 0.75]
      });

      const template: ModelTemplate = {
        id: 'test',
        name: 'Test',
        description: 'Test',
        category: 'classification',
        recommendedDataset: 'mnist',
        layers: []
      };

      loadTemplate(template);

      expect(get(currentEpoch)).toBe(0);
      expect(get(trainingHistory).loss).toHaveLength(0);
    });
  });

  describe('modelSummary derived store', () => {
    it('updates when layers change', () => {
      const initialSummary = get(modelSummary);
      expect(initialSummary.layerCount).toBe(4);
      expect(initialSummary.outputShape).toEqual([10]);

      removeLayer('dense-2');
      
      const updatedSummary = get(modelSummary);
      expect(updatedSummary.layerCount).toBe(3);
      expect(updatedSummary.outputShape).toEqual([128]);
    });

    it('handles empty layers', () => {
      layers.set([]);
      const summary = get(modelSummary);
      
      expect(summary.layerCount).toBe(0);
      expect(summary.outputShape).toEqual([]);
    });
  });

  describe('getAvailableTemplates', () => {
    it('returns model templates', () => {
      const templates = getAvailableTemplates();
      expect(Array.isArray(templates)).toBe(true);
      expect(templates.length).toBeGreaterThan(0);
    });
  });
});
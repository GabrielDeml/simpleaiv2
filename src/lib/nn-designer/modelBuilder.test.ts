import { describe, it, expect, beforeEach } from 'vitest';
import { ModelBuilder } from './modelBuilder';
import type { LayerConfig, TrainingConfig } from './types';

describe('ModelBuilder', () => {
  let modelBuilder: ModelBuilder;

  beforeEach(() => {
    modelBuilder = new ModelBuilder();
  });

  describe('buildModel', () => {
    it('creates a sequential model', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [28, 28] } },
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'dense-1', type: 'dense', name: 'Dense', params: { units: 10, activation: 'softmax' } }
      ];

      const model = modelBuilder.buildModel(layers);
      
      expect(model).toBeDefined();
    });

    it('handles different layer types', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [28, 28, 1] } },
        { id: 'conv-1', type: 'conv2d', name: 'Conv2D', params: { filters: 32, kernelSize: 3 } },
        { id: 'pool-1', type: 'maxpooling2d', name: 'MaxPool', params: { poolSize: 2 } },
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'dropout-1', type: 'dropout', name: 'Dropout', params: { rate: 0.5 } },
        { id: 'dense-1', type: 'dense', name: 'Dense', params: { units: 10 } }
      ];

      const model = modelBuilder.buildModel(layers);
      
      expect(model).toBeDefined();
    });

    it('throws error for empty layer configuration', () => {
      expect(() => modelBuilder.buildModel([])).toThrowError('No layers defined');
    });

    it('throws error for missing input layer', () => {
      const layers: LayerConfig[] = [
        { id: 'dense-1', type: 'dense', name: 'Dense', params: { units: 10 } }
      ];

      expect(() => modelBuilder.buildModel(layers)).toThrowError('First layer must be an input layer');
    });

    it('throws error for unknown layer type', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [28, 28] } },
        { id: 'unknown-1', type: 'unknown' as any, name: 'Unknown', params: {} }
      ];

      expect(() => modelBuilder.buildModel(layers)).toThrowError('Unknown layer type: unknown');
    });
  });

  describe('compileModel', () => {
    beforeEach(() => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [28, 28] } },
        { id: 'dense-1', type: 'dense', name: 'Dense', params: { units: 10 } }
      ];
      modelBuilder.buildModel(layers);
    });

    it('compiles model with training config', () => {
      const trainingConfig: TrainingConfig = {
        epochs: 10,
        batchSize: 32,
        learningRate: 0.001,
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        validationSplit: 0.2
      };

      expect(() => modelBuilder.compileModel(trainingConfig)).not.toThrow();
    });
  });

  describe('compileModel without model', () => {
    it('throws error if model not built', () => {
      const trainingConfig: TrainingConfig = {
        epochs: 10,
        batchSize: 32,
        learningRate: 0.001,
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        validationSplit: 0.2
      };

      expect(() => modelBuilder.compileModel(trainingConfig)).toThrowError('Model must be built before compiling');
    });
  });

  describe('getModel', () => {
    it('returns null if model not built', () => {
      expect(modelBuilder.getModel()).toBeNull();
    });

    it('returns model after building', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [28, 28] } },
        { id: 'dense-1', type: 'dense', name: 'Dense', params: { units: 10 } }
      ];

      modelBuilder.buildModel(layers);
      expect(modelBuilder.getModel()).not.toBeNull();
    });
  });

  describe('stopTrainingProcess', () => {
    it('does not throw when no model exists', () => {
      expect(() => modelBuilder.stopTrainingProcess()).not.toThrow();
    });

    it('sets stop training flag on model', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [28, 28] } },
        { id: 'dense-1', type: 'dense', name: 'Dense', params: { units: 10 } }
      ];

      modelBuilder.buildModel(layers);
      modelBuilder.stopTrainingProcess();
      
      const model = modelBuilder.getModel();
      expect(model).toBeDefined();
    });
  });

  describe('dispose', () => {
    it('does not throw if no model exists', () => {
      expect(() => modelBuilder.dispose()).not.toThrow();
    });

    it('disposes model if exists', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [28, 28] } },
        { id: 'dense-1', type: 'dense', name: 'Dense', params: { units: 10 } }
      ];

      modelBuilder.buildModel(layers);
      modelBuilder.dispose();
      
      expect(modelBuilder.getModel()).toBeNull();
    });
  });
});
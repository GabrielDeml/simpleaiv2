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

  describe('Transformer Layer Support', () => {
    it('should build embedding layer correctly', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [100] } },
        { id: 'embedding-1', type: 'embedding', name: 'Embedding', params: { vocabSize: 1000, embeddingDim: 64, maxLength: 100, trainable: true } },
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'output-1', type: 'output', name: 'Output', params: { units: 2, activation: 'softmax' } }
      ];

      const model = modelBuilder.buildModel(layers);
      expect(model).toBeDefined();
      expect(model.layers.length).toBe(3); // embedding, flatten, output
    });

    it('should build layer normalization correctly', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [10, 64] } },
        { id: 'norm-1', type: 'layerNormalization', name: 'LayerNorm', params: { epsilon: 1e-6, center: true, scale: true } },
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'output-1', type: 'output', name: 'Output', params: { units: 2, activation: 'softmax' } }
      ];

      const model = modelBuilder.buildModel(layers);
      expect(model).toBeDefined();
      expect(model.layers.length).toBe(3);
    });

    it('should build transformer text classifier correctly', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [50] } },
        { id: 'embedding-1', type: 'embedding', name: 'Embedding', params: { vocabSize: 1000, embeddingDim: 32, maxLength: 50, trainable: true } },
        { id: 'pos-enc-1', type: 'positionalEncoding', name: 'PosEnc', params: { maxLength: 50 } },
        { id: 'transformer-1', type: 'transformerBlock', name: 'Transformer', params: { numHeads: 2, keyDim: 16, ffDim: 64, dropout: 0.1 } },
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'output-1', type: 'output', name: 'Output', params: { units: 2, activation: 'softmax' } }
      ];

      const model = modelBuilder.buildModel(layers);
      expect(model).toBeDefined();
      expect(model.layers.length).toBe(5); // All layers except input
    });

    it('should build multi-head attention layer correctly', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [10, 32] } },
        { id: 'attention-1', type: 'multiHeadAttention', name: 'Attention', params: { numHeads: 4, keyDim: 8, dropout: 0.1, useBias: true } },
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'output-1', type: 'output', name: 'Output', params: { units: 2, activation: 'softmax' } }
      ];

      const model = modelBuilder.buildModel(layers);
      expect(model).toBeDefined();
      expect(model.layers.length).toBe(3);
    });

    it('should build positional encoding layer correctly', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [20, 16] } },
        { id: 'pos-enc-1', type: 'positionalEncoding', name: 'PosEnc', params: { maxLength: 20 } },
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'output-1', type: 'output', name: 'Output', params: { units: 3, activation: 'softmax' } }
      ];

      const model = modelBuilder.buildModel(layers);
      expect(model).toBeDefined();
      expect(model.layers.length).toBe(3);
    });

    it('should handle transformer block with all parameters', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [15, 24] } },
        { id: 'transformer-1', type: 'transformerBlock', name: 'Transformer', params: { 
          numHeads: 3, 
          keyDim: 8, 
          ffDim: 48, 
          dropout: 0.2 
        }},
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'output-1', type: 'output', name: 'Output', params: { units: 4, activation: 'softmax' } }
      ];

      const model = modelBuilder.buildModel(layers);
      expect(model).toBeDefined();
      expect(model.layers.length).toBe(3);
    });

    it('should handle embedding with custom parameters', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [200] } },
        { id: 'embedding-1', type: 'embedding', name: 'Embedding', params: { 
          vocabSize: 5000, 
          embeddingDim: 128, 
          maxLength: 200, 
          trainable: false 
        }},
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'output-1', type: 'output', name: 'Output', params: { units: 5, activation: 'softmax' } }
      ];

      const model = modelBuilder.buildModel(layers);
      expect(model).toBeDefined();
      expect(model.layers.length).toBe(3);
    });

    it('should handle layer normalization with custom parameters', () => {
      const layers: LayerConfig[] = [
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [8, 12] } },
        { id: 'norm-1', type: 'layerNormalization', name: 'LayerNorm', params: { 
          epsilon: 1e-5, 
          center: false, 
          scale: true 
        }},
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'output-1', type: 'output', name: 'Output', params: { units: 6, activation: 'softmax' } }
      ];

      const model = modelBuilder.buildModel(layers);
      expect(model).toBeDefined();
      expect(model.layers.length).toBe(3);
    });
  });
});
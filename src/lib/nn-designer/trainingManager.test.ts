import { describe, it, expect, beforeEach, vi } from 'vitest';
import { TrainingManager } from './trainingManager';
import { selectedDataset, layers } from './stores';

// Mock dependencies
vi.mock('@tensorflow/tfjs', () => ({
  tensor2d: vi.fn(() => ({ dispose: vi.fn() })),
  tensor4d: vi.fn(() => ({ dispose: vi.fn() })),
  dispose: vi.fn(),
  tidy: vi.fn((fn) => fn()),
  keep: vi.fn((tensor) => tensor)
}));

// Mock the new dataset interface
vi.mock('../datasets', () => ({
  getDataset: vi.fn(() => {
    const mockTensor = { dispose: vi.fn() };
    return {
      loadTensors: vi.fn(() => Promise.resolve({
        trainData: mockTensor,
        trainLabels: mockTensor,
        testData: mockTensor,
        testLabels: mockTensor
      })),
      getMetadata: vi.fn(() => ({
        name: 'Mock Dataset',
        description: 'Mock dataset for testing',
        inputShape: [28, 28],
        channels: 1,
        numClasses: 10,
        trainSize: 60000,
        testSize: 10000,
        classNames: Array(10).fill('Class')
      }))
    };
  })
}));

// Mock the legacy loaders that the new datasets use internally
vi.mock('../mnist/dataLoader', () => ({
  MnistData: vi.fn().mockImplementation(() => ({
    load: vi.fn(),
    getTrainData: vi.fn(() => ({
      xs: { data: () => Promise.resolve(new Float32Array(60000 * 28 * 28)), dispose: vi.fn() },
      ys: { data: () => Promise.resolve(new Float32Array(60000 * 10)), dispose: vi.fn() }
    })),
    getTestData: vi.fn(() => ({
      xs: { data: () => Promise.resolve(new Float32Array(10000 * 28 * 28)), dispose: vi.fn() },
      ys: { data: () => Promise.resolve(new Float32Array(10000 * 10)), dispose: vi.fn() }
    }))
  }))
}));

vi.mock('../datasets/cifar10', () => ({
  Cifar10Data: vi.fn().mockImplementation(() => ({
    load: vi.fn(),
    getTrainData: vi.fn(() => ({
      xs: { data: () => Promise.resolve(new Float32Array(50000 * 32 * 32 * 3)), dispose: vi.fn() },
      ys: { data: () => Promise.resolve(new Float32Array(50000 * 10)), dispose: vi.fn() }
    })),
    getTestData: vi.fn(() => ({
      xs: { data: () => Promise.resolve(new Float32Array(10000 * 32 * 32 * 3)), dispose: vi.fn() },
      ys: { data: () => Promise.resolve(new Float32Array(10000 * 10)), dispose: vi.fn() }
    }))
  }))
}));

vi.mock('../datasets/fashionMnist', () => ({
  FashionMnistData: vi.fn().mockImplementation(() => ({
    load: vi.fn(),
    getTrainData: vi.fn(() => ({
      xs: { data: () => Promise.resolve(new Float32Array(60000 * 28 * 28)), dispose: vi.fn() },
      ys: { data: () => Promise.resolve(new Float32Array(60000 * 10)), dispose: vi.fn() }
    })),
    getTestData: vi.fn(() => ({
      xs: { data: () => Promise.resolve(new Float32Array(10000 * 28 * 28)), dispose: vi.fn() },
      ys: { data: () => Promise.resolve(new Float32Array(10000 * 10)), dispose: vi.fn() }
    }))
  }))
}));

vi.mock('./modelBuilder', () => ({
  modelBuilder: {
    buildModel: vi.fn(() => ({
      summary: vi.fn(),
      layers: [],
      compile: vi.fn(),
      fit: vi.fn(),
      predict: vi.fn(),
      evaluate: vi.fn(() => Promise.resolve([0.5, 0.8])),
      dispose: vi.fn()
    })),
    compileModel: vi.fn(),
    trainModel: vi.fn(() => Promise.resolve({ history: {} })),
    predict: vi.fn(),
    stopTrainingProcess: vi.fn(),
    dispose: vi.fn(),
    getModel: vi.fn(() => ({
      summary: vi.fn()
    }))
  }
}));

describe('TrainingManager', () => {
  let trainingManager: TrainingManager;

  beforeEach(async () => {
    trainingManager = new TrainingManager();
    vi.clearAllMocks();
    
    // Reset the getDataset mock to its default implementation
    const { getDataset } = await import('../datasets');
    (getDataset as any).mockImplementation(() => {
      const mockTensor = { dispose: vi.fn() };
      return {
        loadTensors: vi.fn(() => Promise.resolve({
          trainData: mockTensor,
          trainLabels: mockTensor,
          testData: mockTensor,
          testLabels: mockTensor
        })),
        getMetadata: vi.fn(() => ({
          name: 'Mock Dataset',
          description: 'Mock dataset for testing',
          inputShape: [28, 28],
          channels: 1,
          numClasses: 10,
          trainSize: 60000,
          testSize: 10000,
          classNames: Array(10).fill('Class')
        }))
      };
    });
  });

  describe('loadDataset', () => {
    it('loads MNIST dataset', async () => {
      const { getDataset } = await import('../datasets');
      selectedDataset.set('mnist');
      await trainingManager.loadDataset();

      expect(getDataset).toHaveBeenCalledWith('mnist');
    });

    it('loads CIFAR-10 dataset', async () => {
      const { getDataset } = await import('../datasets');
      selectedDataset.set('cifar10');
      await trainingManager.loadDataset();

      expect(getDataset).toHaveBeenCalledWith('cifar10');
    });

    it('loads Fashion-MNIST dataset', async () => {
      const { getDataset } = await import('../datasets');
      selectedDataset.set('fashion-mnist');
      await trainingManager.loadDataset();

      expect(getDataset).toHaveBeenCalledWith('fashion-mnist');
    });

    it('disposes previous data before loading new dataset', async () => {
      const mockDispose = vi.fn();
      const mockTensor = { dispose: mockDispose };
      
      // Override getDataset to return tensors with dispose methods
      const { getDataset } = await import('../datasets');
      (getDataset as any).mockImplementation(() => ({
        loadTensors: vi.fn(() => Promise.resolve({
          trainData: mockTensor,
          trainLabels: mockTensor,
          testData: mockTensor,
          testLabels: mockTensor
        }))
      }));
      
      selectedDataset.set('mnist');
      await trainingManager.loadDataset();
      
      // Load different dataset to trigger disposal
      selectedDataset.set('cifar10');
      await trainingManager.loadDataset();

      // Should have disposed 4 tensors from first load
      expect(mockDispose).toHaveBeenCalledTimes(4);
    });

    it('throws error for unknown dataset', async () => {
      const { getDataset } = await import('../datasets');
      (getDataset as any).mockImplementation(() => {
        throw new Error('Unknown dataset: unknown');
      });
      
      selectedDataset.set('unknown' as any);
      await expect(trainingManager.loadDataset()).rejects.toThrow('Unknown dataset: unknown');
    });
  });

  describe('startTraining', () => {
    beforeEach(() => {
      layers.set([
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [28, 28] } },
        { id: 'dense-1', type: 'dense', name: 'Dense', params: { units: 10 } }
      ]);
      
      // Reset the console.log mock to suppress model.summary() output
      vi.spyOn(console, 'log').mockImplementation(() => {});
    });

    it('loads dataset before training', async () => {
      selectedDataset.set('mnist');
      const loadDatasetSpy = vi.spyOn(trainingManager, 'loadDataset');
      
      // Mock model evaluation to return proper tensors
      const { modelBuilder } = await import('./modelBuilder');
      const mockModel = {
        summary: vi.fn(),
        evaluate: vi.fn(() => [
          { data: () => Promise.resolve(new Float32Array([0.5])), dispose: vi.fn() },
          { data: () => Promise.resolve(new Float32Array([0.8])), dispose: vi.fn() }
        ])
      };
      (modelBuilder.buildModel as any).mockReturnValue(mockModel);
      
      try {
        await trainingManager.startTraining();
      } catch (e) {
        // Expected to fail due to mock limitations
      }
      
      expect(loadDatasetSpy).toHaveBeenCalled();
    });

    it('builds and compiles model', async () => {
      selectedDataset.set('mnist');
      const { modelBuilder } = await import('./modelBuilder');
      
      // Mock model evaluation to return proper tensors
      const mockModel = {
        summary: vi.fn(),
        evaluate: vi.fn(() => [
          { data: () => Promise.resolve(new Float32Array([0.5])), dispose: vi.fn() },
          { data: () => Promise.resolve(new Float32Array([0.8])), dispose: vi.fn() }
        ])
      };
      (modelBuilder.buildModel as any).mockReturnValue(mockModel);
      
      await trainingManager.startTraining();
      
      expect(modelBuilder.buildModel).toHaveBeenCalled();
      expect(modelBuilder.compileModel).toHaveBeenCalled();
    });

    it('trains model with loaded data', async () => {
      selectedDataset.set('mnist');
      const { modelBuilder } = await import('./modelBuilder');
      
      // Mock model evaluation to return proper tensors
      const mockModel = {
        summary: vi.fn(),
        evaluate: vi.fn(() => [
          { data: () => Promise.resolve(new Float32Array([0.5])), dispose: vi.fn() },
          { data: () => Promise.resolve(new Float32Array([0.8])), dispose: vi.fn() }
        ])
      };
      (modelBuilder.buildModel as any).mockReturnValue(mockModel);
      
      await trainingManager.startTraining();
      
      expect(modelBuilder.trainModel).toHaveBeenCalled();
    });

    it('handles training completion', async () => {
      selectedDataset.set('mnist');
      const { modelBuilder } = await import('./modelBuilder');
      
      // Mock model evaluation to return proper tensors
      const mockModel = {
        summary: vi.fn(),
        evaluate: vi.fn(() => [
          { data: () => Promise.resolve(new Float32Array([0.5])), dispose: vi.fn() },
          { data: () => Promise.resolve(new Float32Array([0.8])), dispose: vi.fn() }
        ])
      };
      (modelBuilder.buildModel as any).mockReturnValue(mockModel);
      
      // Ensure trainModel resolves successfully
      (modelBuilder.trainModel as any).mockResolvedValue({ history: {} });
      
      await trainingManager.startTraining();
      
      // The training should be called when model is built successfully
      expect(modelBuilder.trainModel).toHaveBeenCalled();
    });

    it('handles epoch end callback', async () => {
      selectedDataset.set('mnist');
      const { modelBuilder } = await import('./modelBuilder');
      
      // Mock model evaluation to return proper tensors
      const mockModel = {
        summary: vi.fn(),
        evaluate: vi.fn(() => [
          { data: () => Promise.resolve(new Float32Array([0.5])), dispose: vi.fn() },
          { data: () => Promise.resolve(new Float32Array([0.8])), dispose: vi.fn() }
        ])
      };
      (modelBuilder.buildModel as any).mockReturnValue(mockModel);
      
      // Mock trainModel to call the callback
      let capturedCallback: any;
      (modelBuilder.trainModel as any).mockImplementation(async (_data: any, _labels: any, _config: any, callback: any) => {
        capturedCallback = callback;
        // Simulate epoch end callback
        if (callback) {
          callback(0, { loss: 0.5, acc: 0.8 });
        }
        return { history: {} };
      });
      
      await trainingManager.startTraining();
      
      // Verify that trainModel was called with a callback
      expect(modelBuilder.trainModel).toHaveBeenCalled();
      // Verify the callback was captured and called
      expect(capturedCallback).toBeDefined();
    });
  });


  describe('stopTraining', () => {
    it('calls modelBuilder stopTrainingProcess', async () => {
      const { modelBuilder } = await import('./modelBuilder');
      
      trainingManager.stopTraining();
      
      expect(modelBuilder.stopTrainingProcess).toHaveBeenCalled();
    });
  });

  describe('dispose', () => {
    it('disposes all tensors and model builder', async () => {
      const mockDispose = vi.fn();
      const mockTensor = { dispose: mockDispose };
      
      // Override getDataset to return tensors with dispose methods
      const { getDataset } = await import('../datasets');
      (getDataset as any).mockImplementation(() => ({
        loadTensors: vi.fn(() => Promise.resolve({
          trainData: mockTensor,
          trainLabels: mockTensor,
          testData: mockTensor,
          testLabels: mockTensor
        }))
      }));

      selectedDataset.set('mnist');
      await trainingManager.loadDataset();
      
      const { modelBuilder } = await import('./modelBuilder');
      
      trainingManager.dispose();
      
      expect(mockDispose).toHaveBeenCalledTimes(4); // 4 tensors
      expect(modelBuilder.dispose).toHaveBeenCalled();
    });
  });

  describe('dynamic output layer handling', () => {
    beforeEach(() => {
      // Reset console.warn mock to capture warnings
      vi.spyOn(console, 'warn').mockImplementation(() => {});
      vi.spyOn(console, 'log').mockImplementation(() => {});
    });

    it('adds output layer with correct units for IMDB dataset (2 classes)', async () => {
      const { getDataset } = await import('../datasets');
      (getDataset as any).mockImplementation(() => ({
        loadTensors: vi.fn(() => Promise.resolve({
          trainData: { dispose: vi.fn() },
          trainLabels: { dispose: vi.fn() },
          testData: { dispose: vi.fn() },
          testLabels: { dispose: vi.fn() }
        })),
        getMetadata: vi.fn(() => ({
          name: 'IMDB Movie Reviews',
          description: 'Binary sentiment classification',
          inputShape: [200],
          channels: 1,
          numClasses: 2,
          trainSize: 20000,
          testSize: 5000,
          classNames: ['Negative', 'Positive']
        }))
      }));

      selectedDataset.set('imdb');
      layers.set([
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [200] } },
        { id: 'embedding-1', type: 'embedding', name: 'Embedding', params: { vocabSize: 10000, embeddingDim: 128, maxLength: 200 } },
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'dense-1', type: 'dense', name: 'Dense', params: { units: 64, activation: 'relu' } }
      ]);

      const { modelBuilder } = await import('./modelBuilder');
      const mockModel = {
        summary: vi.fn(),
        evaluate: vi.fn(() => [
          { data: () => Promise.resolve(new Float32Array([0.5])), dispose: vi.fn() },
          { data: () => Promise.resolve(new Float32Array([0.8])), dispose: vi.fn() }
        ])
      };
      
      let capturedLayerConfigs: any;
      (modelBuilder.buildModel as any).mockImplementation((configs: any) => {
        capturedLayerConfigs = [...configs];
        return mockModel;
      });

      await trainingManager.startTraining();

      // Check that output layer was added with 2 units
      expect(console.warn).toHaveBeenCalledWith('Adding output layer for 2-class classification');
      expect(capturedLayerConfigs).toHaveLength(5);
      expect(capturedLayerConfigs[4]).toMatchObject({
        id: 'output-auto',
        type: 'dense',
        params: {
          units: 2,
          activation: 'softmax'
        }
      });
    });

    it('adds output layer with correct units for AG News dataset (4 classes)', async () => {
      const { getDataset } = await import('../datasets');
      (getDataset as any).mockImplementation(() => ({
        loadTensors: vi.fn(() => Promise.resolve({
          trainData: { dispose: vi.fn() },
          trainLabels: { dispose: vi.fn() },
          testData: { dispose: vi.fn() },
          testLabels: { dispose: vi.fn() }
        })),
        getMetadata: vi.fn(() => ({
          name: 'AG News',
          description: '4-class news categorization',
          inputShape: [150],
          channels: 1,
          numClasses: 4,
          trainSize: 20000,
          testSize: 5000,
          classNames: ['World', 'Sports', 'Business', 'Science/Tech']
        }))
      }));

      selectedDataset.set('ag-news');
      layers.set([
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [150] } },
        { id: 'dense-1', type: 'dense', name: 'Dense', params: { units: 128, activation: 'relu' } }
      ]);

      const { modelBuilder } = await import('./modelBuilder');
      const mockModel = {
        summary: vi.fn(),
        evaluate: vi.fn(() => [
          { data: () => Promise.resolve(new Float32Array([0.5])), dispose: vi.fn() },
          { data: () => Promise.resolve(new Float32Array([0.8])), dispose: vi.fn() }
        ])
      };
      
      let capturedLayerConfigs: any;
      (modelBuilder.buildModel as any).mockImplementation((configs: any) => {
        capturedLayerConfigs = [...configs];
        return mockModel;
      });

      await trainingManager.startTraining();

      // Check that output layer was added with 4 units
      expect(console.warn).toHaveBeenCalledWith('Adding output layer for 4-class classification');
      expect(capturedLayerConfigs).toHaveLength(3);
      expect(capturedLayerConfigs[2]).toMatchObject({
        id: 'output-auto',
        type: 'dense',
        params: {
          units: 4,
          activation: 'softmax'
        }
      });
    });

    it('does not add output layer if last layer has correct units', async () => {
      const { getDataset } = await import('../datasets');
      (getDataset as any).mockImplementation(() => ({
        loadTensors: vi.fn(() => Promise.resolve({
          trainData: { dispose: vi.fn() },
          trainLabels: { dispose: vi.fn() },
          testData: { dispose: vi.fn() },
          testLabels: { dispose: vi.fn() }
        })),
        getMetadata: vi.fn(() => ({
          name: 'MNIST',
          description: '10-class digit classification',
          inputShape: [28, 28],
          channels: 1,
          numClasses: 10,
          trainSize: 60000,
          testSize: 10000,
          classNames: Array(10).fill('Digit')
        }))
      }));

      selectedDataset.set('mnist');
      layers.set([
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [28, 28] } },
        { id: 'dense-1', type: 'dense', name: 'Dense', params: { units: 10, activation: 'softmax' } }
      ]);

      const { modelBuilder } = await import('./modelBuilder');
      const mockModel = {
        summary: vi.fn(),
        evaluate: vi.fn(() => [
          { data: () => Promise.resolve(new Float32Array([0.5])), dispose: vi.fn() },
          { data: () => Promise.resolve(new Float32Array([0.8])), dispose: vi.fn() }
        ])
      };
      
      let capturedLayerConfigs: any;
      (modelBuilder.buildModel as any).mockImplementation((configs: any) => {
        capturedLayerConfigs = [...configs];
        return mockModel;
      });

      await trainingManager.startTraining();

      // Check that no output layer was added
      expect(console.warn).not.toHaveBeenCalled();
      expect(capturedLayerConfigs).toHaveLength(2);
    });

    it('handles text classification model architecture correctly', async () => {
      const { getDataset } = await import('../datasets');
      (getDataset as any).mockImplementation(() => ({
        loadTensors: vi.fn(() => Promise.resolve({
          trainData: { dispose: vi.fn() },
          trainLabels: { dispose: vi.fn() },
          testData: { dispose: vi.fn() },
          testLabels: { dispose: vi.fn() }
        })),
        getMetadata: vi.fn(() => ({
          name: 'IMDB Movie Reviews',
          description: 'Binary sentiment classification',
          inputShape: [200],
          channels: 1,
          numClasses: 2,
          trainSize: 20000,
          testSize: 5000,
          classNames: ['Negative', 'Positive']
        }))
      }));

      selectedDataset.set('imdb');
      // Typical text classification architecture
      layers.set([
        { id: 'input-1', type: 'input', name: 'Input', params: { shape: [200] } },
        { id: 'embedding-1', type: 'embedding', name: 'Embedding', params: { vocabSize: 10000, embeddingDim: 128, maxLength: 200 } },
        { id: 'flatten-1', type: 'flatten', name: 'Flatten', params: {} },
        { id: 'dense-1', type: 'dense', name: 'Dense 1', params: { units: 64, activation: 'relu' } },
        { id: 'dropout-1', type: 'dropout', name: 'Dropout', params: { rate: 0.5 } },
        { id: 'dense-2', type: 'dense', name: 'Output', params: { units: 2, activation: 'softmax' } }
      ]);

      const { modelBuilder } = await import('./modelBuilder');
      const mockModel = {
        summary: vi.fn(),
        evaluate: vi.fn(() => [
          { data: () => Promise.resolve(new Float32Array([0.5])), dispose: vi.fn() },
          { data: () => Promise.resolve(new Float32Array([0.8])), dispose: vi.fn() }
        ])
      };
      
      let capturedLayerConfigs: any;
      (modelBuilder.buildModel as any).mockImplementation((configs: any) => {
        capturedLayerConfigs = [...configs];
        return mockModel;
      });

      await trainingManager.startTraining();

      // Check that no additional output layer was added since last layer already has correct units
      expect(console.warn).not.toHaveBeenCalled();
      expect(capturedLayerConfigs).toHaveLength(6);
      expect(capturedLayerConfigs[5].params.units).toBe(2);
    });
  });

});
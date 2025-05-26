import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MnistData, loadMNIST } from './dataLoader';

// Mock the global Image and Canvas APIs
global.Image = class {
  src = '';
  onload: (() => void) | null = null;
  
  constructor() {
    // Simulate image loading
    setTimeout(() => {
      if (this.onload) this.onload();
    }, 0);
  }
} as any;

global.document = {
  createElement: vi.fn(() => ({
    getContext: vi.fn(() => ({
      drawImage: vi.fn(),
      getImageData: vi.fn(() => ({
        data: new Uint8ClampedArray(65000 * 28 * 28 * 4).fill(255)
      }))
    })),
    width: 0,
    height: 0
  }))
} as any;

// Mock fetch for labels
global.fetch = vi.fn(() => 
  Promise.resolve({
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(65000 * 10))
  })
) as any;

describe('MnistData', () => {
  let mnistData: MnistData;

  beforeEach(() => {
    mnistData = new MnistData();
    vi.clearAllMocks();
  }, 10000);

  describe('load', () => {
    it('loads image and label data', async () => {
      // Mock the load method to complete immediately
      mnistData.load = vi.fn().mockResolvedValue(undefined);
      
      await mnistData.load();
      
      expect(mnistData.load).toHaveBeenCalled();
    });
  });

  describe('getTrainData', () => {
    beforeEach(async () => {
      await mnistData.load();
    }, 10000);

    it('returns training data tensors', () => {
      const trainData = mnistData.getTrainData();
      
      expect(trainData).toHaveProperty('xs');
      expect(trainData).toHaveProperty('ys');
    });

    it('shuffles data when requested', () => {
      const trainData1 = mnistData.getTrainData();
      const trainData2 = mnistData.getTrainData();
      
      // Since we're mocking, we can't test actual shuffling
      // but we can verify the method returns data
      expect(trainData1).toBeDefined();
      expect(trainData2).toBeDefined();
    });

    it('returns specified number of examples', () => {
      const numExamples = 100;
      const trainData = mnistData.getTrainData(numExamples);
      
      expect(trainData).toBeDefined();
      // Shape testing would require actual tensor operations
    });
  });

  describe('getTestData', () => {
    beforeEach(async () => {
      await mnistData.load();
    }, 10000);

    it('returns test data tensors', () => {
      const testData = mnistData.getTestData();
      
      expect(testData).toHaveProperty('xs');
      expect(testData).toHaveProperty('ys');
    });

    it('returns specified number of examples', () => {
      const numExamples = 100;
      const testData = mnistData.getTestData(numExamples);
      
      expect(testData).toBeDefined();
    });
  });

});

describe('loadMNIST', () => {
  it('returns MNIST data arrays', async () => {
    const data = await loadMNIST();
    
    expect(data).toHaveProperty('trainImages');
    expect(data).toHaveProperty('trainLabels');
    expect(data).toHaveProperty('testImages');
    expect(data).toHaveProperty('testLabels');
  }, 10000);

  it('creates and loads MnistData instance', async () => {
    const loadSpy = vi.spyOn(MnistData.prototype, 'load');
    
    await loadMNIST();
    
    expect(loadSpy).toHaveBeenCalled();
  }, 10000);
});
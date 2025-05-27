import { describe, it, expect, beforeEach } from 'vitest';
import { AGNewsDataset } from './agNewsDataset';

describe('AGNewsDataset', () => {
  let dataset: AGNewsDataset;

  beforeEach(() => {
    dataset = new AGNewsDataset();
  });

  it('should create dataset with correct metadata', () => {
    const metadata = dataset.getMetadata();
    
    expect(metadata.name).toBe('AG News');
    expect(metadata.description).toBe('4-class news categorization dataset');
    expect(metadata.inputShape).toEqual([150]);
    expect(metadata.channels).toBe(1);
    expect(metadata.numClasses).toBe(4);
    expect(metadata.trainSize).toBe(20000);
    expect(metadata.testSize).toBe(5000);
    expect(metadata.classNames).toEqual(['World', 'Sports', 'Business', 'Science/Tech']);
  });

  it('should load data successfully', async () => {
    const data = await dataset.loadData();
    
    expect(data).toBeDefined();
    expect(data.trainImages).toBeInstanceOf(Float32Array);
    expect(data.trainLabels).toBeInstanceOf(Float32Array);
    expect(data.testImages).toBeInstanceOf(Float32Array);
    expect(data.testLabels).toBeInstanceOf(Float32Array);
    
    // Check data sizes
    const metadata = dataset.getMetadata();
    expect(data.trainImages.length).toBe(metadata.trainSize * metadata.inputShape[0]);
    expect(data.trainLabels.length).toBe(metadata.trainSize * metadata.numClasses);
    expect(data.testImages.length).toBe(metadata.testSize * metadata.inputShape[0]);
    expect(data.testLabels.length).toBe(metadata.testSize * metadata.numClasses);
  });

  it('should generate balanced classes across 4 categories', async () => {
    const data = await dataset.loadData();
    
    // Count samples in each class
    const classCounts = [0, 0, 0, 0];
    
    for (let i = 0; i < data.trainLabels.length; i += 4) {
      if (data.trainLabels[i] === 1) classCounts[0]++;      // World
      if (data.trainLabels[i + 1] === 1) classCounts[1]++;  // Sports
      if (data.trainLabels[i + 2] === 1) classCounts[2]++;  // Business
      if (data.trainLabels[i + 3] === 1) classCounts[3]++;  // Science/Tech
    }
    
    // Should be perfectly balanced in synthetic data (cycling through classes)
    const expectedCount = 20000 / 4; // 5000 per class
    expect(classCounts[0]).toBe(expectedCount);
    expect(classCounts[1]).toBe(expectedCount);
    expect(classCounts[2]).toBe(expectedCount);
    expect(classCounts[3]).toBe(expectedCount);
  });

  it('should handle shuffle option', async () => {
    // Create fresh dataset instances to avoid caching issues
    const dataset1 = new AGNewsDataset();
    const dataset2 = new AGNewsDataset();
    
    const dataOriginal = await dataset1.loadData({ shuffle: false, cache: false });
    const dataShuffled = await dataset2.loadData({ shuffle: true, seed: 42, cache: false });
    
    expect(dataOriginal.trainImages.length).toBe(dataShuffled.trainImages.length);
    expect(dataOriginal.trainLabels.length).toBe(dataShuffled.trainLabels.length);
    
    // Data should be different after shuffling
    // Check sequences (groups of 150 tokens) rather than individual tokens
    let isDifferent = false;
    for (let i = 0; i < Math.min(5, Math.floor(dataOriginal.trainImages.length / 150)); i++) {
      const start = i * 150;
      const originalSeq = Array.from(dataOriginal.trainImages.slice(start, start + 10));
      const shuffledSeq = Array.from(dataShuffled.trainImages.slice(start, start + 10));
      
      if (JSON.stringify(originalSeq) !== JSON.stringify(shuffledSeq)) {
        isDifferent = true;
        break;
      }
    }
    expect(isDifferent).toBe(true);
  });

  it('should cache data when requested', async () => {
    // Load with caching enabled
    const data1 = await dataset.loadData({ cache: true });
    const data2 = await dataset.loadData({ cache: true });
    
    // Should return the same cached instance
    expect(data1.trainImages).toBe(data2.trainImages);
    expect(data1.trainLabels).toBe(data2.trainLabels);
  });

  it('should load tensors with correct shapes', async () => {
    const tensors = await dataset.loadTensors();
    
    expect(tensors.trainData.shape).toEqual([20000, 150]);
    expect(tensors.trainLabels.shape).toEqual([20000, 4]);
    expect(tensors.testData.shape).toEqual([5000, 150]);
    expect(tensors.testLabels.shape).toEqual([5000, 4]);
    
    // Clean up tensors
    tensors.trainData.dispose();
    tensors.trainLabels.dispose();
    tensors.testData.dispose();
    tensors.testLabels.dispose();
  });

  it('should validate data ranges', async () => {
    const data = await dataset.loadData();
    
    // Check that image data is in valid range (token indices should be positive integers)
    for (let i = 0; i < Math.min(1000, data.trainImages.length); i++) {
      expect(data.trainImages[i]).toBeGreaterThanOrEqual(0);
      expect(Number.isInteger(data.trainImages[i])).toBe(true);
    }
    
    // Check that labels are one-hot encoded (0 or 1)
    for (let i = 0; i < Math.min(1000, data.trainLabels.length); i++) {
      expect(data.trainLabels[i]).toBeGreaterThanOrEqual(0);
      expect(data.trainLabels[i]).toBeLessThanOrEqual(1);
    }
  });

  it('should generate different patterns for different categories', async () => {
    const data = await dataset.loadData();
    
    // Get samples from different categories
    const worldSample = Array.from(data.trainImages.slice(0, 150));        // Category 0
    const sportsSample = Array.from(data.trainImages.slice(150, 300));     // Category 1
    const businessSample = Array.from(data.trainImages.slice(300, 450));   // Category 2
    const techSample = Array.from(data.trainImages.slice(450, 600));       // Category 3
    
    // Should be different patterns for different categories
    let worldVsSports = false;
    let sportsVsBusiness = false;
    let businessVsTech = false;
    
    for (let i = 0; i < 150; i++) {
      if (worldSample[i] !== sportsSample[i]) worldVsSports = true;
      if (sportsSample[i] !== businessSample[i]) sportsVsBusiness = true;
      if (businessSample[i] !== techSample[i]) businessVsTech = true;
    }
    
    expect(worldVsSports).toBe(true);
    expect(sportsVsBusiness).toBe(true);
    expect(businessVsTech).toBe(true);
  });

  it('should handle loading options correctly', async () => {
    const tensors = await dataset.loadTensors({
      shuffle: true,
      cache: false,
      seed: 123
    });
    
    expect(tensors.trainData.shape[0]).toBe(20000);
    expect(tensors.trainLabels.shape[0]).toBe(20000);
    
    // Clean up tensors
    tensors.trainData.dispose();
    tensors.trainLabels.dispose();
    tensors.testData.dispose();
    tensors.testLabels.dispose();
  });

  it('should clear cache correctly', () => {
    expect(() => dataset.clearCache()).not.toThrow();
  });

  it('should use shorter sequence length than IMDB', () => {
    const metadata = dataset.getMetadata();
    expect(metadata.inputShape[0]).toBe(150); // Shorter than IMDB's 200
  });

  it('should have 4 classes vs IMDB\'s 2 classes', () => {
    const metadata = dataset.getMetadata();
    expect(metadata.numClasses).toBe(4);
    expect(metadata.classNames.length).toBe(4);
  });

  it('should generate one-hot encoded labels correctly', async () => {
    const data = await dataset.loadData();
    
    // Check first few samples to ensure proper one-hot encoding
    for (let sampleIdx = 0; sampleIdx < 10; sampleIdx++) {
      const labelStart = sampleIdx * 4;
      const labelSlice = Array.from(data.trainLabels.slice(labelStart, labelStart + 4));
      
      // Should have exactly one 1 and three 0s
      const sum = labelSlice.reduce((a, b) => a + b, 0);
      expect(sum).toBe(1);
      
      const onesCount = labelSlice.filter(x => x === 1).length;
      const zerosCount = labelSlice.filter(x => x === 0).length;
      expect(onesCount).toBe(1);
      expect(zerosCount).toBe(3);
    }
  });
});
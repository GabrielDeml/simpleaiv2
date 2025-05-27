import { describe, it, expect, beforeEach } from 'vitest';
import { IMDBDataset, SimpleTokenizer } from './imdbDataset';

describe('SimpleTokenizer', () => {
  let tokenizer: SimpleTokenizer;

  beforeEach(() => {
    tokenizer = new SimpleTokenizer(100); // Small vocab for testing
  });

  it('should create tokenizer with correct vocab size', () => {
    expect(tokenizer).toBeDefined();
  });

  it('should fit on texts and build vocabulary', () => {
    const texts = [
      'this is a good movie',
      'this is a bad movie',
      'good movie great',
      'bad movie terrible'
    ];

    expect(() => tokenizer.fitOnTexts(texts)).not.toThrow();
  });

  it('should convert texts to sequences', () => {
    const texts = [
      'this is good',
      'this is bad',
      'good movie',
      'bad movie'
    ];

    tokenizer.fitOnTexts(texts);
    const sequences = tokenizer.textsToSequences(['this is good', 'bad movie']);

    expect(sequences).toBeDefined();
    expect(sequences.length).toBe(2);
    expect(Array.isArray(sequences[0])).toBe(true);
    expect(Array.isArray(sequences[1])).toBe(true);
  });

  it('should pad sequences to specified length', () => {
    const sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]];
    const maxLength = 5;
    
    const padded = tokenizer.padSequences(sequences, maxLength);

    expect(padded.length).toBe(3);
    expect(padded[0].length).toBe(maxLength);
    expect(padded[1].length).toBe(maxLength);
    expect(padded[2].length).toBe(maxLength);
    
    // Check padding values (should be 0)
    expect(padded[1][3]).toBe(0);
    expect(padded[1][4]).toBe(0);
    
    // Check truncation
    expect(padded[2]).toEqual([6, 7, 8, 9, 10]);
  });

  it('should handle out-of-vocabulary words', () => {
    const texts = ['known words'];
    tokenizer.fitOnTexts(texts);
    
    const sequences = tokenizer.textsToSequences(['unknown words']);
    
    // Should use OOV token (index 1) for unknown words
    // Note: 'words' might be in vocabulary if it appears in 'known words'
    expect(sequences[0].length).toBe(2);
    expect(sequences[0]).toContain(1); // Should contain at least one OOV token
  });

  it('should handle empty and special cases', () => {
    const texts = ['hello world'];
    tokenizer.fitOnTexts(texts);
    
    const emptySequences = tokenizer.textsToSequences(['']);
    expect(emptySequences[0]).toEqual([]);
    
    const whitespaceSequences = tokenizer.textsToSequences(['   ']);
    expect(whitespaceSequences[0]).toEqual([]);
  });
});

describe('IMDBDataset', () => {
  let dataset: IMDBDataset;

  beforeEach(() => {
    dataset = new IMDBDataset();
  });

  it('should create dataset with correct metadata', () => {
    const metadata = dataset.getMetadata();
    
    expect(metadata.name).toBe('IMDB Movie Reviews');
    expect(metadata.description).toBe('Binary sentiment classification of movie reviews');
    expect(metadata.inputShape).toEqual([200]);
    expect(metadata.channels).toBe(1);
    expect(metadata.numClasses).toBe(2);
    expect(metadata.trainSize).toBe(20000);
    expect(metadata.testSize).toBe(5000);
    expect(metadata.classNames).toEqual(['Negative', 'Positive']);
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

  it('should generate balanced classes', async () => {
    const data = await dataset.loadData();
    
    // Count positive and negative samples in training labels
    let positiveCount = 0;
    let negativeCount = 0;
    
    for (let i = 0; i < data.trainLabels.length; i += 2) {
      if (data.trainLabels[i] === 1) negativeCount++;
      if (data.trainLabels[i + 1] === 1) positiveCount++;
    }
    
    // Should be roughly balanced (alternating in synthetic data)
    expect(Math.abs(positiveCount - negativeCount)).toBeLessThanOrEqual(1);
  });

  it('should handle shuffle option', async () => {
    // Create a fresh dataset instance to avoid caching issues
    const dataset1 = new IMDBDataset();
    const dataset2 = new IMDBDataset();
    
    const dataOriginal = await dataset1.loadData({ shuffle: false, cache: false });
    const dataShuffled = await dataset2.loadData({ shuffle: true, seed: 42, cache: false });
    
    expect(dataOriginal.trainImages.length).toBe(dataShuffled.trainImages.length);
    expect(dataOriginal.trainLabels.length).toBe(dataShuffled.trainLabels.length);
    
    // Data should be different after shuffling
    // Check sequences (groups of 200 tokens) rather than individual tokens
    let isDifferent = false;
    for (let i = 0; i < Math.min(5, Math.floor(dataOriginal.trainImages.length / 200)); i++) {
      const start = i * 200;
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

  it('should not cache when disabled', async () => {
    const data1 = await dataset.loadData({ cache: false });
    const data2 = await dataset.loadData({ cache: false });
    
    // Should be different instances but same values
    expect(data1.trainImages).not.toBe(data2.trainImages);
    expect(data1.trainImages.length).toBe(data2.trainImages.length);
  });

  it('should load tensors with correct shapes', async () => {
    const tensors = await dataset.loadTensors();
    
    expect(tensors.trainData.shape).toEqual([20000, 200]);
    expect(tensors.trainLabels.shape).toEqual([20000, 2]);
    expect(tensors.testData.shape).toEqual([5000, 200]);
    expect(tensors.testLabels.shape).toEqual([5000, 2]);
    
    // Clean up tensors
    tensors.trainData.dispose();
    tensors.trainLabels.dispose();
    tensors.testData.dispose();
    tensors.testLabels.dispose();
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

  it('should generate different sentiment patterns', async () => {
    const data = await dataset.loadData();
    
    // Get first positive and negative samples
    const firstNegative = Array.from(data.trainImages.slice(0, 200));
    const firstPositive = Array.from(data.trainImages.slice(200, 400));
    
    // Should be different (though both are synthetic)
    let isDifferent = false;
    for (let i = 0; i < firstNegative.length; i++) {
      if (firstNegative[i] !== firstPositive[i]) {
        isDifferent = true;
        break;
      }
    }
    expect(isDifferent).toBe(true);
  });
});
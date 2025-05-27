import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import { MultiHeadAttentionLayer, PositionalEncodingLayer, TransformerEncoderBlock } from './transformerLayers';

// Mock console methods to avoid noise in tests
beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'log').mockImplementation(() => {});
});

describe('MultiHeadAttentionLayer', () => {
  let layer: MultiHeadAttentionLayer;
  
  beforeEach(() => {
    layer = new MultiHeadAttentionLayer({
      numHeads: 2,
      keyDim: 4
    });
  });

  it('should create a layer with correct configuration', () => {
    expect(layer).toBeDefined();
    const config = layer.getConfig();
    expect(config.numHeads).toBe(2);
    expect(config.keyDim).toBe(4);
    expect(config.valueDim).toBe(4); // defaults to keyDim
    expect(config.dropout).toBe(0.0);
    expect(config.useBias).toBe(true);
  });

  it('should build correctly with proper input shape', () => {
    const inputShape = [10, 8]; // [seqLength, embeddingDim]
    expect(() => layer.build(inputShape)).not.toThrow();
  });

  it('should compute output shape correctly', () => {
    const inputShape = [10, 8];
    const outputShape = layer.computeOutputShape(inputShape);
    expect(outputShape).toEqual(inputShape);
  });

  it('should process input tensors correctly', () => {
    const batchSize = 2;
    const seqLength = 3;
    const embeddingDim = 8;
    
    layer.build([seqLength, embeddingDim]);
    
    const input = tf.randomNormal([batchSize, seqLength, embeddingDim]);
    const output = layer.apply(input) as tf.Tensor;
    
    expect(output.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    input.dispose();
    output.dispose();
  });

  it('should handle training parameter correctly', () => {
    const batchSize = 2;
    const seqLength = 3;
    const embeddingDim = 8;
    
    layer.build([seqLength, embeddingDim]);
    
    const input = tf.randomNormal([batchSize, seqLength, embeddingDim]);
    
    // Test with training=true
    const outputTrain = layer.apply(input, { training: true }) as tf.Tensor;
    expect(outputTrain.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    // Test with training=false
    const outputEval = layer.apply(input, { training: false }) as tf.Tensor;
    expect(outputEval.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    input.dispose();
    outputTrain.dispose();
    outputEval.dispose();
  });
});

describe('PositionalEncodingLayer', () => {
  let layer: PositionalEncodingLayer;
  
  beforeEach(() => {
    layer = new PositionalEncodingLayer({
      maxLength: 100
    });
  });

  it('should create a layer with correct configuration', () => {
    expect(layer).toBeDefined();
    const config = layer.getConfig();
    expect(config.maxLength).toBe(100);
  });

  it('should build correctly with proper input shape', () => {
    const inputShape = [50, 128]; // [seqLength, embeddingDim]
    expect(() => layer.build(inputShape)).not.toThrow();
  });

  it('should compute output shape correctly', () => {
    const inputShape = [50, 128];
    const outputShape = layer.computeOutputShape(inputShape);
    expect(outputShape).toEqual(inputShape);
  });

  it('should add positional encoding to input', () => {
    const batchSize = 2;
    const seqLength = 10;
    const embeddingDim = 16;
    
    layer.build([seqLength, embeddingDim]);
    
    const input = tf.zeros([batchSize, seqLength, embeddingDim]);
    const output = layer.apply(input) as tf.Tensor;
    
    expect(output.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    // The output should not be all zeros after adding positional encoding
    const outputData = output.dataSync();
    const hasNonZero = outputData.some(val => Math.abs(val) > 1e-6);
    expect(hasNonZero).toBe(true);
    
    input.dispose();
    output.dispose();
  });

  it('should handle sequences shorter than maxLength', () => {
    const batchSize = 1;
    const seqLength = 5; // shorter than maxLength
    const embeddingDim = 8;
    
    layer.build([100, embeddingDim]); // build with maxLength
    
    const input = tf.zeros([batchSize, seqLength, embeddingDim]);
    const output = layer.apply(input) as tf.Tensor;
    
    expect(output.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    input.dispose();
    output.dispose();
  });
});

describe('TransformerEncoderBlock', () => {
  let layer: TransformerEncoderBlock;
  
  beforeEach(() => {
    layer = new TransformerEncoderBlock({
      numHeads: 2,
      keyDim: 4,
      ffDim: 16,
      dropout: 0.1
    });
  });

  it('should create a layer with correct configuration', () => {
    expect(layer).toBeDefined();
    const config = layer.getConfig();
    expect(config.numHeads).toBe(2);
    expect(config.keyDim).toBe(4);
    expect(config.ffDim).toBe(16);
    expect(config.dropout).toBe(0.1);
  });

  it('should build correctly with proper input shape', () => {
    const inputShape = [10, 8]; // [seqLength, embeddingDim]
    expect(() => layer.build(inputShape)).not.toThrow();
  });

  it('should compute output shape correctly', () => {
    const inputShape = [10, 8];
    const outputShape = layer.computeOutputShape(inputShape);
    expect(outputShape).toEqual(inputShape);
  });

  it('should process input tensors correctly', () => {
    const batchSize = 2;
    const seqLength = 4;
    const embeddingDim = 8;
    
    layer.build([seqLength, embeddingDim]);
    
    const input = tf.randomNormal([batchSize, seqLength, embeddingDim]);
    const output = layer.apply(input) as tf.Tensor;
    
    expect(output.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    input.dispose();
    output.dispose();
  });

  it('should handle training parameter correctly', () => {
    const batchSize = 1;
    const seqLength = 3;
    const embeddingDim = 8;
    
    layer.build([seqLength, embeddingDim]);
    
    const input = tf.randomNormal([batchSize, seqLength, embeddingDim]);
    
    // Test with training=true
    const outputTrain = layer.apply(input, { training: true }) as tf.Tensor;
    expect(outputTrain.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    // Test with training=false
    const outputEval = layer.apply(input, { training: false }) as tf.Tensor;
    expect(outputEval.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    input.dispose();
    outputTrain.dispose();
    outputEval.dispose();
  });

  it('should implement residual connections correctly', () => {
    const batchSize = 1;
    const seqLength = 2;
    const embeddingDim = 8;
    
    layer.build([seqLength, embeddingDim]);
    
    // Create input with known values
    const inputValues = new Float32Array(batchSize * seqLength * embeddingDim);
    for (let i = 0; i < inputValues.length; i++) {
      inputValues[i] = 0.1; // small constant value
    }
    const input = tf.tensor3d(inputValues, [batchSize, seqLength, embeddingDim]);
    
    const output = layer.apply(input, { training: false }) as tf.Tensor;
    
    // Output should have same shape as input
    expect(output.shape).toEqual([batchSize, seqLength, embeddingDim]);
    
    // Due to residual connections and layer normalization, 
    // output should not be exactly the same as input but should be in reasonable range
    const outputData = output.dataSync();
    const avgOutput = Array.from(outputData).reduce((a: number, b: number) => a + b, 0) / outputData.length;
    expect(Math.abs(avgOutput)).toBeLessThan(10); // reasonable range
    
    input.dispose();
    output.dispose();
  });
});

describe('Layer Registration', () => {
  it('should register custom layers with TensorFlow.js', () => {
    // Test that our custom layers are registered and can be serialized
    const multiHeadLayer = new MultiHeadAttentionLayer({
      numHeads: 4,
      keyDim: 8
    });
    
    const positionalLayer = new PositionalEncodingLayer({
      maxLength: 50
    });
    
    const transformerLayer = new TransformerEncoderBlock({
      numHeads: 2,
      keyDim: 4,
      ffDim: 16
    });
    
    // Test that getConfig works (needed for serialization)
    expect(multiHeadLayer.getConfig()).toBeDefined();
    expect(positionalLayer.getConfig()).toBeDefined();
    expect(transformerLayer.getConfig()).toBeDefined();
    
    // Test that class names are set correctly
    expect(MultiHeadAttentionLayer.className).toBe('MultiHeadAttentionLayer');
    expect(PositionalEncodingLayer.className).toBe('PositionalEncodingLayer');
    expect(TransformerEncoderBlock.className).toBe('TransformerEncoderBlock');
  });
});

describe('Edge Cases and Error Handling', () => {
  it('should handle invalid input shapes gracefully', () => {
    const layer = new MultiHeadAttentionLayer({
      numHeads: 2,
      keyDim: 4
    });
    
    // Test with invalid input shape
    expect(() => layer.build(42 as any)).toThrow('Invalid input shape');
    expect(() => layer.build(null as any)).toThrow('Invalid input shape');
  });

  it('should validate numHeads configuration', () => {
    // Test with valid configuration
    expect(() => new MultiHeadAttentionLayer({
      numHeads: 1,
      keyDim: 4
    })).not.toThrow();
    
    expect(() => new MultiHeadAttentionLayer({
      numHeads: 8,
      keyDim: 64
    })).not.toThrow();
  });

  it('should handle dropout configuration correctly', () => {
    const layerWithDropout = new MultiHeadAttentionLayer({
      numHeads: 2,
      keyDim: 4,
      dropout: 0.5
    });
    
    const layerWithoutDropout = new MultiHeadAttentionLayer({
      numHeads: 2,
      keyDim: 4,
      dropout: 0.0
    });
    
    expect(layerWithDropout.getConfig().dropout).toBe(0.5);
    expect(layerWithoutDropout.getConfig().dropout).toBe(0.0);
  });
});
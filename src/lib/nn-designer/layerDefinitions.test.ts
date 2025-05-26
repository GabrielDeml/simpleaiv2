import { describe, it, expect } from 'vitest';
import { layerDefinitions } from './layerDefinitions';
import type { LayerType } from './types';

describe('layerDefinitions', () => {
  const expectedLayerTypes: LayerType[] = ['input', 'dense', 'conv2d', 'maxpooling2d', 'dropout', 'flatten'];

  it('contains all expected layer types', () => {
    expectedLayerTypes.forEach(type => {
      expect(layerDefinitions).toHaveProperty(type);
    });
  });

  describe('each layer definition', () => {
    Object.entries(layerDefinitions).forEach(([type, definition]) => {
      describe(`${type} layer`, () => {
        it('has required properties', () => {
          expect(definition).toHaveProperty('type', type);
          expect(definition).toHaveProperty('displayName');
          expect(definition).toHaveProperty('icon');
          expect(definition).toHaveProperty('color');
          expect(definition).toHaveProperty('defaultParams');
          expect(definition).toHaveProperty('description');
        });

        it('has valid color format', () => {
          expect(definition.color).toMatch(/^#[0-9a-fA-F]{6}$/);
        });

        it('has non-empty strings', () => {
          expect(definition.displayName).toBeTruthy();
          expect(definition.icon).toBeTruthy();
          expect(definition.description).toBeTruthy();
        });

        it('has defaultParams as object', () => {
          expect(typeof definition.defaultParams).toBe('object');
          expect(definition.defaultParams).not.toBeNull();
        });
      });
    });
  });

  describe('input layer', () => {
    const inputLayer = layerDefinitions.input;

    it('has correct default shape', () => {
      expect(inputLayer.defaultParams.shape).toEqual([28, 28]);
    });

    it('has appropriate description', () => {
      expect(inputLayer.description).toContain('entry point');
      expect(inputLayer.description).toContain('shape');
    });
  });

  describe('dense layer', () => {
    const denseLayer = layerDefinitions.dense;

    it('has valid default parameters', () => {
      expect(denseLayer.defaultParams.units).toBe(128);
      expect(denseLayer.defaultParams.activation).toBe('relu');
      expect(denseLayer.defaultParams.useBias).toBe(true);
      expect(denseLayer.defaultParams.kernelInitializer).toBe('glorotUniform');
    });

    it('has appropriate description', () => {
      expect(denseLayer.description).toContain('fully connected');
    });
  });

  describe('conv2d layer', () => {
    const convLayer = layerDefinitions.conv2d;

    it('has valid default parameters', () => {
      expect(convLayer.defaultParams.filters).toBe(32);
      expect(convLayer.defaultParams.kernelSize).toBe(3);
      expect(convLayer.defaultParams.strides).toBe(1);
      expect(convLayer.defaultParams.padding).toBe('same');
      expect(convLayer.defaultParams.activation).toBe('relu');
      expect(convLayer.defaultParams.useBias).toBe(true);
    });
  });

  describe('maxpooling2d layer', () => {
    const poolingLayer = layerDefinitions.maxpooling2d;

    it('has valid default parameters', () => {
      expect(poolingLayer.defaultParams.poolSize).toBe(2);
      expect(poolingLayer.defaultParams.strides).toBe(2);
      expect(poolingLayer.defaultParams.padding).toBe('valid');
    });
  });

  describe('dropout layer', () => {
    const dropoutLayer = layerDefinitions.dropout;

    it('has valid default rate', () => {
      expect(dropoutLayer.defaultParams.rate).toBeGreaterThanOrEqual(0);
      expect(dropoutLayer.defaultParams.rate).toBeLessThanOrEqual(1);
      expect(dropoutLayer.defaultParams.rate).toBe(0.2);
    });

    it('has appropriate description', () => {
      expect(dropoutLayer.description).toContain('prevent overfitting');
    });
  });

  describe('flatten layer', () => {
    const flattenLayer = layerDefinitions.flatten;

    it('has empty default parameters', () => {
      expect(Object.keys(flattenLayer.defaultParams)).toHaveLength(0);
    });

    it('has appropriate description', () => {
      expect(flattenLayer.description).toContain('1D array');
    });
  });
});
/// <reference types="@testing-library/jest-dom" />
import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/svelte';
import { afterEach, vi } from 'vitest';

// Cleanup after each test
afterEach(() => {
	cleanup();
});

// Mock TensorFlow.js to avoid WebGL errors in tests
vi.mock('@tensorflow/tfjs', () => ({
	tensor2d: vi.fn((data, shape) => ({ 
		dispose: vi.fn(),
		shape: shape || [Array.isArray(data) ? data.length : 0, Array.isArray(data[0]) ? data[0].length : 0]
	})),
	tensor4d: vi.fn((data, shape) => ({ 
		dispose: vi.fn(),
		shape: shape || [0, 0, 0, 0]
	})),
	sequential: vi.fn(() => {
		const mockModel = {
			layers: [],
			add: vi.fn((layer) => {
				mockModel.layers.push(layer);
			}),
			compile: vi.fn(),
			fit: vi.fn(() => Promise.resolve({ history: {} })),
			predict: vi.fn(() => ({ dataSync: () => new Float32Array([0.1, 0.9]) })),
			dispose: vi.fn(),
			summary: vi.fn(),
			stopTraining: false
		};
		return mockModel;
	}),
	layers: {
		Layer: class MockLayer {
			constructor(config?: any) {}
			build(inputShape: any): void {}
			call(inputs: any, kwargs?: any): any {
				return inputs;
			}
			computeOutputShape(inputShape: any): any {
				return inputShape;
			}
			getConfig(): any {
				return {};
			}
			apply(inputs: any, kwargs?: any): any {
				return this.call(inputs, kwargs);
			}
		},
		dense: vi.fn((config) => ({ 
			...config, 
			type: 'dense',
			build: vi.fn(),
			call: vi.fn((inputs) => inputs),
			apply: vi.fn((inputs) => inputs),
			computeOutputShape: vi.fn((shape) => shape),
			getConfig: vi.fn(() => config)
		})),
		conv2d: vi.fn((config) => ({ ...config, type: 'conv2d' })),
		maxPooling2d: vi.fn((config) => ({ ...config, type: 'maxPooling2d' })),
		dropout: vi.fn((config) => ({ 
			...config, 
			type: 'dropout',
			build: vi.fn(),
			call: vi.fn((inputs) => inputs),
			apply: vi.fn((inputs) => inputs),
			computeOutputShape: vi.fn((shape) => shape),
			getConfig: vi.fn(() => config)
		})),
		flatten: vi.fn((config) => ({ ...config, type: 'flatten' })),
		embedding: vi.fn((config) => ({ ...config, type: 'embedding' })),
		layerNormalization: vi.fn((config) => ({ 
			...config, 
			type: 'layerNormalization',
			build: vi.fn(),
			call: vi.fn((inputs) => inputs),
			apply: vi.fn((inputs) => inputs),
			computeOutputShape: vi.fn((shape) => shape),
			getConfig: vi.fn(() => config)
		}))
	},
	train: {
		adam: vi.fn(() => 'adam'),
		sgd: vi.fn(() => 'sgd'),
		rmsprop: vi.fn(() => 'rmsprop')
	},
	tidy: vi.fn((fn) => fn()),
	keep: vi.fn((tensor) => tensor),
	// Tensor creation functions
	range: vi.fn((start, limit, delta, dtype) => ({
		dispose: vi.fn(),
		shape: [Math.ceil((limit - start) / (delta || 1))]
	})),
	expandDims: vi.fn((tensor, axis) => ({ 
		...tensor, 
		shape: tensor.shape ? [...tensor.shape.slice(0, axis), 1, ...tensor.shape.slice(axis)] : [1]
	})),
	stack: vi.fn((tensors, axis) => ({ 
		dispose: vi.fn(),
		shape: tensors[0]?.shape || [1]
	})),
	reshape: vi.fn((tensor, shape) => ({ ...tensor, shape })),
	// Math functions
	exp: vi.fn((x) => x),
	mul: vi.fn((a, b) => a),
	div: vi.fn((a, b) => a),
	sin: vi.fn((x) => x),
	cos: vi.fn((x) => x),
	sqrt: vi.fn((x) => x),
	scalar: vi.fn((value) => ({ dispose: vi.fn(), shape: [] })),
	// Tensor operations
	randomNormal: vi.fn((shape) => ({ 
		dispose: vi.fn(), 
		shape,
		dataSync: () => new Float32Array(shape.reduce((a, b) => a * b, 1))
	})),
	zeros: vi.fn((shape) => ({ 
		dispose: vi.fn(), 
		shape,
		dataSync: () => {
			const size = shape.reduce((a: any, b: any) => a * b, 1);
			const data = new Float32Array(size);
			// Fill with small non-zero values for tests that check for changes
			for (let i = 0; i < size; i++) {
				data[i] = 0.001 * (i % 10);
			}
			return data;
		}
	})),
	tensor3d: vi.fn((data, shape) => ({ 
		dispose: vi.fn(),
		shape: shape || [1, 1, 1],
		dataSync: () => {
			if (Array.isArray(data)) {
				return new Float32Array(data);
			} else {
				const size = shape ? shape.reduce((a: any, b: any) => a * b, 1) : 1;
				const arr = new Float32Array(size);
				for (let i = 0; i < size; i++) {
					arr[i] = typeof data === 'number' ? data : 0.1; // use constant value for test
				}
				return arr;
			}
		}
	})),
	slice: vi.fn((tensor, begin, size) => ({ 
		...tensor,
		shape: size || tensor.shape
	})),
	add: vi.fn((a, b) => a),
	transpose: vi.fn((tensor, perm) => ({ 
		...tensor,
		shape: perm ? perm.map((i: any) => tensor.shape[i]) : tensor.shape
	})),
	matMul: vi.fn((a, b) => a),
	softmax: vi.fn((tensor) => tensor),
	dropout: vi.fn((tensor, rate) => tensor),
	serialization: {
		registerClass: vi.fn()
	},
	dispose: vi.fn()
}));
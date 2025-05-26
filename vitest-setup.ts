import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/svelte';
import { afterEach, vi } from 'vitest';

// Cleanup after each test
afterEach(() => {
	cleanup();
});

// Mock TensorFlow.js to avoid WebGL errors in tests
vi.mock('@tensorflow/tfjs', () => ({
	tensor2d: vi.fn(() => ({ 
		dispose: vi.fn(),
		shape: [0, 0]
	})),
	tensor4d: vi.fn(() => ({ 
		dispose: vi.fn(),
		shape: [0, 0, 0, 0]
	})),
	sequential: vi.fn(() => ({
		add: vi.fn(),
		compile: vi.fn(),
		fit: vi.fn(() => Promise.resolve({ history: {} })),
		predict: vi.fn(() => ({ dataSync: () => new Float32Array([0.1, 0.9]) })),
		dispose: vi.fn(),
		layers: [],
		summary: vi.fn(),
		stopTraining: false
	})),
	layers: {
		dense: vi.fn((config) => ({ ...config, type: 'dense' })),
		conv2d: vi.fn((config) => ({ ...config, type: 'conv2d' })),
		maxPooling2d: vi.fn((config) => ({ ...config, type: 'maxPooling2d' })),
		dropout: vi.fn((config) => ({ ...config, type: 'dropout' })),
		flatten: vi.fn((config) => ({ ...config, type: 'flatten' }))
	},
	train: {
		adam: vi.fn(() => 'adam'),
		sgd: vi.fn(() => 'sgd'),
		rmsprop: vi.fn(() => 'rmsprop')
	},
	dispose: vi.fn()
}));
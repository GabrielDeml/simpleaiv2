# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Development
- `npm run dev` - Start development server (default port 5173)
- `npm run dev -- --open` - Start dev server and open browser
- `npm run build` - Build for production
- `npm run preview` - Preview production build

### Type Checking
- `npm run check` - Run svelte-kit sync and type check once
- `npm run check:watch` - Run type checking in watch mode

## Architecture Overview

This is a SvelteKit application demonstrating machine learning in the browser using TensorFlow.js. The project implements an MNIST digit recognition CNN (Convolutional Neural Network) trainer.

### Key Components

1. **Frontend Framework**: SvelteKit with TypeScript
   - File-based routing in `src/routes/`
   - Components use `.svelte` extension with script/template/style sections
   - `$lib` alias points to `src/lib/`

2. **ML Architecture** (`src/lib/mnist/`):
   - `dataLoader.ts`: Handles MNIST dataset loading from Google's hosted sprites
   - `model.ts`: Defines CNN architecture (Conv2D → MaxPool → Conv2D → MaxPool → Dense)
   - Training happens directly in the browser using TensorFlow.js

3. **Main Component** (`src/lib/components/MnistTrainer.svelte`):
   - Manages model training lifecycle
   - Provides drawing canvas for digit prediction
   - Real-time training progress display

### Project Structure
- SvelteKit app structure with Vite as build tool
- TypeScript with strict mode enabled
- TensorFlow.js as the only runtime dependency
- Component-scoped styling in Svelte files

### Important Notes
- No linting configured yet (ESLint config was removed)
- No test framework set up
- Models train in-browser; no backend required
- Canvas drawing uses 280x280px scaled down to 28x28px for MNIST
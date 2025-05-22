import React from 'react';

/**
 * Control panel component for MNIST trainer actions
 * @param {Object} props - Component props
 * @param {Function} props.onTrain - Callback for train button click
 * @param {Function} props.onTest - Callback for test button click  
 * @param {Function} props.onDrawRandom - Callback for draw random image button
 * @param {boolean} props.isTraining - Whether model is currently training
 * @param {boolean} props.canTrain - Whether training is enabled
 * @param {boolean} props.canTest - Whether testing is enabled
 * @param {boolean} props.canDraw - Whether drawing is enabled
 * @param {Object} props.trainingProgress - Training progress info
 */
function ControlPanel({
  onTrain,
  onTest,
  onDrawRandom,
  isTraining,
  canTrain,
  canTest,
  canDraw,
  trainingProgress
}) {
  return (
    <div className="controls-section">
      <button 
        onClick={onTrain} 
        disabled={isTraining || !canTrain}
        className="control-button train-button"
        aria-label={isTraining ? 'Training in progress' : 'Start model training'}
      >
        {isTraining && <span className="loading-indicator" aria-hidden="true"></span>}
        {isTraining ? (
          `Training... (${trainingProgress.epoch}/${trainingProgress.totalEpochs})`
        ) : (
          'Train Model'
        )}
      </button>
      
      <button 
        onClick={onTest} 
        disabled={!canTest}
        className="control-button test-button"
        aria-label="Test trained model on test dataset"
      >
        Test Model
      </button>
      
      <button 
        onClick={onDrawRandom} 
        disabled={!canDraw}
        className="control-button draw-button"
        aria-label="Display random test image with prediction"
      >
        Draw Random Test Image
      </button>
    </div>
  );
}

export default ControlPanel;
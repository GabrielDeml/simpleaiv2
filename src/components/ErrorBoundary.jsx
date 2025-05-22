import React from 'react';

/**
 * Error boundary component to catch and display errors gracefully
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state to show error UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log error for debugging
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-card">
            <h2 className="error-title">Something went wrong</h2>
            <div className="error-message">
              <p>An unexpected error occurred while running the MNIST trainer.</p>
              {this.state.error && (
                <details className="error-details">
                  <summary>Error Details</summary>
                  <pre className="error-stack">
                    {this.state.error.toString()}
                    {this.state.errorInfo?.componentStack}
                  </pre>
                </details>
              )}
            </div>
            <div className="error-actions">
              <button 
                onClick={this.handleRetry}
                className="control-button retry-button"
              >
                Try Again
              </button>
              <button 
                onClick={() => window.location.reload()}
                className="control-button reload-button"
              >
                Reload Page
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
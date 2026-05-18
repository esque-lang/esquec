// Package autodiff implements automatic differentiation for neural network training.
//
// This package provides tape-based reverse-mode automatic differentiation
// (backpropagation) for computing gradients of loss functions with respect
// to model parameters.
//
// # Overview
//
// Automatic differentiation works by recording operations on a "tape" during
// the forward pass, then playing the tape backwards to compute gradients.
// This is more efficient and accurate than numerical differentiation.
//
// # Basic Usage
//
//	// Create a tape to record operations
//	tape := autodiff.NewTape()
//
//	// Record operations during forward pass
//	for _, inst := range fn.Body {
//	    tape.Record(inst.Op, inst.Args, inst.Result)
//	}
//
//	// Compute gradients with respect to loss
//	gradInsts, err := tape.Backward(lossValue)
//
//	// Get gradient for a specific value
//	grad, ok := tape.GetGradient(paramID)
//
// # Supported Operations
//
// Gradients are implemented for:
//
// Basic arithmetic:
//   - Add: ∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1
//   - Sub: ∂(x-y)/∂x = 1, ∂(x-y)/∂y = -1
//   - Mul: ∂(x*y)/∂x = y, ∂(x*y)/∂y = x
//   - Div: ∂(x/y)/∂x = 1/y, ∂(x/y)/∂y = -x/y²
//   - Neg: ∂(-x)/∂x = -1
//
// Transcendental functions:
//   - Exp: ∂exp(x)/∂x = exp(x)
//   - Log: ∂log(x)/∂x = 1/x
//   - Sqrt: ∂√x/∂x = 1/(2√x)
//   - Pow: ∂(x^y)/∂x = y*x^(y-1), ∂(x^y)/∂y = x^y*log(x)
//   - Sin: ∂sin(x)/∂x = cos(x)
//   - Cos: ∂cos(x)/∂x = -sin(x)
//
// Neural network activations:
//   - ReLU: ∂relu(x)/∂x = 1 if x > 0, else 0
//   - Sigmoid: ∂σ(x)/∂x = σ(x)(1-σ(x))
//   - Tanh: ∂tanh(x)/∂x = 1 - tanh²(x)
//
// Tensor operations:
//   - MatMul: ∂(A@B)/∂A = ∂out@Bᵀ, ∂(A@B)/∂B = Aᵀ@∂out
//   - ReduceSum: gradient is broadcast to input shape
//   - Element-wise: similar to scalar operations
//
// # Gradient Accumulation
//
// When a value is used multiple times, gradients are accumulated:
//
//	// If x is used in both y = x + 1 and z = x * 2
//	// ∂L/∂x = ∂L/∂y * ∂y/∂x + ∂L/∂z * ∂z/∂x
//	//       = ∂L/∂y * 1 + ∂L/∂z * 2
//
// # Differentiating Functions
//
// The Differentiate function creates a gradient-computing version:
//
//	gradFn, err := autodiff.Differentiate(originalFn)
//	// gradFn computes both forward pass and gradients
//
// # Chain Rule
//
// Backpropagation applies the chain rule automatically:
//
//	// Forward: y = f(g(x))
//	// Backward: ∂L/∂x = ∂L/∂y * ∂y/∂g * ∂g/∂x
//
// The tape records the computation graph and computes partial derivatives
// in reverse order, accumulating gradients through the chain rule.
//
// # Memory Efficiency
//
// The tape stores references to intermediate values needed for gradients.
// For large models, consider:
//   - Gradient checkpointing (recompute vs store)
//   - Mixed precision (fp16 for activations)
//   - In-place operations where safe
//
// # Example: Training Loop
//
//	for epoch := range epochs {
//	    for batch := range data {
//	        // Forward pass with recording
//	        tape := autodiff.NewTape()
//	        loss := forward(model, batch, tape)
//
//	        // Backward pass
//	        grads, _ := tape.Backward(loss)
//
//	        // Update parameters
//	        for param, grad := range grads {
//	            param -= learningRate * grad
//	        }
//	    }
//	}
//
// Added in v0.9.
package autodiff

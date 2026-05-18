package types

// Shape is a placeholder for tensor shapes.
//
// v0.0 has no tensors, but downstream phases will normalise shape
// expressions to polynomials over shape variables here. We keep the
// stub so the package shape is stable as we grow.
type Shape struct {
	Dims []int64 // empty == rank-0 (scalar)
}

// Scalar returns the rank-0 shape.
func Scalar() Shape { return Shape{} }

// Rank reports the number of dimensions.
func (s Shape) Rank() int { return len(s.Dims) }

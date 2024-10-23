# SVM-pytorch

My model supports the four different kernels that are found in the Wikipedia article:
1. Polynomial (homogeneous) kernel:: k(xi,xj)=(xi⋅xj)
d
. When d=1, it becomes the linear kernel.
2. Polynomial (inhomogeneous) kernel:: k(xi,xj)=(xi⋅xj+r)
d
.
3. Gaussian (Radial Basis Function - RBF) kernel:
k(xi,xj)=exp(−γ∥xi−xj∥
2
)k(xi,xj)=exp(−γ ∥ xi−xj ∥
2
), where γ>0.
4. Sigmoid (Hyperbolic tangent) kernel:
k(xi,xj)=tanh(κxi⋅xj+c) for some (but not all) κ>0 και c<0.

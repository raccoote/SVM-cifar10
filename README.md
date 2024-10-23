# SVM-pytorch

My implementation of the Support Vector Machine (SVM) focused on classifying two classes from the CIFAR-10 dataset: airplane (label 0) and dog (label 5). As part of my college project, I built the SVM from scratch, exploring both primal and dual approaches. 

My model supports the four different kernels that are found in the [Wikipedia article](https://en.wikipedia.org/wiki/Support_vector_machine):
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

![image](https://github.com/user-attachments/assets/f2911b6c-c9c4-48ae-b685-ea5dd992aa2e)


Below we see the results of the SVM (training times, training/test accuracies) after many trials and parameter modifications. Due to computational restraints, only tests on primal SVM were taken:(

![image](https://github.com/user-attachments/assets/47ca5e99-8a19-4498-bb02-748c49526abd)

![image](https://github.com/user-attachments/assets/18402583-77db-4e26-955f-bac6e9cb40b8)

![image](https://github.com/user-attachments/assets/9c39c469-5122-4836-8b46-03c9551c983b)

![image](https://github.com/user-attachments/assets/1731c940-49f6-4b06-8171-541d8e94ee49)



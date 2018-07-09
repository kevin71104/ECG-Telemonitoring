# Eigenspace-Aided Compressed Analysis

We propose eigenspace-aided compressed analysis to integrate **Principle Component Analysis ** with **Compressed Sensing** and **Task-Driven Dictionary Learning** with this toolbox in the hope to help peer researchers.

- **Author**: Kai-Chieh, Hsu, Bo-Hong Cho, Ching-Yao Chou and An-Yeu (Andy) Wu
- **Related publications:** Kai-Chieh, Hsu, Bo-Hong Cho, Ching-Yao Chou and An-Yeu (Andy) Wu, "Low-Complexity Compressed Analysis in Eigenspace with Limited Labeled Data for Real-Time Electrocardiography Telemonitoring," *submitted to 2018 IEEE Global Conference on Signal and Information Processing.*

---

## Required Packages

- python 3.6.3
- numpy 1.13.3 
- scipy 1.0.0 
- tensorflow 1.8.0 
- scikit-learn 0.18.1 

## Demo

- Users need to customize the *loadData* function in *utils.py* with data format stored as
  - `XArr` : Collection of training samples, whose dimension of (N, n) 
  - `YArr` : Collection of testing samples, whose dimension of (N, n_t) 
  - `trls` : Collection of training labels, whose dimension of (1, n) 
  - `ttls` : Collection of testing labels, whose dimension of (1, n_t) 
  - `N` : The dimension of data features
  - `n` : The number of training samples
  - `n_t` : The number of testing samples

- Run:  

  ```
  python3 CA-E.py -f datapath -c #classes -i -int -nr 1  -tn 1
  ```

  - Users are encouraged to see more script detail in line 16-22 of *CA-E.py*

## Contact Information

   ```
Kai-Chieh Hsu:
   + b03901026@ntu.edu.tw
   + kevin71104@gmail.com
   ```

## References

1. [**(FISTA)** ](https://epubs.siam.org/doi/abs/10.1137/080716542) A. Beck and M. Teboulle, “A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems,” in SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183-202, 2009.
2. [**(ODL)**](http://www.jmlr.org/papers/volume11/mairal10a/mairal10a.pdf)  J. Mairal, F. Bach, J. Ponce and G. Sapiro, “Online learning for matrix factorization and sparse coding”, in J. Mach. Learn. Res., vol. 11, pp. 19 - 60, Mar. 2010.
3. [**(TDDL)**](https://ieeexplore.ieee.org/document/5975166/)  J. Mairal, F. Bach and J. Ponce, “Task-Driven Dictionary Learning,” in IEEE Trans. Pattern Anal. Mach. Intell., vol. 34, no. 4, pp. 791 - 804, Apr. 2012.


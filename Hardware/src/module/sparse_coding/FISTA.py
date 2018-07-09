import numpy as np
from ..utils import *

#################################
#              FISTA            #
#################################
def lasso_fista(Y, D, Xinit, alambda, opts, show_progress = False):
    """
    * Syntax: `[X, iter] = lasso_fista(Y, D, Xinit, lambda, opts)`
    * Solving a Lasso problem using FISTA [[11]](#fn_fista):
        `X, = arg min_X 0.5*||Y - DX||_F^2 + lambda||X||_1`.
        Note that `lambda` can be either a positive scalar or a matrix with
        positive elements.
      - INPUT:
        + `Y, D, lambda`: as in the problem.
        + `Xinit`: Initial guess
        + `opts`: options. See also [`fista`](#fista)
      - OUTPUT:
        + `X`: solution.
        + `iter`: number of fistat iterations.
        + `cost_info` : cost function of each iteration
    """
    if Xinit.size == 0:
        Xinit = np.zeros((D.shape[1], Y.shape[1]))

    def calc_f(X):
        return 0.5*normF2(Y - np.dot(D, X))

    def calc_F(X):
        if isinstance(alambda, np.ndarray):
            return calc_f(X) + alambda*abs(X) # element-wise multiplication
        else:
            return calc_f(X) + alambda*norm1(X)

    DtD = np.dot(D.T, D)
    DtY = np.dot(D.T, Y)
    def grad(X):
        g =  np.dot(DtD, X) - DtY
        return g
    L = np.max(LA.eig(DtD)[0])
    (X, it, cost_info) = fista(grad, Xinit, L, alambda, opts, calc_F, show_progress)
    return (X, it, cost_info)

def shrinkage(U, alambda):
    """
    * Soft thresholding function.
    * Syntax: ` X = shrinkage(U, lambda)`
    * Solve the following optimization problem:
    `X = arg min_X 0.5*||X - U||_F^2 + lambda||X||_1`
    where `U` and `X` are matrices with same sizes. `lambda` can be either
    positive a scalar or a positive matrix (all elements are positive) with
    same size as `X`. In the latter case, it is a weighted problem.
    """
    return np.maximum(0, U - alambda) + np.minimum(0, U + alambda)

def fista(fn_grad, Xinit, L, alambda, opts, fn_calc_F, show_progress = False):
    """
    * A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse
            Problems.
    * Solve the problem: `X = arg min_X F(X) = f(X) + lambda||X||_1` where:
       - `X`: variable, can be a matrix.
       - `f(X)` is a smooth convex function with continuously differentiable
       with Lipschitz continuous gradient `L(f)`
    * This is modified from https://github.com/tiepvupsu/DICTOL_python.git
       - Termination condition
    * Syntax: `[X, iter] = fista(grad, Xinit, L, lambda, opts, calc_F)` where:
       - INPUT:
            + `grad`: a _function_ calculating gradient of `f(X)` given `X`.
            + `Xinit`: initial guess.
            + `L`: the Lipschitz constant of the gradient of `f(X)`.
            + `lambda`: a regularization parameter, can be either positive a
                    scalar or a weighted matrix.
            + `opts`: a _structure_ variable describing the algorithm.
              * `opts.max_iter`: maximum iterations of the algorithm.
                    Default `300`.
              * `opts.tol`: a tolerance, the algorithm will stop if difference
                    between two successive `X` is smaller than this value.
                    Default `1e-8`.
              * `opts.show_progress`: showing `F(X)` after each iteration or
                    not. Default `false`.
            + `calc_F`: optional, a _function_ calculating value of `F` at `X`
                    via `feval(calc_F, X)`.
      - OUTPUT:
        + `X`: solution.
        + `iter`: number of iterations.
        + `cost_info` : cost function of each iteration
    """
    Linv = 1/L
    lambdaLiv = alambda/L
    x_old = Xinit.copy()
    y_old = Xinit.copy()
    t_old = 1
    it = 0
    cost_old = float("inf") # the positive infinity number
    cost_info = np.zeros((opts.max_iter))
    while it < opts.max_iter:
        it += 1
        x_new = np.real(shrinkage(y_old - Linv*fn_grad(y_old), lambdaLiv))
        t_new = 0.5*(1 + math.sqrt(1 + 4*t_old**2))
        y_new = x_new + (t_old - 1)/t_new * (x_new - x_old)
        cost_new = fn_calc_F(x_new)
        e = cost_old - cost_new
        cost_old = cost_new
        cost_info[it-1] = cost_new
        #e = norm1(x_new - x_old)/x_new.size
        if e < opts.tol:
            cost_info = cost_info[:it-1]
            if opts.verbose:
                print(cost_info.shape)
            break
        x_old = x_new.copy()
        t_old = t_new
        y_old = y_new.copy()
        if opts.verbose:
            print ('iter = '+str(it)+', cost = %.6f' % cost_new)
        if not opts.verbose and show_progress and it%10 == 0:
            str0 = progress_str(it, opts.max_iter, 50)
            sys.stdout.write("\r%s %.2f%%" % (str0, (it*100.0)/opts.max_iter ))
            sys.stdout.flush()
    if not opts.verbose and show_progress and it%10 == 0:
        print ('')
    return (x_new, it, cost_info)   
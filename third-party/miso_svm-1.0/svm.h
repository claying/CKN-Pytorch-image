#ifndef SVM_H
#define SVM_H

#include "linalg.h"

template <typename T>
T normsq(const Vector<T>& x, const Vector<T>& y) {
    return x.nrm2sq()+y.nrm2sq()-2*y.dot(x);
}

template <typename T>
void miso_svm_multiclass_accelerated_aux(const Vector<T>& y, const Matrix<T>& X, Matrix<T>& W, Matrix<T>& alpha, Vector<T>& C, const T lambda, const T kappa, const int max_iter, const int loss = 1) {
    /// assumes the right relation holds for W
    const int n = X.n();
    const int nclasses = W.n();
    Vector<T> xi;
    Vector<T> alphai;
    Vector<T> diff_alpha;
    Vector<T> beta;
    for (int ii=0; ii<max_iter; ++ii) {
    const int ind = random() % n;
    const T yi=y[ind];
    X.refCol(ind,xi);
    alpha.refCol(ind,alphai);
    diff_alpha.copy(alphai);
    W.multTrans(xi,beta);
    if (loss==1) {
    for (int jj=0; jj<nclasses; ++jj) {
        alphai[jj]=MAX(beta[jj]-beta[yi]+T(1.0),0);
    }
    alphai[yi]=0;
    C[ind] = T(0.5)*alphai.nrm2sq();
    alphai[yi]=-alphai.asum();
    } else {
        const T mmax=beta.maxval();
        alphai.copy(beta);
        alphai.add(-mmax);
        alphai.exp();
        C[ind]=-beta[yi]+mmax+log(alphai.sum());
        T sum=alphai.asum();
        alphai.scal(T(1.0)/sum);
        alphai[yi]-=T(1.0);
    }
    C[ind] -= alphai.dot(beta);
    diff_alpha.sub(alphai);
    W.rank1Update(xi,diff_alpha,T(1.0)/(n*(lambda+kappa)));
    }
}


template <typename T>
void miso_svm_multiclass_accelerated(const Vector<T>& y, const Matrix<T>& X, Matrix<T>& W, const T lambda, const T eps, const int epochs, const T L, const int loss = 1) {
    Timer time1;
    Timer time2;
    const int n = X.n();
    const int m = X.m();
    const int nclasses = W.n();
    Vector<T> C(n);
    Matrix<T> alpha(nclasses,n);
    Matrix<T> Z(m,nclasses);
    Matrix<T> diffZ(m,nclasses);
    Matrix<T> diffW(m,nclasses);
    Z.setZeros();
    diffZ.setZeros();
    diffW.setZeros();
    alpha.setZeros();
    C.setZeros();
    W.setZeros();
    const T kappa = L/n - lambda;
    std::cout << "kappa: " << kappa << std::endl;
    const T q = lambda/(lambda+kappa);
    const T qp = T(0.9)*sqrt(q);
    const T alphak = sqrt(q);
    const T betak=(T(1.0)-alphak)/(T(1.0)+alphak);

    time2.start();
    for (int ii=0; ii<epochs; ++ii) {
        diffW.copy(W);
        //if (true) {
        if (ii > 0 && ii % (10) == 0) {
            Matrix<T> tmp2;
            W.mult(X,tmp2,true,false);
            T los = 0;
            if (loss == 1) {
                for (int jj=0; jj<n; ++jj) {
                    for (int kk=0; kk<nclasses; ++kk) {
                        if (kk != y[jj]) {
                            const T rjk = tmp2[jj*nclasses+kk] - tmp2[jj*nclasses+y[jj]] + 1;
                            if (rjk > 0)
                            los += rjk*rjk;
                        }
                    }
                }
                los /= 2;
            } else if (loss==2) {
                Vector<T> beta;
                for (int jj=0; jj<n; ++jj) {
                    tmp2.refCol(jj,beta);
                    los += -beta[y[jj]];
                    const T mmax=beta.maxval();
                    beta.add(-mmax);
                    beta.exp();
                    los += mmax+log(beta.sum());
                }
            }
            const T reg = T(0.5)*lambda*W.normFsq();
            const T primal = los / n + reg;
            tmp2.copy(W);
            tmp2.add(diffZ,-kappa/(kappa+lambda));
            tmp2.add(Z,-kappa/(kappa+lambda));
            tmp2.scal((kappa+lambda)/lambda);
            const T dual = C.sum()/n - T(0.5)*lambda*tmp2.normFsq();
            const T gap = primal-dual;
            std::cout << "Iteration " << ii << ", objective: " << primal << ", dual: " << dual << ", gap: " << gap << std::endl;
            if (gap < eps) break;
        }

        if (ii % 5 == 0) {
            W.copy(Z);
            X.mult(alpha,W,false,true,-T(1.0)/((kappa+lambda)*n),kappa/(kappa+lambda)); // for numerical stability reasons
        } else {
            W.add(diffZ,-kappa/(kappa+lambda));
        }
        time1.start();
        miso_svm_multiclass_accelerated_aux(y,X,W,alpha,C,lambda,kappa,n,loss); // one epoch only
        time1.stop();
        diffW.sub(W);
        diffZ.copy(Z);
        Z.copy(W);
        Z.add(diffW,-betak);
        diffZ.sub(Z);
    }
    time2.stop();
    time1.printElapsed();
    time2.printElapsed();
}


template <typename T>
void miso_svm_multiclass_aux(const Vector<T>& y, const Matrix<T>& X, Matrix<T>& W, const T lambda, const T eps, const int epochs, const int loss = 1) {
    const int n = X.n();
    const int nclasses = W.n();
    Vector<T> C(n);
    Matrix<T> alpha(nclasses,n);
    alpha.setZeros();
    C.setZeros();
    W.setZeros();

    Vector<T> xi;
    Vector<T> alphai;
    Vector<T> diff_alpha;
    Vector<T> beta;
    const int max_iter=n*epochs;
    for (int ii=0; ii<max_iter; ++ii) {
        //if (ii % (n) == 0) {
        if (ii > 0 && ii % (10*n) == 0) {
            X.mult(alpha,W,false,true,-T(1.0)/(lambda*n)); // to improve numerical stability
            Matrix<T> tmp2;
            W.mult(X,tmp2,true,false);
            T los = 0;
            if (loss == 1) {
                for (int jj=0; jj<n; ++jj) {
                    for (int kk=0; kk<nclasses; ++kk) {
                        if (kk != y[jj]) {
                            const T rjk = tmp2[jj*nclasses+kk] - tmp2[jj*nclasses+y[jj]] + 1;
                            if (rjk > 0)
                                los += rjk*rjk;
                        }
                    }
                }
                los /= 2;
            } else if (loss==2) {
                Vector<T> beta;
                for (int jj=0; jj<n; ++jj) {
                    tmp2.refCol(jj,beta);
                    los += -beta[y[jj]];
                    const T mmax=beta.maxval();
                    beta.add(-mmax);
                    beta.exp();
                    los += mmax+log(beta.sum());
                }
            }
            const T reg = T(0.5)*lambda*W.normFsq();
            const T primal = los / n + reg;
            const T dual = C.sum()/n - reg;
            const T gap = primal-dual;
            std::cout << "Iteration " << ii << ", objective: " << primal << ", dual: " << dual << ", gap: " << gap << std::endl;
            if (gap < eps) break;
        }
        const int ind = random() % n;
        const T yi=y[ind];
        X.refCol(ind,xi);
        alpha.refCol(ind,alphai);
        diff_alpha.copy(alphai);
        W.multTrans(xi,beta);
        if (loss==1) {
            for (int jj=0; jj<nclasses; ++jj) {
                alphai[jj]=MAX(beta[jj]-beta[yi]+T(1.0),0);
            }
            alphai[yi]=0;
            C[ind] = T(0.5)*alphai.nrm2sq();
            alphai[yi]=-alphai.asum();
        } else {
            const T mmax=beta.maxval();
            alphai.copy(beta);
            alphai.add(-mmax);
            alphai.exp();
            C[ind]=-beta[yi]+mmax+log(alphai.sum());
            T sum=alphai.asum();
            alphai.scal(T(1.0)/sum);
            alphai[yi]-=T(1.0);
        }
        C[ind] -= alphai.dot(beta);
        diff_alpha.sub(alphai);
        W.rank1Update(xi,diff_alpha,T(1.0)/(n*lambda));
    }
}

template <typename T>
void miso_svm_multiclass(const Vector<T>& y, const Matrix<T>& X, Matrix<T>& W, const T lambda, const T eps, const int max_iter, const bool accelerated = false, const int loss = 1) {
    const int n = y.n();
    const int p = X.m();
    const int nclasses=y.maxval()+1;
    W.resize(p,nclasses);
    Vector<T> normX;
    X.norm_2sq_cols(normX);
    const T R = normX.mean();
    //const T L = 4*normX.mean();
    const T L = 2*(1+sqrt(nclasses))*normX.mean();
    std::cout << "Value of R: " << R << std::endl;
    std::cout << "Value of L/mu: " << L/lambda << std::endl;
    std::cout << "Problem size: p x n: " << p << " " << n << std::endl;
    std::cout << "*********************" << std::endl;
    std::cout << "Processes Lambda " << lambda << std::endl;
    std::cout << "Eps " << eps << std::endl;
    std::cout << "Loss " << loss << std::endl;
    if (accelerated && n < L/lambda) {
        std::cout << "Accelerated algorithm" << std::endl;
        miso_svm_multiclass_accelerated(y,X,W,lambda,eps,max_iter,L,loss);
    } else {
        miso_svm_multiclass_aux(y,X,W,lambda,eps,max_iter,loss);
    }
}

template <typename T>
void mult(const Matrix<T>& X, const Vector<int>& ind, const Vector<T>& alpha, Vector<T>& w, const T a, const T b = 0.0) {
    w.resize(X.m());
    w.scal(b);
    Vector<T> col;
    for (int ii=0; ii<ind.n(); ++ii) {
        X.refCol(ind[ii],col);
        w.add(col,a*alpha[ii]);
    }
};

template <typename T>
void multTrans(const Matrix<T>& X, const Vector<int>& ind, const Vector<T>& w, Vector<T>& tmp) {
    tmp.resize(ind.n());
    tmp.setZeros();
    Vector<T> col;
    for (int ii=0; ii<ind.n(); ++ii) {
        X.refCol(ind[ii],col);
        tmp[ii]=col.dot(w);
    }
};

template <typename T>
void miso_svm_aux(const Vector<T>& y, const Matrix<T>& X, const Vector<int>& indices, Vector<T>& w, const T R, const T lambda, const T eps, const int max_iter, int& num_it,T& primal,T& loss, const int verbose=0) {
    const int n = y.n();
    w.setZeros();
    const T L = R+lambda;
    const T deltaT = n*MIN(T(1.0)/n,lambda/(2*L));
    Vector<T> xi;
    Vector<T> alpha(n);
    alpha.setZeros();
    Vector<T> C(n);
    C.setZeros();
    Vector<T> tmp;
    T dualold=0;
    T dual=0;
    num_it=0;
    for (int ii = 0; ii<max_iter; ++ii) {
        if (ii > 0 && (ii % (10*n)) == 0) {
            num_it+=10;
            if (indices.n() > 0) {
                mult(X,indices,alpha,w,T(1.0)/n);
                multTrans(X,indices,w,tmp);
            } else {
                X.mult(alpha,w,T(1.0)/n); // to improve numerical stability
                X.multTrans(w,tmp);
            }
            primal=0;
            for (int kk=0; kk<n; ++kk) {
                const T los=MAX(0,1-y[kk]*tmp[kk]);
                primal += los*los;
            }
            primal *= T(0.5)/n;
            loss=primal;
            T reg=0.5*lambda*w.nrm2sq();
            primal += reg;
            dual=C.mean() - reg;
            if (dual <= dualold || (primal - dual) < eps) {
                if (verbose) {
#pragma omp critical
                    {
                        std::cout << "Solver has finished after " << ii << " iterations, primal: " << primal << ", dual: " << dual << ", gap: " << (primal-dual) << std::endl;
                    }
                }
                break;
            }
            dualold=dual;
        }
        const int ind = random() % n;
        const T yi=y[ind];
        if (indices.n() > 0) {
            X.refCol(indices[ind],xi);
        } else {
            X.refCol(ind,xi);
        }
        const T beta = yi*xi.dot(w);
        const T gamma=MAX(T(1.0)-beta,0);
        T newalpha;
        C[ind]=(T(1.0)-deltaT)*C[ind]+deltaT*(T(0.5)*gamma*gamma+beta*gamma);
        newalpha=(T(1.0)-deltaT)*(alpha[ind])+deltaT*yi*gamma/lambda;
        w.add(xi,(newalpha-alpha[ind])/n);
        alpha[ind]=newalpha;
    }
};

template <typename T>
void miso_svm_onevsrest(const Vector<T>& yAll, const Matrix<T>& X,
                        Matrix<T>& W, Vector<int>& info, Vector<T>& primals, Vector<T>& losses,
                        const T lambda, const T eps, const int max_iter,
                        const bool accelerated = false, const int reweighted = 0, const bool non_uniform=true, const int verbose=0) {
    const int n = yAll.n();
    const int p = X.m();
    const int nclasses=yAll.maxval()+1;

    info.resize(nclasses);
    primals.resize(nclasses);
    losses.resize(nclasses);
    W.resize(p,nclasses);

    Vector<T> normX;
    X.norm_2sq_cols(normX);
    const T R = normX.maxval();
    if (verbose) {
        if (reweighted)
            std::cout << "Reweighted algorithm" << std::endl;
        if (non_uniform)
            std::cout << "Non-uniform sampling" << std::endl;
        std::cout << "Value of R: " << R << std::endl;

        std::cout << "Problem size: p x n: " << p << " " << n << std::endl;
        std::cout << "*********************" << std::endl;
        std::cout << "Processes Lambda " << lambda << std::endl;
        std::cout << "Eps " << eps << std::endl;
    }
    int jj;
#pragma omp parallel for private(jj)
    for (jj = 0; jj<nclasses; ++jj) {
        Vector<T> w;
        W.refCol(jj,w);
        int num_it;
        T primal;
        T loss;
        if (non_uniform) {
            Vector<T> y(n);
            Vector<int> ind;
            for (int ii = 0; ii<n; ++ii)
                y[ii]= abs<T>((yAll[ii] - T(jj))) < T(0.1) ? T(1.0) : -T(1.0);
            if (accelerated && T(2.0)*normX.mean()/n > lambda) {
                nonu_accelerated_miso_svm_aux(y,X,w,normX,lambda,eps,max_iter,num_it,primal,loss, verbose);
            } else {
                nonu_miso_svm_aux(y,X,w,normX,lambda,eps,max_iter,num_it,primal,loss, verbose);
            }
        } else {
            if (reweighted) {
                int npos=0;
                for (int ii = 0; ii<n; ++ii)
                    if (abs<T>((yAll[ii] - T(jj))) < T(0.1)) npos++;
                const int beta= reweighted==1 ? nclasses-2 : static_cast<int>(floor(sqrt(nclasses-2)));
                int nn = n + npos*(beta);
                Vector<int> ind(nn);
                Vector<T> y(nn);
                int counter=0;
                for (int ii = 0; ii<n; ++ii) {
                    if (abs<T>((yAll[ii] - T(jj))) < T(0.1)) {
                        for (int kk=0; kk<beta+1; ++kk) {
                            ind[counter]=ii;
                            y[counter++] = T(1.0);
                        }
                    } else {
                        ind[counter]=ii;
                        y[counter++] = -T(1.0);
                    }
                }
                if (accelerated && T(2.0)*R/nn > lambda) {
                    accelerated_miso_svm_aux(y,X,ind,w,R,lambda,eps,max_iter,num_it,primal,loss, verbose);
                } else {
                    miso_svm_aux(y,X,ind,w,R,lambda,eps,max_iter,num_it,primal,loss, verbose);
                }
            } else {
                Vector<T> y(n);
                Vector<int> ind;
                for (int ii = 0; ii<n; ++ii)
                    y[ii]= abs<T>((yAll[ii] - T(jj))) < T(0.1) ? T(1.0) : -T(1.0);
                if (accelerated && T(2.0)*R/n > lambda) {
                    accelerated_miso_svm_aux(y,X,ind,w,R,lambda,eps,max_iter,num_it,primal,loss, verbose);
                } else {
                    miso_svm_aux(y,X,ind,w,R,lambda,eps,max_iter,num_it,primal,loss, verbose);
                }
            }
        }
        info[jj]=num_it;
        primals[jj]=primal;
        losses[jj]=loss;
    }
    if (verbose) {
        std::cout << "primal: " << primals.sum()/nclasses << std::endl;
        std::cout << "loss: " << losses.sum()/nclasses << std::endl;
    }
}

// template <typename T>
// void miso_svm(const Vector<T>& y, const Matrix<T>& X, Matrix<T>& W, const Vector<T>& tablambda, const T eps, const int max_iter) {
//    const int n = y.n();
//    const int p = X.m();
//    const int nlambda=tablambda.n();
//    W.resize(p,nlambda);
//    W.setZeros();
//    Vector<T> normX;
//    X.norm_2sq_cols(normX);
//    const T R = normX.fmax();
//
//    std::cout << "Problem size: p x n: " << p << " " << n << std::endl;
//    for (int jj = 0; jj<nlambda; ++jj) {
//       const T lambda=tablambda[jj];
//       std::cout << "*********************" << std::endl;
//       std::cout << "Processes Lambda " << lambda << std::endl;
//       Vector<T> w;
//       W.refCol(jj,w);
//       miso_svm_aux(y,X,w,R,lambda,eps,max_iter,loss);
//    }
// }

template <typename T>
void accelerated_miso_svm_aux(const Vector<T>& y, const Matrix<T>& X, const Vector<int>& indices, Vector<T>& w, const T R, const T lambda, const T eps, const int max_iter, int& num_it, T& primal, T& loss, const int verbose) {
    const int n = y.n();
    const int p = X.m();
    w.setZeros();
    Vector<T> alpha(n);
    alpha.setZeros();
    Vector<T> C(n);
    C.setZeros();
    Vector<T> z(p);
    z.setZeros();
    Vector<T> zold(p);
    zold.setZeros();
    Vector<T> wold(p);
    wold.setZeros();
    Vector<T> xtw(n);
    xtw.setZeros();
    Vector<T> bestw(p);
    bestw.copy(w);
    T bestprimal=INFINITY;
    T bestloss=INFINITY;
    const T kappa = (T(2.0)*R/n-lambda);
    const T q = lambda/(lambda+kappa);
    const T qp = T(0.9)*sqrt(q);
    const T alphak = sqrt(q);
    const T betak=(T(1.0)-alphak)/(T(1.0)+alphak);
    T epsk=T(1.0);
    T gapk=T(1.0);
    T gap=T(1.0);
    int total_iters=0;
    int counter = 1;
    T gapold=T(1.0);
    for (int ii=0; ii<max_iter; ++ii) {
        epsk *= (T(1.0)-qp);
        // check if continue or not
        wold.copy(w);
        if ((total_iters / (10*n)) >= counter) {
            ++counter;
            w.copy(z);
            w.scal(kappa/(kappa+lambda));
            if (indices.n() > 0) {
                mult(X,indices,alpha,w,lambda/(n*(kappa+lambda)),T(1.0));
            } else {
                X.mult(alpha,w,lambda/(n*(kappa+lambda)),T(1.0));
            }
        } else {
            w.add(z,kappa/(kappa+lambda));
            w.add(zold,-kappa/(kappa+lambda));
        }
        const T diffNorm = normsq(z,zold);
        gapk=(n*(gapk + T(0.5)*(kappa*kappa/(lambda+kappa))*diffNorm));
        int num_iters;
        accelerated_miso_svm_aux2(y, X, indices, w, alpha, C, loss, gapk, num_iters, z, kappa, R, lambda, epsk);
        total_iters += num_iters;
        primal = loss+T(0.5)*lambda*w.nrm2sq();
        if (primal < bestprimal) {
            bestw.copy(w);
            bestprimal=primal;
            bestloss=loss;
        }
        Vector<T> ws;
        ws.copy(w);
        ws.scal((kappa+lambda)/lambda);
        ws.add(z,-kappa/lambda);
        const T dual=C.mean() - T(0.5)*lambda*ws.nrm2sq();
        gap=primal-dual;
        if (gap <= eps || total_iters >= max_iter) {
            if (verbose) {
#pragma omp critical
                {
                    std::cout << "Iteration " << total_iters << ", inner it: " << ii << ", loss: " << loss << ", primal: " << primal << ", dual: " << dual << ", gap: " << (primal-dual) << std::endl;
                }
            }
            break;
        }
        gapold=gap;
        zold.copy(z);
        z.copy(w);
        z.scal(T(1.0)+betak);
        z.add(wold,-betak);
    }
    w.copy(bestw);
    num_it=total_iters/n;
    primal=bestprimal;
    loss=bestloss;
};


// need to restart !
template <typename T>
void accelerated_miso_svm_aux2(const Vector<T>& y, const Matrix<T>& X, const Vector<int>& indices, Vector<T>& w, Vector<T>& alpha, Vector<T>& C, T& loss,T& gap, int& num_iters,  const Vector<T>& z, const T kappa, const T R, const T lambda, const T eps) {
    const int n = y.n();
    const long long max_iter = MAX(static_cast<long long>(floor(log(double(eps)/double(gap))/log(double(1.0)-double(1.0)/n))),n);
    Vector<T> tmp;
    Vector<T> xi;
    for (int ii = 0; ii<max_iter; ++ii) {
        if (ii > 0 && (ii % (n)) == 0) {
            loss=0;
            if (indices.n() > 0) {
                multTrans(X,indices,w,tmp);
            } else {
                X.multTrans(w,tmp);
            }
            for (int kk=0; kk<n; ++kk) {
                const T los=MAX(0,1-y[kk]*tmp[kk]);
                loss += los*los;
            }
            loss *= T(0.5)/n;
            const T reg=T(0.5)*(lambda+kappa)*w.nrm2sq();
            const T primal = loss+ reg  - kappa*w.dot(z);
            const T dual=C.mean() - reg;
            if ((primal - dual) < eps) {
                gap=primal-dual;
                num_iters=ii;
                break;
            }
        }
        const int ind = random() % n;
        const T yi=y[ind];
        if (indices.n() > 0) {
            X.refCol(indices[ind],xi);
        } else {
            X.refCol(ind,xi);
        }
        const T beta = yi*xi.dot(w);
        const T gamma=MAX(T(1.0)-beta,0);
        T newalpha;
        C[ind]=T(0.5)*gamma*gamma+beta*gamma;
        newalpha=yi*gamma/lambda;
        if (newalpha != alpha[ind])
            w.add(xi,lambda*(newalpha-alpha[ind])/(n*(lambda+kappa)));
        alpha[ind]=newalpha;
        if (ii==max_iter-1) {
            num_iters=max_iter;
            loss=0;
            if (indices.n() > 0) {
                multTrans(X,indices,w,tmp);
            } else {
                X.multTrans(w,tmp);
            }
            for (int kk=0; kk<n; ++kk) {
                const T los=MAX(0,1-y[kk]*tmp[kk]);
                loss += los*los;
            }
            loss *= T(0.5)/n;
            const T reg=T(0.5)*(lambda+kappa)*w.nrm2sq();
            const T primal = loss+ reg  - kappa*w.dot(z);
            const T dual=C.mean() - reg;
            gap=primal-dual;
        }
    }
}

template <typename T>
int nonu_sampling(const Vector<T>& sumpi) {
    const T val = static_cast<T>(random())/RAND_MAX;
    const int n = sumpi.n();
    if (sumpi[0] >= val) return 0;
    int m = 0;
    int M = n-1;
    while (M > m+1) {
        const int mid=(m+M)/2;
        if (sumpi[mid] >= val) {
            M=mid;
        } else {
            m=mid;
        }
    }
    return M;
};

//nonu_accelerated_miso_svm_aux(y,X,w,normX,lambda,eps,max_iter,num_it,primal,loss);

template <typename T>
void nonu_miso_svm_aux(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const Vector<T>& Li, const T lambda, const T eps, const int max_iter, int& num_it,T& primal,T& loss, const int verbose) {
    const int n = y.n();
    w.setZeros();
    Vector<T> deltaT(n);
    for (int ii=0; ii<n; ++ii) deltaT[ii]=MIN(n*lambda/(2*Li[ii]),T(1.0));
    Vector<T> sumpi(n);
    for (int ii=0; ii<n; ++ii) sumpi[ii]=T(1.0)/(deltaT[ii]*(T(1.0)- deltaT[ii]*Li[ii]/(lambda*n)));
    sumpi.scal(T(1.0)/sumpi.sum());
    for (int ii=1; ii<n; ++ii) sumpi[ii] += sumpi[ii-1];

    Vector<T> xi;
    Vector<T> alpha(n);
    alpha.setZeros();
    Vector<T> C(n);
    C.setZeros();
    Vector<T> tmp;
    T dualold=0;
    T dual=0;
    num_it=0;
    for (int ii = 0; ii<max_iter; ++ii) {
        if (ii > 0 && (ii % (10*n)) == 0) {
            num_it+=10;
            X.mult(alpha,w,T(1.0)/n); // to improve numerical stability
            X.multTrans(w,tmp);
            primal=0;
            for (int kk=0; kk<n; ++kk) {
                const T los=MAX(0,1-y[kk]*tmp[kk]);
                primal += los*los;
            }
            primal *= T(0.5)/n;
            loss=primal;
            T reg=0.5*lambda*w.nrm2sq();
            primal += reg;
            dual=C.mean() - reg;
            if (dual <= dualold || (primal - dual) < eps) {
                if (verbose) {
#pragma omp critical
                    {
                        std::cout << "Solver has finished after " << ii << " iterations, primal: " << primal << ", dual: " << dual << ", gap: " << (primal-dual) << std::endl;
                    }
                }
                break;
            }
            dualold=dual;
        }
        const int ind = nonu_sampling(sumpi);
        const T yi=y[ind];
        X.refCol(ind,xi);
        const T beta = yi*xi.dot(w);
        const T gamma=MAX(T(1.0)-beta,0);
        T newalpha;
        C[ind]=(T(1.0)-deltaT[ind])*C[ind]+deltaT[ind]*(T(0.5)*gamma*gamma+beta*gamma);
        newalpha=(T(1.0)-deltaT[ind])*(alpha[ind])+deltaT[ind]*yi*gamma/lambda;
        w.add(xi,(newalpha-alpha[ind])/n);
        alpha[ind]=newalpha;
    }
};

template <typename T>
void nonu_accelerated_miso_svm_aux(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const Vector<T>& Li, const T lambda, const T eps, const int max_iter, int& num_it, T& primal, T& loss, const int verbose) {
    const int n = y.n();
    const int p = X.m();

    w.setZeros();
    Vector<T> alpha(n);
    alpha.setZeros();
    Vector<T> C(n);
    C.setZeros();
    Vector<T> z(p);
    z.setZeros();
    Vector<T> zold(p);
    zold.setZeros();
    Vector<T> wold(p);
    wold.setZeros();
    Vector<T> bestw(p);
    bestw.copy(w);
    T bestprimal=INFINITY;
    T bestloss=INFINITY;
    const T kappa = (T(2.0)*Li.mean()/n-lambda);
    const T q = lambda/(lambda+kappa);
    const T qp = T(0.9)*sqrt(q);
    const T alphak = sqrt(q);
    const T betak=(T(1.0)-alphak)/(T(1.0)+alphak);
    Vector<T> deltaT(n);
    for (int ii=0; ii<n; ++ii) deltaT[ii]=MIN(n*(lambda+kappa)/(2*Li[ii]),T(1.0));
    Vector<T> sumpi(n);
    for (int ii=0; ii<n; ++ii) sumpi[ii]=T(1.0)/(deltaT[ii]*(T(1.0)- deltaT[ii]*Li[ii]/((lambda+kappa)*n)));
    sumpi.scal(T(1.0)/sumpi.sum());
    for (int ii=1; ii<n; ++ii) sumpi[ii] += sumpi[ii-1];
    T gap=T(1.0);
    int total_iters=0;
    int counter = 1;
        for (int ii=0; ii<max_iter; ++ii) {
        // check if continue or not
        wold.copy(w);
        if ((total_iters / (10*n)) >= counter) {
            ++counter;
            w.copy(z);
            w.scal(kappa/(kappa+lambda));
            X.mult(alpha,w,lambda/(n*(kappa+lambda)),T(1.0));
        } else {
            w.add(z,kappa/(kappa+lambda));
            w.add(zold,-kappa/(kappa+lambda));
        }
        int num_iters;
        nonu_accelerated_miso_svm_aux2(y, X, w, alpha, C, z, kappa, Li, lambda,n,deltaT,sumpi);
        total_iters += n;
        loss=0;
        Vector<T> tmp;
        X.multTrans(w,tmp);
        for (int kk=0; kk<n; ++kk) {
            const T los=MAX(0,1-y[kk]*tmp[kk]);
            loss += los*los;
        }
        loss *= T(0.5)/n;
        primal = loss+T(0.5)*lambda*w.nrm2sq();
        if (primal < bestprimal) {
            bestw.copy(w);
            bestprimal=primal;
            bestloss=loss;
        }
        Vector<T> ws;
        ws.copy(w);
        ws.scal((kappa+lambda)/lambda);
        ws.add(z,-kappa/lambda);
        const T dual=C.mean() - T(0.5)*lambda*ws.nrm2sq();
        gap=primal-dual;
        if (gap <= eps || total_iters >= max_iter) {
            if (verbose) {
#pragma omp critical
                {
                    std::cout << "Iteration " << total_iters << ", inner it: " << ii << ", loss: " << loss << ", primal: " << primal << ", dual: " << dual << ", gap: " << (primal-dual) << std::endl;
                }
            }
            break;
        }
        zold.copy(z);
        z.copy(w);
        z.scal(T(1.0)+betak);
        z.add(wold,-betak);
    }
    w.copy(bestw);
    num_it=total_iters/n;
    primal=bestprimal;
    loss=bestloss;
};


// need to restart !
template <typename T>
void nonu_accelerated_miso_svm_aux2(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, Vector<T>& alpha, Vector<T>& C, const Vector<T>& z, const T kappa, const Vector<T>& Li, const T lambda, const int max_iter, const Vector<T>& deltaT, const Vector<T>& sumpi) {
    const int n = y.n();
    Vector<T> tmp;
    Vector<T> xi;
    for (int ii = 0; ii<max_iter; ++ii) {
        const int ind = nonu_sampling(sumpi);
        const T yi=y[ind];
        X.refCol(ind,xi);
        const T beta = yi*xi.dot(w);
        const T gamma=MAX(T(1.0)-beta,0);
        T newalpha;
        C[ind]=(T(1.0)-deltaT[ind])*C[ind]+deltaT[ind]*(T(0.5)*gamma*gamma+beta*gamma);
        newalpha=(T(1.0)-deltaT[ind])*(alpha[ind])+deltaT[ind]*yi*gamma/lambda;
        if (newalpha != alpha[ind])
            w.add(xi,lambda*(newalpha-alpha[ind])/(n*(lambda+kappa)));
        alpha[ind]=newalpha;
    }
}


#endif
//
// vim: tabstop=3 softtabstop=3 shiftwidth=3 expandtab

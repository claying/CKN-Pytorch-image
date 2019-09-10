#ifndef COMMON_H
#define COMMON_H
#include <functional>
#include <cmath>
#include <numeric>
#include "linalg.h"
#define PI T(3.14159265359)

#ifdef TIMINGS
#define NTIMERS 15
Timer time_mat[NTIMERS];

#define RESET_TIMERS \
   for (int llll = 0; llll < NTIMERS; ++llll) { \
      time_mat[llll].stop(); \
      time_mat[llll].reset(); \
   }
#define START_TIMER(i) \
   time_mat[i].start();
#define STOP_TIMER(i) \
   time_mat[i].stop();

#define PRINT_TIMERS \
   for (int llll = 0; llll < NTIMERS; ++llll) { \
      std::cout << "Timer " << llll << std::endl; \
      time_mat[llll].printElapsed(); \
   }
#else
#define START_TIMER(i)
#define RESET_TIMERS
#define STOP_TIMER(i)
#define PRINT_TIMER(i)
#endif

//#ifdef MKL_DNN
//#include "common_dnn.h"
//#endif
//


template <typename T>
inline void centering(Matrix<T>& X, const INTM V = 1) {
    T* prX=X.rawX();
    const INTM n = X.n();
    const INTM m = X.m();
#pragma omp parallel for
    for (INTM ii=0;ii <n; ++ii) {
        Vector<T> mean(V);
        mean.setZeros();
        for (INTM jj=0;jj<m;jj+=V) /// assumes R,G,B,R,G,B,R,G,B
            for (INTM kk=0; kk<V; ++kk)
                mean[kk]+=prX[ii*m+jj+kk];
        for (INTM kk=0; kk<V; ++kk)
            mean[kk] *= V/static_cast<T>(m);
        for (INTM jj=0;jj<m;jj+=V)
            for (INTM kk=0; kk<V; ++kk)
                prX[ii*m+jj+kk] -= mean[kk];
    }
};

#define EPS_NORM 0.00001

template <typename T>
inline void normalize(Matrix<T>& X, Vector<T>& norms) {
    T* prX=X.rawX();
    const INTM n = X.n();
    const INTM m = X.m();
    norms.resize(n);
//#pragma omp parallel for
    for (INTM ii=0;ii <n; ++ii) {
        norms[ii]=cblas_nrm2<T>(m,prX+m*ii,1);
        cblas_scal<T>(m,T(1.0)/MAX(norms[ii],T(EPS_NORM)),prX+ii*m,1);
    }
};

template <typename T>
inline void normalize(Vector<T>& X) {
    T* prX=X.rawX();
    const INTM m = X.n();
    T nrm=cblas_nrm2<T>(m,prX,1);
    cblas_scal<T>(m,T(1.0)/MAX(nrm,T(EPS_NORM)),prX,1);
}

template <typename T>
inline void normalize(Matrix<T>& X) {
    T* prX=X.rawX();
    const INTM n = X.n();
    const INTM m = X.m();
//#pragma omp parallel for
    for (INTM ii=0;ii <n; ++ii) {
        T nrm=cblas_nrm2<T>(m,prX+m*ii,1);
        cblas_scal<T>(m,T(1.0)/MAX(nrm,T(EPS_NORM)),prX+ii*m,1);
    }
};

template <typename T>
void inline whitening(Matrix<T>& X) {
    Matrix<T> Wfilt;
    Vector<T> mu;
    X.meanCol(mu);
    Vector<T> ones(X.n());
    ones.set(T(1.0));
    X.rank1Update(mu,ones,-T(1.0));
    Matrix<T> U;
    Vector<T> S;
    X.svd2(U,S,X.m(),2);
    const T maxS=S.fmaxval();
    for (int ii=0; ii<S.n(); ++ii)
        S[ii] = (S[ii] > maxS*1e-8) ? T(1.0)/alt_sqrt<T>(S[ii]) : 0;
    Matrix<T> U2;
    U2.copy(U);
    U2.multDiagRight(S);
    U2.mult(U,Wfilt,false,true);
    Matrix<T> tmp, tmp2;
    for (INTM ii = 0; ii<X.n(); ii+=10000) {
        const INTM size_block=MIN(10000,X.n()-ii);
        X.refSubMat(ii,size_block,tmp);
        tmp2.copy(tmp);
        Wfilt.mult(tmp2,tmp);
    }
};




template <typename T>
void inline whitening(Matrix<T>& X, Matrix<T>& Wfilt, Vector<T>& mu) {
    X.meanCol(mu);
    Vector<T> ones(X.n());
    ones.set(T(1.0));
    X.rank1Update(mu,ones,-T(1.0));
    Matrix<T> U;
    Vector<T> S;
    X.svd2(U,S,X.m(),2);
    const T maxS=S.fmaxval();
    for (int ii=0; ii<S.n(); ++ii)
        S[ii] = (S[ii] > maxS*1e-8) ? T(1.0)/sqr<T>(S[ii]) : 0;
    Matrix<T> U2;
    U2.copy(U);
    U2.multDiagRight(S);
    U2.mult(U,Wfilt,false,true);
    Matrix<T> tmp, tmp2;
    for (INTM ii = 0; ii<X.n(); ii+=10000) {
        const INTM size_block=MIN(10000,X.n()-ii);
        X.refSubMat(ii,size_block,tmp);
        tmp2.copy(tmp);
        Wfilt.mult(tmp2,tmp);
    }
};

template <typename T> struct Map {
    public:
        Map() : _x(0), _y(0), _z(0) { };
        Map(T* X, const INTM x, const INTM y, const INTM z) {
            this->setData(X,x,y,z);
        };
        virtual ~Map() {  };
        void copy(const Map<T>& map) {
            _x=map._x;
            _y=map._y;
            _z=map._z;
            _vec.copy(map._vec);
        }
        void resize(const INTM x,const INTM y,const INTM z) {
            _x=x;
            _y=y;
            _z=z;
            _vec.resize(x*y*z);
        }
        void refSubMapZ(const INTM ind, Map<T>& map) const {
            map._x=_x;
            map._y=_y;
            map._z=1;
            map._vec.setData(_vec.rawX()+_x*_y*ind,_x*_y);
        }
        void setData(T* X, const INTM x, const INTM y, const INTM z) {
            _vec.setData(X,x*y*z);
            _x=x;
            _y=y;
            _z=z;
        };
        inline INTM x() const {return _x; };
        inline INTM y() const {return _y; };
        inline INTM z() const {return _z; };
        inline void print() const { _vec.print("map"); };
        inline void print_size() const { printf("%d x %d x %d \n",_x,_y,_z); };
        inline T* rawX() const {return _vec.rawX(); };
        void two_dim_gradient(Map<T>& DX, Map<T>& DY) const;
        void subsampling(Map<T>& out,const int factor, const T beta) const;
        void subsampling_new(Map<T>& out,const int factor, const T beta, const bool verbose = false) const;
        void subsampling_new2(Map<T>& out,const T factor, const T beta, const bool verbose = false) const;
        void subsampling_new3(Map<T>& out,const T factor, const T beta, const bool verbose = false) const;
        void upscaling_new(Map<T>& out,const int factor, const T beta, const bool verbose = false) const;
        void upscaling_new2(Map<T>& out,const T factor, const T beta, const bool verbose = false) const;
        void upscaling_new3(Map<T>& out,const T factor, const T beta, const bool verbose = false) const;
        void im2col(Matrix<T>& out,const int e, const bool zero_pad = false, const int stride = 1, const bool normal = false) const;
        void col2im(const Matrix<T>& in,const int e, const bool zero_pad = false, const bool normal = true, const int stride = 1);
        void refMat(Matrix<T>& out) const { out.setData(_vec.rawX(),_x,_y*_z); };
        void refVec(Vector<T>& out) const { out.setData(_vec.rawX(),_x*_y*_z); };
    private:
        INTM _x;
        INTM _y;
        INTM _z;
        Vector<T> _vec;
};

template <typename T>
void Map<T>::two_dim_gradient(Map<T>& DX, Map<T>& DY) const {
    assert(_x==1);
    DX.resize(_x,_y,_z);
    DY.resize(_x,_y,_z);
    T* dx = DX.rawX();
    T* dy = DY.rawX();
    T* X = _vec.rawX();
    const INTM m = _y;
    // compute DX;
    cblas_copy<T>(m*(_z-1),X+m,1,dx,1);
    cblas_copy<T>(m,X+(_z-1)*m,1,dx+(_z-1)*m,1);
    cblas_axpy<T>(m,-T(1.0),X,1,dx,1);
    cblas_axpby<T>(m*(_z-2),-T(0.5),X,1,T(0.5),dx+m,1);
    cblas_axpy<T>(m,-T(1.0),X+(_z-2)*m,1,dx+(_z-1)*m,1);
    // compute DY;
#pragma omp parallel for
    for (INTM ii=0; ii<_z; ++ii) {
        const INTM ind=ii*m;
        dy[ind]=X[ind+1]-X[ind];
        cblas_copy<T>(m-2,X+ind+2,1,dy+ind+1,1);
        cblas_axpby<T>(m-2,-T(0.5),X+ind,1,T(0.5),dy+ind+1,1);
        dy[ind+m-1]=X[ind+m-1]-X[ind+m-2];
    }
};

template <typename T>
void Map<T>::subsampling(Map<T>& out,const int sub, const T sigma) const {
    const INTM sizeimage=_y*_z;
    const INTM h = _y;
    const INTM w = _z;
    const INTM s = 2*sub+1;
    T* filt = new T[s];
    T sum = 0;
    for(int i=-sub; i<=sub; ++i){
        filt[i+sub] = exp(-(1.0/(2*sigma*sigma))*i*i);
        sum += filt[i+sub];
    }
    for(int i=0; i<s; ++i){
        filt[i] /= sum;
    }
    const INTM offset = ceil(float(sub) / 2)-1;
    const INTM hout = ceil((float(h) - offset)/sub);
    const INTM wout = ceil((float(w) - offset)/sub);
    const INTM sizeimageout = hout * wout;
    const INTM m = _x;
    out.resize(_x,hout,wout);
    T* prIn =_vec.rawX();
    T* prOut =out._vec.rawX();
    memset(prOut, 0, sizeof(T)*hout*wout*m);
    T* buff = new T[m*w*hout];

#pragma omp parallel for
    for(INTM ii = 0; ii<w; ++ii) {
        for(INTM jj = 0; jj<hout; ++jj) {
            const INTM dx=jj*sub+offset;
            const INTM discardx =-MIN(dx - sub,0);
            const INTM discardy =MAX(dx + sub - h +1,0);
            cblas_gemv<T>(CblasColMajor,CblasNoTrans,m,s-discardx-discardy,T(1.0),prIn + (ii*h+dx -sub + discardx)*m,m,filt+discardx,1,0,buff+(jj*w+ii)*m,1);
        }
    }

#pragma omp parallel for
    for (INTM ii=0; ii<hout; ++ii) {
        for (INTM jj=0; jj<wout; ++jj) {
            const INTM dx=jj*sub+offset;
            const INTM discardx =-MIN(dx - sub,0);
            const INTM discardy =MAX(dx + sub - w +1,0);
            cblas_gemv<T>(CblasColMajor,CblasNoTrans,m,s-discardx-discardy,T(1.0),buff + (ii*w+dx -sub + discardx)*m,m,filt+discardx,1,0,prOut+(jj*hout+ii)*m,1);
        }
    }
    delete[](filt);
    delete[](buff);

};

template <typename T>
void Map<T>::im2col(Matrix<T>& out,const int e, const bool zero_pad, const int stride, const bool normal) const {
    const INTM n1=zero_pad ? _y : _y-e+1;
    const INTM n2=zero_pad ? _z : _z-e+1;
    const INTM s=e*e*_x;
    T* X=_vec.rawX();
    const INTM nn1=ceil(n1 / static_cast<double>(stride));
    const INTM nn2=ceil(n2 / static_cast<double>(stride));
    out.resize(s,nn1*nn2);
    T* Y=out.rawX();
    out.setZeros();
    INTM num_patch=0;
    if (zero_pad) {
        const INTM ed2=e/2;
        for (INTM jj=0; jj<n2; jj+=stride) {
            for (INTM ii=0; ii<n1; ii+=stride) {
                const int kmin = MAX(jj-ed2,0);
                const int kmax = MIN(jj-ed2+e,_z);
                const int imin = MAX(ii-ed2,0);
                const int imax = MIN(ii-ed2+e,_y);
                const INTM ex=(imax-imin)*_x;
                if (normal) {
                    for (INTM kk=kmin; kk<kmax; ++kk) {
                        //cblas_axpy<T>(ex,1,X+(kk*_y+imin)*_x,1,Y+num_patch*s+((kk-jj+ed2)*e+(imin-ii+ed2))*_x,1);
                        cblas_axpy<T>(ex,static_cast<T>(e*e)/((kmax-kmin)*(imax-imin)),X+(kk*_y+imin)*_x,1,Y+num_patch*s+((kk-jj+ed2)*e+(imin-ii+ed2))*_x,1);
                        //memcpy(Y+num_patch*s+((kk-jj+ed2)*e+(imin-ii+ed2))*_x,X+(kk*_y+imin)*_x,ex*sizeof(T));
                    }
                } else {
                    for (INTM kk=kmin; kk<kmax; ++kk) {
                        memcpy(Y+num_patch*s+((kk-jj+ed2)*e+(imin-ii+ed2))*_x,X+(kk*_y+imin)*_x,ex*sizeof(T));
                    }
                }
                ++num_patch;
            }
        }
    } else {
        const INTM ex=e*_x;
        for (INTM jj=0; jj<n2; jj+=stride) {
            for (INTM ii=0; ii<n1; ii+=stride) {
                for (INTM kk=0; kk<e; ++kk) {
                    memcpy(Y+num_patch*s+kk*ex,X+((jj+kk)*_y+ii)*_x,ex*sizeof(T));
                }
                ++num_patch;
            }
        }
    }
}

template <typename T>
void Map<T>::col2im(const Matrix<T>& in,const int e, const bool zero_pad, const bool norm, const int stride) {
    const INTM n1=zero_pad ? _y : _y-e+1;
    const INTM n2=zero_pad ? _z : _z-e+1;
    const INTM s=e*e*_x;
    _vec.setZeros();
    Vector<T> count;
    count.copy(_vec);
    T* X=_vec.rawX();
    T* pr_count=count.rawX();
    T* Y=in.rawX();
    _vec.setZeros();
    INTM num_patch=0;
    if (zero_pad) {
        const INTM ed2=e/2;
        for (INTM jj=0; jj<n2; jj+=stride) {
            for (INTM ii=0; ii<n1; ii+=stride) {
                const int kmin = MAX(jj-ed2,0);
                const int kmax = MIN(jj-ed2+e,_z);
                const int imin = MAX(ii-ed2,0);
                const int imax = MIN(ii-ed2+e,_y);
                const INTM ex=(imax-imin)*_x;
                for (INTM kk=kmin; kk<kmax; ++kk) {
                    cblas_axpy<T>(ex,T(1.0),Y+num_patch*s+((kk-jj+ed2)*e+(imin-ii+ed2))*_x,1,X+(kk*_y+imin)*_x,1);
                }
                if (norm) {
                    for (INTM kk=kmin; kk<kmax; ++kk) {
                        for (INTM ll=0; ll<ex; ++ll)
                            pr_count[(kk*_y+imin)*_x+ll]++;
                    }
                }
                ++num_patch;
            }
        }
    } else {
        const INTM ex=e*_x;
        for (INTM jj=0; jj<n2; jj+=stride) {
            for (INTM ii=0; ii<n1; ii+=stride) {
                for (INTM kk=0; kk<e; ++kk) {
                    cblas_axpy<T>(ex,T(1.0),Y+num_patch*s+kk*ex,1,X+((jj+kk)*_y+ii)*_x,1);
                }
                if (norm) {
                    for (INTM kk=0; kk<e; ++kk) {
                        for (INTM ll=0; ll<ex; ++ll)
                            pr_count[((jj+kk)*_y+ii)*_x+ll]++;
                    }
                }
                ++num_patch;
            }
        }
    }
    if (norm)
        _vec.div(count);
}

// out is the high resolution image
template <typename T>
void Map<T>::upscaling_new(Map<T>& out,const int sub, const T sigma, const bool verbose) const {
    const INTM sizeimage=out._y*out._z;
    const INTM h = out._y;
    const INTM w = out._z;
    const INTM hin = ceil((float(h))/sub);
    const INTM win = ceil((float(w))/sub);
    const INTM diffh = h- ((hin-1)*sub+1);
    const bool even = diffh % 2 == 1;
    const INTM offset = even ? diffh/2 + 1 : diffh/2;
    const INTM s = even ? 2*sub: 2*sub+1;
    if (verbose && even)
        printf("Even filter\n");
    if (verbose)
        printf("Offset %d\n",offset);
    T* filt = new T[s];
    T sum = 0;
    if (even) {
        for(int i=-sub; i<sub; ++i){
            const T ind=i+T(0.5);
            filt[i+sub] = exp(-(1.0/(2*sigma*sigma))*ind*ind);
            sum += filt[i+sub];
        }
    } else {
        for(int i=-sub; i<=sub; ++i){
            filt[i+sub] = exp(-(1.0/(2*sigma*sigma))*i*i);
            sum += filt[i+sub];
        }
    }
    for(int i=0; i<s; ++i){
        filt[i] /= sum;
    }
    const INTM m = _x;
    if (hin != _y || win != _z) {
        printf("Wrong image input size\n");
        return;
    }
    out._vec.setZeros();
    //out.resize(_x,hout,wout);
    T* prIn =_vec.rawX();
    T* prOut =out._vec.rawX();
    T* buff = new T[m*w*hin];
    memset(buff,0,sizeof(T)*m*w*hin);
    const T subdisc = even ? sub-1 : sub;
    for (INTM ii=0; ii<hin; ++ii) {
        for (INTM jj=0; jj<win; ++jj) {
            const INTM dx=jj*sub+offset;
            const INTM discardx =-MIN(dx - sub,0);
            const INTM discardy =MAX(dx + subdisc - w +1,0);
            cblas_ger<T>(CblasColMajor,m,s-discardx-discardy,T(1.0),prIn+(jj*hin+ii)*m,1,filt+discardx,1,
                    buff + (ii*w+dx -sub + discardx)*m,m);
//            cblas_gemv<T>(CblasColMajor,CblasNoTrans,m,s-discardx-discardy,T(1.0),buff + (ii*w+dx -sub + discardx)*m,m,filt+discardx,1,0,prOut+(jj*hin+ii)*m,1);
        }
    }
    for(INTM ii = 0; ii<w; ++ii) {
        for(INTM jj = 0; jj<hin; ++jj) {
            const INTM dx=jj*sub+offset;
            const INTM discardx =-MIN(dx - sub,0);
            const INTM discardy =MAX(dx + subdisc - h +1,0);
            cblas_ger<T>(CblasColMajor,m,s-discardx-discardy,T(1.0),buff+(jj*w+ii)*m,1,filt+discardx,1,
                    prOut + (ii*h+dx -sub + discardx)*m,m);
//          cblas_gemv<T>(CblasColMajor,CblasNoTrans,m,s-discardx-discardy,T(1.0),prIn + (ii*h+dx -sub + discardx)*m,m,filt+discardx,1,0,buff+(jj*w+ii)*m,1);
        }
    }
    delete[](filt);
    delete[](buff);

};

template <typename T>
void Map<T>::subsampling_new(Map<T>& out,const int sub, const T sigma, const bool verbose) const {
    const INTM sizeimage=_y*_z;
    const INTM h = _y;
    const INTM w = _z;
    const INTM hout = ceil((float(h))/sub);
    const INTM wout = ceil((float(w))/sub);
    const INTM diffh = h- ((hout-1)*sub+1);
    const bool even = diffh % 2 == 1;
    const INTM offset = even ? diffh/2 + 1 : diffh/2;
    const INTM s = even ? 2*sub: 2*sub+1;
    if (verbose && even)
        printf("Even filter\n");
    if (verbose)
        printf("Offset %d\n",offset);
    T* filt = new T[s];
    T sum = 0;
    if (even) {
        for(int i=-sub; i<sub; ++i){
            const T ind=i+T(0.5);
            filt[i+sub] = exp(-(1.0/(2*sigma*sigma))*ind*ind);
            sum += filt[i+sub];
        }
    } else {
        for(int i=-sub; i<=sub; ++i){
            filt[i+sub] = exp(-(1.0/(2*sigma*sigma))*i*i);
            sum += filt[i+sub];
        }
    }
    for(int i=0; i<s; ++i){
        filt[i] /= sum;
    }
    const INTM m = _x;
    out.resize(_x,hout,wout);
    T* prIn =_vec.rawX();
    T* prOut =out._vec.rawX();
    memset(prOut, 0, sizeof(T)*hout*wout*m);
    T* buff = new T[m*w*hout];
    const T subdisc = even ? sub-1 : sub;

#pragma omp parallel for
    for(INTM ii = 0; ii<w; ++ii) {
        for(INTM jj = 0; jj<hout; ++jj) {
            const INTM dx=jj*sub+offset;
            const INTM discardx =-MIN(dx - sub,0);
            const INTM discardy =MAX(dx + subdisc - h +1,0);
            cblas_gemv<T>(CblasColMajor,CblasNoTrans,m,s-discardx-discardy,T(1.0),prIn + (ii*h+dx -sub + discardx)*m,m,filt+discardx,1,0,buff+(jj*w+ii)*m,1);
        }
    }

#pragma omp parallel for
    for (INTM ii=0; ii<hout; ++ii) {
        for (INTM jj=0; jj<wout; ++jj) {
            const INTM dx=jj*sub+offset;
            const INTM discardx =-MIN(dx - sub,0);
            const INTM discardy =MAX(dx + subdisc - w +1,0);
            cblas_gemv<T>(CblasColMajor,CblasNoTrans,m,s-discardx-discardy,T(1.0),buff + (ii*w+dx -sub + discardx)*m,m,filt+discardx,1,0,prOut+(jj*hout+ii)*m,1);
        }
    }
    delete[](filt);
    delete[](buff);
};

template <typename T>
void Map<T>::subsampling_new2(Map<T>& out,const T sub, const T sigma, const bool verbose) const {
    const INTM sizeimage=_y*_z;
    const INTM h = _y;
    const INTM w = _z;
    const INTM hout = lrint((T(h))/sub);
    const INTM wout = lrint((T(w))/sub);
    const T diffH = T(h)/T(hout);
    const T diffW = T(w)/T(wout);
    const T offsetH = diffH/2;
    const T offsetW = diffW/2;
    const INTM m = _x;
    out.resize(_x,hout,wout);
    T* prIn =_vec.rawX();
    T* prOut =out._vec.rawX();
    memset(prOut, 0, sizeof(T)*hout*wout*m);
    INTM s = ceil(2*sigma);
    const T fact = -1/(sigma*sigma);
#pragma omp parallel for
    for(INTM ii = 0; ii<wout; ++ii) {
        const T ixf = offsetW + ii*diffW;
        const INTM ix = lrint(ixf);
        const INTM indminx = MAX(ix-s,0);
        const INTM indmaxx = MIN(ix+s,w-1);
        for(INTM jj = 0; jj<hout; ++jj) {
            const T jyf = offsetH + jj*diffH;
            const INTM jy = lrint(jyf);
            const INTM indminy = MAX(jy-s,0);
            const INTM indmaxy = MIN(jy+s,h-1);
            for(INTM kk = indminx; kk<=indmaxx; ++kk) {
                const T dx = kk-ixf;
                for(INTM ll = indminy; ll<=indmaxy; ++ll) {
                    const T dy = ll-jyf;
                    cblas_axpy<T>(m,expf(fact*(dx*dx+dy*dy)),prIn+(kk*h+ll)*m,1,prOut+(ii*hout+jj)*m,1);
                }
            }
        }
    }
};

template <typename T>
void Map<T>::subsampling_new3(Map<T>& out,const T sub, const T sigma, const bool verbose) const {
    const INTM sizeimage=_y*_z;
    const INTM h = _y;
    const INTM w = _z;
    const INTM hout = lround((T(h))/sub);
    const INTM wout = lround((T(w))/sub);
    const T diffH = T(h)/T(hout);
    const T diffW = T(w)/T(wout);
    const T offsetH = diffH/2;
    const T offsetW = diffW/2;
    const INTM m = _x;
    out.resize(_x,hout,wout);
    T* prIn =_vec.rawX();
    T* prOut =out._vec.rawX();
    memset(prOut, 0, sizeof(T)*hout*wout*m);
    INTM s = ceil(3*sigma);
    const T fact = -1/(sigma*sigma);
#pragma omp parallel for
    for(INTM ii = 0; ii<wout; ++ii) {
        const T ixf = offsetW + ii*diffW;
        const INTM ix = lround(ixf);
        const INTM indminx = MAX(ix-s,0);
        const INTM indmaxx = MIN(ix+s,w-1);
        for(INTM jj = 0; jj<hout; ++jj) {
            const T jyf = offsetH + jj*diffH;
            const INTM jy = lround(jyf);
            const INTM indminy = MAX(jy-s,0);
            const INTM indmaxy = MIN(jy+s,h-1);
            for(INTM kk = indminx; kk<=indmaxx; ++kk) {
                const T dx = kk+T(0.5)-ixf;
                for(INTM ll = indminy; ll<=indmaxy; ++ll) {
                    const T dy = ll+T(0.5)-jyf;
                    cblas_axpy<T>(m,expf(fact*(dx*dx+dy*dy)),prIn+(kk*h+ll)*m,1,prOut+(ii*hout+jj)*m,1);
                }
            }
        }
    }
};


template <typename T>
void Map<T>::upscaling_new2(Map<T>& out,const T sub, const T sigma, const bool verbose) const {
    const INTM sizeimage=out._y*out._z;
    const INTM h = out._y;
    const INTM w = out._z;
    const INTM hout = lrint((T(h))/sub);
    const INTM wout = lrint((T(w))/sub);
    const T diffH = T(h)/T(hout);
    const T diffW = T(w)/T(wout);
    const T offsetH = diffH/2;
    const T offsetW = diffW/2;
    const INTM m = out._x;
    T* prIn =_vec.rawX();
    T* prOut =out._vec.rawX();
    memset(prOut, 0, sizeof(T)*h*w*m);
    INTM s = ceil(2*sigma);
    const T fact = -1/(2*sigma*sigma);
#pragma omp parallel for
    for(INTM ii = 0; ii<wout; ++ii) {
        const T ixf = offsetW + ii*diffW;
        const INTM ix = lrint(ixf);
        const INTM indminx = MAX(ix-s,0);
        const INTM indmaxx = MIN(ix+s,w-1);
        for(INTM jj = 0; jj<hout; ++jj) {
            const T jyf = offsetH + jj*diffH;
            const INTM jy = lrint(jyf);
            const INTM indminy = MAX(jy-s,0);
            const INTM indmaxy = MIN(jy+s,h-1);
            for(INTM kk = indminx; kk<=indmaxx; ++kk) {
                const T dx = kk-ixf;
                for(INTM ll = indminy; ll<=indmaxy; ++ll) {
                    const T dy = ll-jyf;
                    cblas_axpy<T>(m,expf(fact*(dx*dx+dy*dy)),prIn+(ii*hout+jj)*m,1,prOut+(kk*h+ll)*m,1);
                }
            }
        }
    }
};

template <typename T>
void Map<T>::upscaling_new3(Map<T>& out,const T sub, const T sigma, const bool verbose) const {
    const INTM sizeimage=out._y*out._z;
    const INTM h = out._y;
    const INTM w = out._z;
    const INTM hout =lround((T(h))/sub);
    const INTM wout = lround((T(w))/sub);
    const T diffH = T(h)/T(hout);
    const T diffW = T(w)/T(wout);
    const T offsetH = diffH/2;
    const T offsetW = diffW/2;
    const INTM m = out._x;
    T* prIn =_vec.rawX();
    T* prOut =out._vec.rawX();
    memset(prOut, 0, sizeof(T)*h*w*m);
    INTM s = ceil(3*sigma);
    const T fact = -1/(2*sigma*sigma);
#pragma omp parallel for
    for(INTM ii = 0; ii<wout; ++ii) {
        const T ixf = offsetW + ii*diffW;
        const INTM ix = lround(ixf);
        const INTM indminx = MAX(ix-s,0);
        const INTM indmaxx = MIN(ix+s,w-1);
        for(INTM jj = 0; jj<hout; ++jj) {
            const T jyf = offsetH + jj*diffH;
            const INTM jy = lround(jyf);
            const INTM indminy = MAX(jy-s,0);
            const INTM indmaxy = MIN(jy+s,h-1);
            for(INTM kk = indminx; kk<=indmaxx; ++kk) {
                const T dx = kk+T(0.5)-ixf;
                for(INTM ll = indminy; ll<=indmaxy; ++ll) {
                    const T dy = ll+T(0.5)-jyf;
                    cblas_axpy<T>(m,expf(fact*(dx*dx+dy*dy)),prIn+(ii*hout+jj)*m,1,prOut+(kk*h+ll)*m,1);
                }
            }
        }
    }
};


template <typename T>
inline T convert_image_data(const double in) {
    return static_cast<T>(in);
};

template <typename T>
inline T convert_image_data(const float in) {
    return static_cast<T>(in);
};

template <typename T>
inline T convert_image_data(const unsigned char in) {
    return static_cast<T>(in)/255;
};

template <typename Tin, typename Tout>
inline void convert_image_data_map(const Tin* input, Tout* output, const int n) {
    for (int ii=0; ii<n; ++ii) output[ii]=convert_image_data<Tout>(input[ii]);
};

template <typename Tin, typename Tout>
inline void convert_image_data_map_switch(const Tin* input, Tout* output, const int nc, const int channels, const int nimages) {
    for (int ii=0; ii<nimages; ++ii)
        for (int jj=0; jj<channels; ++jj)
            for (int kk=0; kk<nc; ++kk)
                output[ii*nc*channels+kk*channels+jj]=convert_image_data<Tout>(input[ii*nc*channels+jj*nc+kk]);
};

template <typename T>
inline void convert_nchw_to_nhwc(T* output,const T* input,const int h, const int w, const int c, const int n) {
    const int s=h*w;
    const int nc=h*w*c;
    for (int ii=0; ii<n; ++ii)
        for (int jj=0; jj<c; ++jj)
            for (int kk=0; kk<s; ++kk)
                output[ii*nc+kk*c+jj]=input[ii*nc+jj*s+kk];
};



template <typename Tin, typename T>
inline void get_zeromap(const Map<Tin>& mapin, Map<T>& map, const int type_layer) {
    Tin* pr_im = mapin.rawX();
    const INTM ex = mapin.x(); // image is assumed to be e x e x mapin.z
    const INTM ey = mapin.y(); // image is assumed to be e x e x mapin.z
    if (ey == 3*ex) { // assumes this means RGB  (RRR,GGG,BBB)
        if (type_layer==4) {
            map.resize(1,ex,ex); // extract green channel
            T* X = map.rawX();
            for (INTM ii=0; ii<ex; ++ii)
                for (INTM jj=0; jj<ex; ++jj)
                    X[ii*ex+jj]=convert_image_data<T>(pr_im[ii*ex+ex*ex+jj]);
        } else {
            map.resize(3,ex,ex);
            T* X = map.rawX();
            for (INTM kk=0; kk<3; ++kk)
                for (INTM ii=0; ii<ex; ++ii)
                    for (INTM jj=0; jj<ex; ++jj)
                        X[(ii*ex+jj)*3+kk]=convert_image_data<T>(pr_im[ii*ex+kk*ex*ex+jj]); // output is R,G,B,R,G,B,R,G,B
        }
    } else {// assumes this means gray scale
        map.resize(1,ex,ey);
        T* X = map.rawX();
        for (INTM ii=0; ii<ey; ++ii)
            for (INTM jj=0; jj<ex; ++jj)
                X[ii*ex+jj]=convert_image_data<T>(pr_im[ii*ex+jj]);
    }
};

typedef enum
{
    SQLOSS = 0,
    ONE_VS_ALL_SQHINGE_LOSS = 1,
    SQLOSS_CONV = 2,
    NEGNORM = 3
} loss_t;

typedef enum
{
    POOL_GAUSSIAN_FILTER = 0,
    POOL_AVERAGE = 1,
    POOL_AVERAGE2 = 2
} pooling_mode_t;

template <typename T> struct Layer {
    int num_layer;
    int npatch;
    int nfilters;
    int subsampling;
    int stride;
    T sub_float;
    bool new_subsampling;
    bool zero_padding;
    int type_layer;
    /// 0 = RAW
    /// 1 = centering
    /// 2 = centering + whitening
    /// 3 = whitening
    /// 4 = gradient
    /// 5 = centering + whitening per image
    int type_kernel;
    /// 0 = Gaussian
    /// 1 = polynomial, degree 2 (x'x + 1)^2
    /// 2 = polynomial, degree 3 (x'x + 1)^3
    /// 3 = polynomial, degree 2 (no filters)
    /// 4 = polynomial, degree 2 (x'x)^2
    int type_learning;
    /// 0 = Integral approximation
    /// 1 = Optimized Nystrom
    int type_regul;
    /// 0 = no regul
    /// 1 = with regul
    T sigma;
    Matrix<T> W;
    Vector<T> b;
    Matrix<T> Wfilt;
    Vector<T> mu;
    Matrix<T> W2;
    Matrix<T> W3;
    Matrix<T> W4;
    Matrix<T> gradW;
    pooling_mode_t pooling_mode;
};

template <typename T>
inline void pre_processing(Matrix<T>& X, const Layer<T>& layer, const int channels) {
    if (layer.type_layer == 1 || layer.type_layer == 2 || layer.type_layer == 5)
        centering(X,channels);
    if (layer.type_layer == 5)
        whitening(X);
    if (layer.type_layer == 2 || layer.type_layer==3) {
        Vector<T> ones(X.n());
        ones.set(T(1.0));
        X.rank1Update(layer.mu,ones,-T(1.0));
        if (layer.type_layer==2) {
            Matrix<T> Z;
            Z.copy(X);
            layer.Wfilt.mult(Z,X);
        }
    }
}

template <typename T>
inline void encode_layer(const Map<T>& mapin, Map<T>& mapout, const Layer<T>& layer,const bool verbose = false, const bool compute_covs = false, const bool normal = false) {
    Map<T> map;
    if (layer.num_layer==1 && layer.type_layer==4) {
        Map<T> DX, DY;
        mapin.two_dim_gradient(DY,DX);
        const INTM num_orients=layer.nfilters;
        const INTM mx = DX.y();
        const INTM my = DX.z();
        map.resize(num_orients,mx,my);
        const T* dx = DX.rawX();
        const T* dy = DY.rawX();
        T* X = map.rawX();
        Vector<T> theta(num_orients);
        Vector<T> costheta(num_orients);
        Vector<T> sintheta(num_orients);
        const T sigma=layer.sigma;
        for (int ii=0; ii<num_orients; ++ii) theta[ii]=(2*PI/num_orients)*ii;
        for (int ii=0; ii<num_orients; ++ii) costheta[ii]=cos(theta[ii]);
        for (int ii=0; ii<num_orients; ++ii) sintheta[ii]=sin(theta[ii]);
#pragma omp parallel for
        for (INTM ii=0; ii<mx; ++ii) {
            for (INTM jj=0; jj<my; ++jj) {
                const T rho = sqr<T>(dx[ii*my+jj]*dx[ii*my+jj]+dy[ii*my+jj]*dy[ii*my+jj]);
                for (INTM kk=0; kk<num_orients; ++kk) {
                    const T ddx=(dx[ii*my+jj]/rho-costheta[kk]);
                    const T ddy=(dy[ii*my+jj]/rho-sintheta[kk]);
                    X[ii*(num_orients*my) +jj*num_orients+kk] = rho ? rho*exp(-(T(1.0)/(2*sigma*sigma))*(ddx*ddx+ddy*ddy)) : 0;
                }
            }
        }
    } else {
        Matrix<T> X;
        mapin.im2col(X,layer.npatch,layer.zero_padding,layer.stride,normal);
        pre_processing(X,layer,layer.num_layer==1 ? mapin.x() : 1);
        const int yyout = layer.zero_padding ? mapin.y() : mapin.y() - layer.npatch + 1;
        const int zzout = layer.zero_padding ? mapin.z() : mapin.z() - layer.npatch + 1;
        const int yout = ceil(yyout/static_cast<double>(layer.stride));
        const int zout = ceil(zzout/static_cast<double>(layer.stride));
        if (compute_covs) {
            X.mult(X,const_cast<Matrix<T>&>(layer.W3),false,true,T(1.0)/X.n(),T(1.0));
        }
        if (layer.type_kernel==3) {
            const int newp = X.m()*X.m();
            map.resize(newp, yout, zout);
            Matrix<T> Y(map.rawX(), newp, yout*zout);
            expand_XXt(X,Y);
            //Y.multDiagRight(norms);
        } else {
            Vector<T> ones(X.n());
            ones.set(T(1.0));
            Vector<T> norms;
            normalize(X,norms);
            map.resize(layer.W.n(), yout, zout);
            if (verbose) {
                PRINT_I(map.x())
                PRINT_I(map.y())
                PRINT_I(map.z())
            }
            Matrix<T> Y(map.rawX(), layer.W.n(), yout*zout);
            layer.W.mult(X,Y,true);
            if (layer.type_learning==0) {
                if (layer.type_kernel==0) {
                    Y.rank1Update(layer.b,ones);
                    Y.exp();
                } else if (layer.type_kernel==1) {
                    Y.thrsPos();
                } else if (layer.type_kernel==2) {
                    Y.thrsPos();
                    Y.sqr();
                }
            } else if (layer.type_learning>=1) {
                if (layer.type_kernel==0) {
                    Y.rank1Update(layer.b,ones);
                    Y.exp();
                } else if (layer.type_kernel==1) {
                    Y.rank1Update(layer.b,ones);
                    Y.scal(T(0.5));
                    Y.pow(T(2.0));
                } else if (layer.type_kernel==2) {
                    Y.rank1Update(layer.b,ones);
                    Y.scal(T(0.5));
                    Y.pow(T(3.0));
                } else if (layer.type_kernel==4) {
                    Y.pow(T(2.0));
                }
            }
            Y.multDiagRight(norms);
        }
    }
    if (layer.subsampling == 0) {
        const T beta = layer.new_subsampling ? layer.sub_float/sqr<T>(T(2.0)) :  layer.sub_float/(T(2.0));
        map.subsampling_new2(mapout,layer.sub_float,beta,verbose);
    } else if (layer.subsampling == -1) {
        const T beta = layer.new_subsampling ? layer.sub_float/sqr<T>(T(2.0)) :  layer.sub_float/(T(2.0));
        map.subsampling_new3(mapout,layer.sub_float,beta,verbose);
    } else if (layer.subsampling == -2) {
        const T beta = layer.new_subsampling ? layer.sub_float/sqr<T>(T(2.0)) :  layer.sub_float/(T(2.0));
        map.subsampling_new3(mapout,T(1.0),beta,verbose);
    } else if (layer.subsampling > 1) {
        const T beta =layer.subsampling/sqr<T>(T(2.0));
        if (verbose) {
            if (layer.new_subsampling)
                printf("New subsampling scheme\n");
            printf("Subsampling by %d\n",layer.subsampling);
        }
        if (layer.new_subsampling) {
            map.subsampling_new(mapout,layer.subsampling,beta,verbose);
        } else {
            map.subsampling(mapout,layer.subsampling,beta);
        }
    } else {
        mapout.copy(map);
    }
    if (layer.type_learning >= 1 && layer.W2.n() > 1) {
        Matrix<T> Z(mapout.rawX(), mapout.x(), mapout.y()*mapout.z());
        Matrix<T> Z2;
        Z2.copy(Z);
        layer.W2.mult(Z2,Z);
    }
}

template <typename T>
inline void encode_layer_mspace(const Map<T>& mapin, Map<T>& mapout, const Layer<T>& layer,const bool verbose = false) {
    Map<T> map;
    Matrix<T> X;
    mapin.im2col(X,layer.npatch,layer.zero_padding,layer.stride);
    pre_processing(X,layer,layer.num_layer==1 ? mapin.x() : 1);
    const int yyout = layer.zero_padding ? mapin.y() : mapin.y() - layer.npatch + 1;
    const int zzout = layer.zero_padding ? mapin.z() : mapin.z() - layer.npatch + 1;
    const int yout = ceil(yyout/static_cast<double>(layer.stride));
    const int zout = ceil(zzout/static_cast<double>(layer.stride));
    const int nspaces=layer.W2.n()/layer.W2.m()-1;
    const int p = layer.W2.m();
    const int n = yout*zout;
    const int m = layer.W2.n();
    map.resize(layer.W2.n(), yout, zout);
    Matrix<T> Ymap;
    map.refMat(Ymap);
    T* prYmap = Ymap.rawX();
    Ymap.setZeros();
    Vector<T> ones(X.n());
    ones.set(T(1.0));
    Matrix<T> Y(p, yout*zout);
    Matrix<T> Y2(p, yout*zout);
    const T* prY = Y.rawX();
    Matrix<T> W;
    Matrix<T> W2;
    Vector<T> b;
    Vector<T> norms;
    Vector<T> col, col2, col3;

    normalize(X,norms);
    layer.W.refSubMat(0,p,W);
    layer.W2.refSubMat(0,p,W2);
    layer.b.refSubVec(0,p,b);
    W.mult(X,Y2,true);
    List<int> lists[nspaces];
    for (int ii=0; ii< Y.n(); ++ii) {
        Y2.refCol(ii,col);
        lists[col.max()].push_back(ii);
    }
    Y2.rank1Update(b,ones);
    Y2.exp();
    Y2.multDiagRight(norms);
    W2.mult(Y2,Y);
    for (int ii=0; ii<n;++ii)
        memcpy(prYmap+m*ii,prY+p*ii,p*sizeof(T));

    for (int ii=0; ii<nspaces; ++ii) {
        layer.W.refSubMat((ii+1)*p,p,W);
        layer.W2.refSubMat((ii+1)*p,p,W2);
        layer.b.refSubVec((ii+1)*p,p,b);
        List<int>& list_space = lists[ii];
        for (ListIterator<int> it=list_space.begin();
                it != list_space.end(); ++it) {
            X.refCol(*it,col);
            Y2.refCol(*it,col2);
            Y.refCol(*it,col3);
            W.multTrans(col,col2);
            col2.add(b);
            col2.exp();
            col2.scal(norms[*it]);
            W2.mult(col2,col3);
            memcpy(prYmap+m*(*it)+ (ii+1)*p,prY+p*(*it),p*sizeof(T));
        }
    }

    if (layer.subsampling == 0) {
        const T beta = layer.new_subsampling ? layer.sub_float/sqr<T>(T(2.0)) :  layer.sub_float/(T(2.0));
        map.subsampling_new2(mapout,layer.sub_float,beta,verbose);
    } else if (layer.subsampling > 1) {
        const T beta =layer.subsampling/sqr<T>(T(2.0));
        if (layer.new_subsampling) {
            map.subsampling_new(mapout,layer.subsampling,beta,verbose);
        } else {
            map.subsampling(mapout,layer.subsampling,beta);
        }
    } else {
        mapout.copy(map);
    }
}



template <typename Tin, typename T>
inline void encode_ckn(const Map<Tin>& maps, Layer<T> layers[], const int nlayers, Matrix<T>& psi, const bool compute_covs = false, const bool normal = false, const int subspaces = 1) {
    Timer time;
    time.start();
    const int n = maps.z();
    if (compute_covs) {
        for (int ii=0; ii<nlayers; ++ii)
            layers[ii].W3.setZeros();
    }
    int count=0;
#pragma omp parallel for
    for (int ii=0; ii<n; ++ii) {
        Map<T> map;
        Map<Tin> mapii;
        maps.refSubMapZ(ii,mapii);
        encode_ckn_map(mapii,layers,nlayers,map,false,compute_covs && ii < 50000, normal,subspaces);
        memcpy(psi.rawX()+ii*psi.m(),map.rawX(),psi.m()*sizeof(T));
    }
    time.printElapsed();
    if (compute_covs) {
        for (int ii=0; ii<nlayers; ++ii)
            layers[ii].W3.scal(T(1.0)/MIN(50000,n));
    }
};

template <typename Tin, typename T>
inline void encode_ckn_map(const Map<Tin>& mapin, Layer<T> layers[], const int nlayers, Map<T>& mapout, const bool verbose = false, const bool compute_covs = false, const bool normal = false, const int subspaces = 1) {
    Map<T> maptmp;
    if (verbose) {
        PRINT_I(mapin.x())
        PRINT_I(mapin.y())
        PRINT_I(mapin.z())
    }
    get_zeromap(mapin,mapout,layers[0].type_layer);
    for (int jj=0; jj<nlayers; ++jj) {

        maptmp.copy(mapout);
#ifdef _OPENMP
        int numT=omp_get_thread_num();
#else
        int numT=0;
#endif
        if (jj==nlayers-1 && subspaces > 1) {
            encode_layer_mspace(maptmp,mapout,layers[jj],verbose);
        } else {
            encode_layer(maptmp,mapout,layers[jj],verbose,compute_covs,normal);
        }
        if (verbose) {
            PRINT_I(jj)
            PRINT_I(mapout.x())
            PRINT_I(mapout.y())
            PRINT_I(mapout.z())
        }
    }
};

template <typename Tin, typename T>
inline void extract_dataset(const Map<Tin>& maps, Layer<T> layers[], const int nlayers, Matrix<T>& X, Vector<int>& labels) {
    Timer time;
    time.start();
    const int n = maps.z();
    const int per_image=X.n()/n;
    Layer<T>& last_layer=layers[nlayers-1];
#pragma omp parallel for
    for (int ii=0; ii<n; ++ii) {
        Map<T> map;
        Map<Tin> mapii;
        maps.refSubMapZ(ii,mapii);
        encode_ckn_map(mapii,layers,nlayers-1,map);
        Matrix<T> psi;
        map.im2col(psi,last_layer.npatch,false,1);

        /// track empty patches
        const int m = psi.n();
        Matrix<T> psiC;
        psiC.copy(psi);
        centering(psiC,nlayers==1 ? map.x() : 1);
        //Vector<T> norms;
        //psiC.norm_2_cols(norms); // select patches with more variance than the median
        //Vector<T> norms2;  /// one change here
        //norms2.copy(norms);
        //norms2.sort(false);
        //const T thrs = norms2[MIN(MAX(floor(0.75*m),per_image),m-1)];
        Vector<int> per;
        per.randperm(m);
        int count=0;
        Vector<T> col, col2;
        if (last_layer.type_layer == 5)
            whitening(psiC);
        for (int kk=0; kk<m; ++kk) {
          // if (norms[per[kk]] >= thrs) {
                X.refCol(ii*per_image+count,col);
                if (labels.n() > 0) labels[ii*per_image+count]=ii;
                if (last_layer.type_layer==1 || last_layer.type_layer==2 || last_layer.type_layer==5) {
                    psiC.refCol(per[kk],col2);
                } else {
                    psi.refCol(per[kk],col2);
                }
                col.copy(col2);
                //if (nrms.n() > 0) nrms[ii*per_image+count]=col.nrm2();
                ++count;
                if (count==per_image) break;
        //    }
        }
    }
    time.printElapsed();
};


template <typename T>
inline void whitening_map(Map<T>& map, const bool zero_pad = false) {
    Matrix<T> X;
    map.im2col(X,3,zero_pad);
    centering(X,3);
    whitening(X);
    map.col2im(X,3,zero_pad);
};

template <typename T>
inline void whitening_maps(Map<T>& maps) {
    Timer time;
    time.start();
    const int n = maps.z();
    int count=0;
#pragma omp parallel for
    for (int ii=0; ii<n; ++ii) {
        Map<T> map, map_zero;
        maps.refSubMapZ(ii,map);
        get_zeromap(map,map_zero,0);
        whitening_map(map_zero);
        T* mapX = map.rawX();
        T* map_zeroX = map_zero.rawX();
        const int c = map_zero.x();
        const int h = map_zero.y();
        const int w = map_zero.z();
        for (int jj = 0; jj<w; ++jj)
            for (int kk=0; kk<h; ++kk)
                for (int ll=0; ll<c; ++ll)
                    mapX[ll*h*w+jj*h+kk]=map_zeroX[jj*(h*c)+kk*c+ll];
    }
    time.printElapsed();
};



#endif

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include "cblas_alt_template.h"

#include "linalg.h"

#include "ctypes_utils.h"
#include "svm.h"

#include <iostream>
using namespace std;

#define MAKE_INIT_NAME(x) init ## x (void)
#define MODNAME_INIT(s) MAKE_INIT_NAME(s)

#define STR_VALUE(arg)        #arg
#define FUNCTION_NAME(name) STR_VALUE(name)

#define MODNAME_STR FUNCTION_NAME(MODNAME)

/*
    Get the include directories within python using

    import distutils.sysconfig
    print distutils.sysconfig.get_python_inc()
    import numpy as np
    print np.get_include()

    gcc  -fPIC -shared -g -Wall -O3 \
    -I /usr/include/python2.7 -I /usr/lib64/python2.7/site-packages/numpy/core/include \
    mymath.c -o mymath.so

*/


template<typename T>
bool all_finite(const T* const x, const int n) {
    bool finite(true);
#pragma omp parallel for shared(finite)
    for (int i=0; i<n; ++i)
        if (!npy_isfinite(x[i]))
            finite = false; // we only ever write this value to finite, so no race condition
    return finite;
}

static PyObject * pymiso_miso_one_vs_rest(PyObject *self, PyObject *args, PyObject *keywds) {
    // inputs
    PyArrayObject* X=NULL;
    PyArrayObject* y=NULL;

    int seed = 0;
    int threads = -1;
    int verbose = 0;
    int accelerated = true;
    int reweighted = 0;
    int non_uniform = false;
    float eps = 1e-3;
    int max_iter = -1;
    float lambda = 1;

    /* parse inputs */
    static char *kwlist[] = {
              "X",
              "y",
              "lambda",
              "max_iter",
              "eps",
              "accelerated",
              "reweighted",
              "non_uniform",
              "threads",
              "seed",
              "verbose",
              NULL};
    const char* format =  "O!O!f|ifpipiii";

    if (!PyArg_ParseTupleAndKeywords(args, keywds, format, kwlist,
                                                &PyArray_Type, &X,
                                                &PyArray_Type, &y,
                                                &lambda,
                                                &max_iter,
                                                &eps,
                                                &accelerated,
                                                &reweighted,
                                                &non_uniform,
                                                &threads,
                                                &seed,
                                                &verbose))
        return NULL;

    srandom(seed);
    Matrix<float> Xmat;
    Vector<float> yvec;
    if (!npyToVector(y, yvec, "y")) return NULL;
    if (!npyToMatrix(X, Xmat, "X")) return NULL;
    if (max_iter <= 0) {
        max_iter = 1000 * Xmat.n();
        if (verbose)
            cout << "Setting max_iter to 1000*n = " << max_iter << endl;
    }

    assert_py_obj(all_finite(Xmat.rawX(), Xmat.m()*Xmat.n()), "X contains inf or nan values!");
    assert_py_obj(all_finite(yvec.rawX(), yvec.n()), "y contains inf or nan values!");

    Vector<int>* iter_count = new Vector<int>();
    Vector<float>* primals = new Vector<float>();
    Vector<float>* losses = new Vector<float>();

    const int num_classes = yvec.maxval()+1;
    Matrix<float>* Wmat = new Matrix<float>(Xmat.m(), num_classes);

    threads = set_omp_threads(threads);

    /* actual computation */
    miso_svm_onevsrest(yvec, Xmat, *Wmat, *iter_count, *primals, *losses, lambda, eps, max_iter, accelerated, reweighted, non_uniform, verbose);

    PyObject* PyW = (PyObject*) wrapMatrix(Wmat);
    PyObject* PyIterCount =  (PyObject*)wrapVector(iter_count);
    PyObject* PyPrimals =  (PyObject*)wrapVector(primals);
    PyObject* PyLosses =  (PyObject*)wrapVector(losses);

    return Py_BuildValue("OOOO", PyW, PyIterCount, PyPrimals, PyLosses);
}


template <typename T>
PyArrayObject* new_array(vector<npy_intp> shape) {
    const int ndim = shape.size();
    PyArrayObject* result = (PyArrayObject *) PyArray_SimpleNew(ndim, shape.data(), getTypeNumber<T>());
    return result;
}


static PyMethodDef method_list[] = {
          {"miso_one_vs_rest",  (PyCFunction)pymiso_miso_one_vs_rest, METH_VARARGS | METH_KEYWORDS, "Train a linear SVM using the MISO algorithm."},
          {NULL, NULL, 0, NULL}          /* Sentinel */
};

static struct PyModuleDef misomodule = {
          PyModuleDef_HEAD_INIT,
          "_miso",    /* name of module */
          NULL, /* module documentation, may be NULL */
          -1,         /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          method_list,
          NULL//, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__miso(void) {

    PyObject* m;
    m = PyModule_Create(&misomodule);
    assert_py_obj(m!=NULL, "failed to create miso module object");

    // initialize wrapper classes
    MatrixWrapperType.tp_new = PyType_GenericNew;
    VectorWrapperType.tp_new = PyType_GenericNew;
    MapWrapperType.tp_new = PyType_GenericNew;
    assert_py_obj(PyType_Ready(&MapWrapperType) >= 0,
                      "Map wrapper type failed to initialize");
    assert_py_obj(PyType_Ready(&MatrixWrapperType) >= 0,
                      "Matrix wrapper type failed to initialize");
    assert_py_obj(PyType_Ready(&VectorWrapperType) >= 0,
                      "Vector wrapper type failed to initialize");

    /* required, otherwise numpy functions do not work */
    import_array();

    Py_INCREF(&MatrixWrapperType);
    Py_INCREF(&MapWrapperType);
    Py_INCREF(&VectorWrapperType);
    PyModule_AddObject(m, "MyDealloc_Type_Mat", (PyObject *)&MatrixWrapperType);
    PyModule_AddObject(m, "MyDealloc_Type_Map", (PyObject *)&MapWrapperType);
    PyModule_AddObject(m, "MyDealloc_Type_Vec", (PyObject *)&VectorWrapperType);

    return m;
}

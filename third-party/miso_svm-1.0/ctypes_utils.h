#include <Python.h>
#include <numpy/arrayobject.h>
#include "linalg.h"

#include "common.h"

/**
 * From Daan Wynen
 */


// check for a condition, and fail with an exception strin set if it is false
#define assert_py_obj(condition, error) if (! (condition) ) { \
    PyErr_SetString(PyExc_TypeError, (error)); \
    return NULL; \
  }

// the same macro, but for cases where the calling method returns an integer
#define assert_py_int(condition, error) if (! (condition) ) { \
    PyErr_SetString(PyExc_TypeError, (error)); \
    return 0; \
  }

// and another version that throws the error message as a const char* instead
#define assert_py_throw(condition, error) if (! (condition) ) { \
    throw (error); \
  }



template <typename T> inline string getTypeName();
template <> inline string getTypeName<int>() { return "intc"; };
template <> inline string getTypeName<unsigned char>() { return "uint8"; };
template <> inline string getTypeName<float>() { return "float32"; };
template <> inline string getTypeName<double>() { return "float64"; };

template <typename T> inline int getTypeNumber();
template <> inline int getTypeNumber<int>() { return NPY_INT; };
template <> inline int getTypeNumber<unsigned char>() { return NPY_UINT8; };
template <> inline int getTypeNumber<float>() { return NPY_FLOAT32; };
template <> inline int getTypeNumber<double>() { return NPY_FLOAT64; };


// these structs hold define the python type objects for Vector, Matrix and Map
// they only hold pointers to the actual C++ objects
// this way, the data does not get deallocated immediately when objects leave
// the scope
template <typename T> struct VectorWrapper {
    PyObject_HEAD;
    Vector<T> *obj;
};

template <typename T> struct MatrixWrapper {
    PyObject_HEAD;
    Matrix<T> *obj;
};

template <typename T> struct MapWrapper {
    PyObject_HEAD;
    Map<T> *obj;
};


// these are the deallocation methods for she structs defined above
// they'll be linked to the destructor hooks of the python objects further down
template <typename T>
static void _delete_cpp_mat(MatrixWrapper<T>* self){
    if (self && self->obj) {
        delete self->obj;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

template <typename T>
static void _delete_cpp_vec(VectorWrapper<T>* self){
    if (self && self->obj) {
        delete self->obj;
    Py_TYPE(self)->tp_free((PyObject*)self);
    }
}

template <typename T>
static void _delete_cpp_map(MapWrapper<T>* self){
    if (self && self->obj) {
        delete self->obj;
    Py_TYPE(self)->tp_free((PyObject*)self);
    }
}


static PyTypeObject MatrixWrapperType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "miso_svm.MatrixWrapper", /*tp_name*/
    sizeof(MatrixWrapper<float>), /*tp_basicsize*/ // FIXME: does this break if using double?
    0, /*tp_itemsize*/
    (destructor)_delete_cpp_mat<float>, /*tp_dealloc*/ // FIXME: does this break if using double?
    0, /*tp_print*/
    0, /*tp_getattr*/
    0, /*tp_setattr*/
    0, /*tp_compare*/
    0, /*tp_repr*/
    0, /*tp_as_number*/
    0, /*tp_as_sequence*/
    0, /*tp_as_mapping*/
    0, /*tp_hash */
    0, /*tp_call*/
    0, /*tp_str*/
    0, /*tp_getattro*/
    0, /*tp_setattro*/
    0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /*tp_flags*/
    "Internal deallocator object for the Matrix class", /* tp_doc */
};

static PyTypeObject VectorWrapperType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "miso_svm.VectorWrapper", /*tp_name*/
    sizeof(VectorWrapper<float>), /*tp_basicsize*/ // FIXME: does this break if using double?
    0, /*tp_itemsize*/
    (destructor)_delete_cpp_vec<float>, /*tp_dealloc*/ // FIXME: does this break if using double?
    0, /*tp_print*/
    0, /*tp_getattr*/
    0, /*tp_setattr*/
    0, /*tp_compare*/
    0, /*tp_repr*/
    0, /*tp_as_number*/
    0, /*tp_as_sequence*/
    0, /*tp_as_mapping*/
    0, /*tp_hash */
    0, /*tp_call*/
    0, /*tp_str*/
    0, /*tp_getattro*/
    0, /*tp_setattro*/
    0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /*tp_flags*/
    "Internal deallocator object for the Vector class", /* tp_doc */
};

static PyTypeObject MapWrapperType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "miso_svm.MapWrapper", /*tp_name*/
    sizeof(MapWrapper<float>), /*tp_basicsize*/ // FIXME: does this break if using double?
    0, /*tp_itemsize*/
    (destructor)_delete_cpp_map<float>, /*tp_dealloc*/ //FIXME does this break if using double?
    0, /*tp_print*/
    0, /*tp_getattr*/
    0, /*tp_setattr*/
    0, /*tp_compare*/
    0, /*tp_repr*/
    0, /*tp_as_number*/
    0, /*tp_as_sequence*/
    0, /*tp_as_mapping*/
    0, /*tp_hash */
    0, /*tp_call*/
    0, /*tp_str*/
    0, /*tp_getattro*/
    0, /*tp_setattro*/
    0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /*tp_flags*/
    "Internal deallocator object for the Map class", /* tp_doc */
};

template <typename T>
inline PyArrayObject* copyMatrix(Matrix<T>* obj) {
    std::cout << "matrix data: " << obj->rawX() << std::endl;
    int nd=2;
    std::cout << "n: " << obj->n() << " m: " << obj->m() << std::endl;
    npy_intp dims[2]={obj->n(), obj->m()};
    PyArrayObject* arr=NULL;
    arr = (PyArrayObject*)PyArray_EMPTY(nd, dims, getTypeNumber<T>(), 0);
    Matrix<T> copymat((T*)PyArray_DATA(arr), dims[1], dims[0]);
    std::cout << "numpy array data: " << PyArray_DATA(arr) << std::endl;
    if (arr == NULL) goto fail;
    copymat.copy(*obj);
    return arr;
fail:
    delete obj; // FIXME Error Handling!?
    std::cout << "FAIL in copyMatrix" << std::endl;
    Py_XDECREF(arr);
    return NULL;
}


template <typename T>
inline PyArrayObject* wrapMatrix(Matrix<T>* obj) {
    int nd=2;
    npy_intp dims[2]={obj->n(), obj->m()};
    PyObject* newobj=NULL;
    PyArrayObject* arr=NULL;
    void *mymem = (void*)(obj->rawX());
    arr = (PyArrayObject*)PyArray_SimpleNewFromData(nd, dims, getTypeNumber<T>(), mymem);

    npy_intp* strides = PyArray_STRIDES(arr);
    for (int idx=0; idx<PyArray_NDIM(arr); ++idx) {
        if (arr == NULL)
            goto fail;
    }
    newobj = (PyObject*)PyObject_New(MatrixWrapper<T>, &MatrixWrapperType);
    if (newobj == NULL) goto fail;
    ((MatrixWrapper<T> *)newobj)->obj = obj;
    PyArray_SetBaseObject((PyArrayObject*)arr, newobj);
    return arr;
   fail:
    delete obj; // FIXME Error Handling!?
    std::cout << "FAIL in wrapMatrix" << std::endl;
    Py_XDECREF(arr);
    return NULL;
}

template <typename T>
inline PyArrayObject* wrapVector(Vector<T>* obj) {
    int nd=1;
    npy_intp dims[1]={obj->n()};
    PyObject* newobj=NULL;
    void *mymem = (void*)(obj->rawX());
    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNewFromData(nd, dims, getTypeNumber<T>(), mymem);
    if (arr == NULL) goto fail;
    newobj = (PyObject*)PyObject_New(VectorWrapper<T>, &VectorWrapperType);
    if (newobj == NULL) goto fail;
    ((VectorWrapper<T> *)newobj)->obj = obj;
    PyArray_SetBaseObject((PyArrayObject*)arr, newobj);
    return arr;
fail:
    delete obj; // FIXME Error Handling!?
    Py_XDECREF(arr);
    return NULL;
}

template <typename T>
inline PyArrayObject* wrapMap(Map<T>* obj) {
    int nd=3;
    npy_intp dims[3]={obj->z(), obj->y(), obj->x()};
    PyObject* newobj=NULL;
    PyArrayObject* arr=NULL;
    void *mymem = (void*)(obj->rawX());
    arr = (PyArrayObject*)PyArray_SimpleNewFromData(nd, dims, getTypeNumber<T>(), mymem);
    if (arr == NULL) goto fail;
    newobj = (PyObject*)PyObject_New(MapWrapper<T>, &MapWrapperType);
    if (newobj == NULL) goto fail;
    ((MapWrapper<T> *)newobj)->obj = obj;
    PyArray_SetBaseObject((PyArrayObject*)arr, newobj);
    return arr;
fail:
    delete obj; // FIXME Error Handling!?
    Py_XDECREF(arr);
    return NULL;
}

template <typename T>
static int npyToMatrix(PyArrayObject* array, Matrix<T>& matrix, string obj_name) {
    if (array==NULL) {
        return 1;
    }
    if(!(PyArray_NDIM(array) == 2 &&
                PyArray_TYPE(array) == getTypeNumber<T>() &&
                (PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS))) {
        PyErr_SetString(PyExc_TypeError, (obj_name + " should be c-contiguous 2D "+getTypeName<T>()+" array").c_str());
        return 0;
    }

    T *rawX =  reinterpret_cast<T*>(PyArray_DATA(array));
    const npy_intp *shape = PyArray_DIMS(array);
    npy_intp n = shape[0];
    npy_intp m = shape[1];

    matrix.setData(rawX, m, n);
    return 1;
}

template <typename T>
static int npyToVector(PyArrayObject* array, Vector<T>& matrix, string obj_name) {
    if (array==NULL) {
        return 1;
    }
    T *rawX =  reinterpret_cast<T*>(PyArray_DATA(array));
    const npy_intp *shape = PyArray_DIMS(array);
    npy_intp n = shape[0];

    if(!(PyArray_NDIM(array) == 1 &&
                PyArray_TYPE(array) == getTypeNumber<T>() &&
                (PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS))) {
        PyErr_SetString(PyExc_TypeError, (obj_name + " should be c-contiguous 1D "+getTypeName<T>()+" array").c_str());
        return 0;
    }
    matrix.setData(rawX, n);
    return 1;
}

static vector<int> get_array_shape(PyArrayObject* array) {
    vector<int> result;
    if (array == NULL) {
        return result;
    }
    const int ndim = PyArray_NDIM(array);
    const npy_intp* shape = PyArray_DIMS(array);
    for (int i = 0; i < ndim; ++i)
        result.push_back(shape[i]);
    return result;
}

template <typename T>
static int npyToMap(PyArrayObject* array, Map<T>& matrix, string obj_name) {
    if (array==NULL) {
        return 1;
    }
    const int ndim = PyArray_NDIM(array);
    if(ndim != 3) {
        PyErr_SetString(PyExc_TypeError, (obj_name + " should have 3 dimensions but has " + to_string(ndim)).c_str());
        return 0;
    }

    if (PyArray_TYPE(array) != getTypeNumber<T>()) {
        PyErr_SetString(PyExc_TypeError, (obj_name + " has wrong data type.").c_str());
        return 0;
    }
        if (!(PyArray_FLAGS(array) & NPY_ARRAY_C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_TypeError, (obj_name + " is not contiguous.").c_str());
        return 0;
    }

    T *rawX =  reinterpret_cast<T*>(PyArray_DATA(array));
    const npy_intp *shape = PyArray_DIMS(array);
    matrix.setData(rawX, shape[2], shape[1], shape[0]);
    return 1;
}

template <typename T>
static int sequenceToVector(PyObject* seq, std::vector<T>& res) {
    if (!PySequence_Check(seq)) {
        PyErr_SetString(PyExc_TypeError, "input should be a sequence");
        return 0;
    }

    int n = PySequence_Size(seq);
    res.resize(n);

    for (int i=0; i<n; ++i) {
        PyObject* elem_i = PySequence_GetItem(seq, i);
        // FIXME this should be possible to do for nearly arbitrary types
        if (!PyLong_Check(elem_i)) {
            PyErr_SetString(PyExc_TypeError, "Expected integer elements only!");
            return 0;
        }
        res[i] = PyLong_AsLong(elem_i);
    }
    return 1;
};

template<typename T>
static PyObject* convert_primitive(T i);

template <> PyObject* convert_primitive<long>(long i){ return PyLong_FromLong(i); }
template <> PyObject* convert_primitive<int>(int i){ return PyLong_FromLong(i); }

template <typename T>
static PyObject* vector_to_pylist(vector<T> vec) {
    PyObject* result = PyList_New(vec.size());
    for (int i = 0; i < vec.size(); ++i) {
        if (PyList_SetItem(result, i, convert_primitive(vec[i])) == -1) {
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

template <typename T>
static int npy_list_to_vector(PyObject* list, vector<Matrix<T>*>& vec, string list_name) {
    const int n = PyList_Size(list);
    vec.resize(n);
    int i;
    for (i = 0; i < n; ++i) {
        PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(PyList_GetItem(list, i));
        if (arr == NULL)
            goto fail;
        Matrix<T>* mat = new Matrix<T>();
        if(npyToMatrix(arr, *mat, list_name+"["+to_string(i)+"]") == 0) {
            delete mat;
            goto fail;
        }
        vec[i] = mat;
    }

    return 1;

   fail:
    for (int j = 0; j < i; ++j) {
        delete vec[j];
    }
    return 0;
}

static PyObject* wrapMatrices(vector<Matrix<float> *> matrices) {
    PyObject* result = PyList_New(matrices.size());
    if (result == NULL)
        return NULL;
    for (int i = 0; i < matrices.size(); ++i)
        PyList_SET_ITEM(result, i, reinterpret_cast<PyObject*>(wrapMatrix(matrices[i])));
    return result;
}


inline int set_omp_threads(int threads) {
    if (threads <= 0) {
        threads=1;
#ifdef _OPENMP
        threads =  MIN(MAX_THREADS, omp_get_num_procs());
#endif
    }
    threads=init_omp(threads);
    return threads;
}

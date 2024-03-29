#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <matrix.cpp>

namespace py = pybind11;

void debugView(const float* a,
            int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << a[i * n + j] << ", ";
        }
        std::cout << ' ' << std::endl;
    }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, const size_t m, const size_t n, const size_t k,
								  const float lr, const size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format (m x n)
     *     y (const unsigned char *): pointer to y data, of size m (m,)
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format (n x k)
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int itr = m / batch;
    for (int i0 = 0; i0 < itr; i0++) {
        int start = i0 * batch;
        const float *cur_X = X + start * int(n); 
        const unsigned char *cur_y = y + start; 

        float *Z;
        Z = new float[batch * (int)k];
        matrix::mMul(cur_X, theta, Z, batch, n, k);
        matrix::mExp(Z, batch, k);
        matrix::mNormRow(Z, batch, k);

        float *Iy;
        Iy = new float[batch * (int)k]();
        for (int i = 0; i < batch; ++i) {
            Iy[i * (int)k + cur_y[i]] = 1;
        }

        float* xT;
        xT = new float[batch * (int)n];
        matrix::mT(cur_X, xT, (int)n, batch);


        float* gradient;
        gradient = new float[(int)n * (int)k];
        matrix::mSub(Z, Iy, Z, batch, k);
        matrix::mMul(xT, Z, gradient, (int)n, batch, (int)k);
        matrix::mMul(gradient, lr / batch, (int)n, (int)k);
        matrix::mSub(theta, gradient, theta, (int)n, (int)k);
        debugView(theta, n, k);
        
        delete []Z;
        delete []Iy;
        delete []xT;
        delete []gradient;
    }

    /// END YOUR CODE
    return;
}
// {
//     /**
//      * A C++ version of the softmax regression epoch code.  This should run a
//      * single epoch over the data defined by X and y (and sizes m,n,k), and
//      * modify theta in place.  Your function will probably want to allocate
//      * (and then delete) some helper arrays to store the logits and gradients.
//      *
//      * Args:
//      *     X (const float *): pointer to X data, of size m*n, stored in row
//      *          major (C) format
//      *     y (const unsigned char *): pointer to y data, of size m
//      *     theta (float *): pointer to theta data, of size n*k, stored in row
//      *          major (C) format
//      *     m (size_t): number of examples
//      *     n (size_t): input dimension
//      *     k (size_t): number of classes
//      *     lr (float): learning rate / SGD step size
//      *     batch (int): SGD minibatch size
//      *
//      * Returns:
//      *     (None)
//      */

//     /// BEGIN YOUR CODE

//     for (size_t num=0; num<m/batch; num++) {
//         size_t base = num * batch * n;
//         float *Z = new float[batch * k];
//         for (size_t i=0; i<batch; i++) {
//             for (size_t j=0; j<k; j++) {
//                 float sum = 0;
//                 // Z[i][j] = \sum X[i][x] * theta[x][j]
//                 for (size_t x=0; x<n; x++)
                    
//                     sum += X[base + i * n + x] * theta[x * k + j];
//                 Z[i * k + j] = exp(sum); // exp here
//             }
//         }

//         // normalize
//         float *Z_sum = new float[batch];
//         for (size_t i=0; i<batch; i++) {
//             float sum = 0;
//             for (size_t j=0; j<k; j++)
//                 sum += Z[i * k + j];
//             Z_sum[i] = sum;
//         }
//         for (size_t i=0; i<batch; i++)
//             for (size_t j=0; j<k; j++)
//                 Z[i*k+j] /= Z_sum[i];
//         // Z -= I
//         for (size_t i=0; i<batch; i++)
//             Z[i*k + y[num*batch + i]] -= 1.0;
//         // X.T * Z
//         for (size_t i=0; i<n; i++) {
//             for (size_t j=0; j<k; j++) {
//                 // dtheta[i][j] = \sum X.T[i][x] * Z[x][j]
//                 // X.T[i][x] = X[x][i]
//                 float sum = 0;
//                 for (size_t x=0; x<batch; x++)
//                     sum += X[base+x*n+i] * Z[x*k+j];
//                 theta[i*k+j] -= lr / batch * sum;
//             }
//         }
//     }
//     /// END YOUR CODE
// }


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}

#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

namespace matrix {
    void mT(const float *a, float *res,
            int mRes, int nRes) {
        for (int i = 0; i < mRes; ++i) {
            for (int j = 0; j < nRes; ++j) {
                res[i * nRes + j] = a[j * mRes + i]; 
            }
        }
    }

    void mSum(const float *a, const float *b, float *res,
            int m, int n) {
        int itr = m * n;
        for (int i = 0; i < itr; ++i) {
            res[i] = a[i] + b[i];
        }
    }

    void mSub(const float *a, const float *b, float *res,
            int m, int n) {
        int itr = m * n;
        for (int i = 0; i < itr; ++i) {
            res[i] = a[i] - b[i];
        }
    }

    // 2x3 3x4
    void _rowMul(const float *a, const float *b, float *res, 
                int rowLoc, int n, int k) {
        float *tar = res + rowLoc * k;
        for (int i = 0; i < k; ++i) {
            float cur = 0;
            for (int j = 0; j < n; ++j) {
                // std::cout << rowLoc * n + j << " " << i + j * k << std::endl;
                cur += a[rowLoc * n + j] * b[i + j * k];
            }
            *tar = cur;
            ++tar;
        }
        return;
    }

    void mMul(const float *a, const float *b, float *res,
                int m, int n, int k) {
        for (int i = 0; i < m; ++i) {
            _rowMul(a, b, res, i, n, k);
        }
        return;
    }

    void mExp(float *res,
            int m, int n) {
        int itr = m * n;
        for (int i = 0; i < itr; ++i) {
            res[i] = exp(res[i]);
        }
    }

    void mDiv(float *a, const float b,
                int m, int n) {
        int itr = m * n;
        for (int i = 0; i < itr; ++i) {
            a[i] /= b;
        }
    }

    void mMul(float *a, const float b,
                int m, int n) {
        int itr = m * n;
        for (int i = 0; i < itr; ++i) {
            a[i] *= b;
        }
    }

    void mNormRow(float *a,
                int m, int n) {
        for (int i = 0; i < m; ++i) {
            float rowSum = 0;
            for (int j = 0; j < n; ++j) {
                rowSum += a[i * n + j];
            }
            for (int j = 0; j < n; ++j) {
                a[i * n + j] /= rowSum;
            }
        }
    }
}
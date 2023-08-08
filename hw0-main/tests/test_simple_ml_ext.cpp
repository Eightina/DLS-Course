#pragma once
#include <matrix.cpp>
#include <simple_ml_ext.cpp>

namespace py = pybind11;

int main() {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float res[2 * 4] = {};
    matrix::mMul(a, b, res, 2, 3, 4);
    matrix::mExp(b, 3, 4);
    matrix::mNormRow(a, 2, 3);
    matrix::mDiv(a, 10.0f, 2, 3);

//x: 9 x 4
    float X[] = {1.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f,
                2.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f,
                3.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f,};
//y: 9
    unsigned char y[] = {1, 2, 1, 2, 1, 2,
                        1, 2, 0};
//theta: 4 x 3
    float *theta;
    theta = new float[12]();
    softmax_regression_epoch_cpp(
        X, y, theta,
        9, 4, 3, 0.1, 3
    );

    return 0;
}
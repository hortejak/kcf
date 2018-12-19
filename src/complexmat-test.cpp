#include "complexmat.hpp"
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    ComplexMat_<32, 32, 44, 1> a(32, 32, 44, 1), b(32, 32, 44, 1), c(32, 32, 44, 1);
    ComplexMat_<32, 32, 1, 1> x(32, 32, 1, 1), y(32, 32, 1, 1);

    a = b + c;

    b = a.mul(y);

    cout << a << endl;

    return 0;
}

#include <iostream>
#include "crun.h"

int MyClass::Add(int a, int b) {
    return a + b;
}

/**
 * mkdir build
 * cd build
 * cmake ..
 * make
 * ./crun_test
 */
#ifndef RUN_TESTS
int main(int argc, char **argv) {
    std::cout << "Hello World" << std::endl;
}
#endif
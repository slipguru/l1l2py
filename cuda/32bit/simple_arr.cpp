#include <iostream>

using namespace std;

extern "C" {

    int simple_arr(float * a, int n, int p, float * res) {
        
        int j;
        *res = 0;
        
        for (j = 0; j < n*p; j++) {
            *res += a[j];
        }
        
        return 0;
    }

}
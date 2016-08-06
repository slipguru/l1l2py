#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <string.h> 
 
 
//#include "split.h"

using namespace std;

void read_csv(string file_name, int n, int p, float * v) {
    
    ifstream in_file;
    string line;
    char * pch;
    char * cstr;
    
    in_file.open(file_name.c_str());
    
    //### This index increase once for every element of the matrix, which is represented as a vector
    int ij = 0;
    
    for (int i = 0; i < n; i++) {
        getline(in_file, line);
        
        cstr = new char [line.length() + 1];
        
        strcpy(cstr, line.c_str());
        
        pch = strtok (cstr, " ");
        
        while (pch != NULL)
        {
            v[ij] = (float)atof(pch);
            pch = strtok (NULL, " ");
            ij++;
        }
    }
}
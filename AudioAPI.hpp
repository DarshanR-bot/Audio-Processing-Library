#ifndef AudioAPI_h__
#define AudioAPI_h__


using namespace std;

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <mkl/mkl.h>


typedef struct{
    int label;
    float prob;
}pred_t;

extern pred_t* libaudioAPI(const char* audiofeatures, pred_t* pred);


#endif
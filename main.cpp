
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>
#include "AudioAPI.hpp"

using namespace std;

int main(int argc, const char *argv[])
{
    string labels[]={"silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"};
    pred_t pred[3];
    if(argc!=3){
        cout<<"Invalid arguments";
        return 0;
    }
    
    libaudioAPI(argv[1], pred);
    ofstream o(argv[2]);
    o<<argv[1]<<" ";
    o<<labels[pred[0].label]<<" ";
    o<<labels[pred[1].label]<<" ";
    o<<labels[pred[2].label]<<" ";
    o<<pred[0].prob<<" ";
    o<<pred[1].prob<<" ";
    o<<pred[2].prob<<" ";
    
    return 0;
}
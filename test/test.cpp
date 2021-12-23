
#include "include/attention.h"
#include "utils/helper.h"
#include <iostream>
#include "Python.h"
#include <iostream>
/*
#define QL 1024
#define KL 1024
#define HL 512
*/


void test_full_attn(int QL, int KL, int HL)
{
    float* Q = new float[QL*HL];
    float* K = new float[KL*HL];
    float* V = new float[KL*HL];

    float* res = new float[QL*HL];
#ifdef TIME
    Clk timer("AttentionFull");
#endif

    AttentionFull(Q, K, V, QL, KL, HL, res);

#ifdef TIME
    Clk::printDuration(timer);
#endif

/*TODO: check result.*/

    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] res;

}

void test_local_attn(int QL, int KL, int HL, int local_w, int local_h)
{
    float* Q = new float[QL*HL];
    float* K = new float[KL*HL];
    float* V = new float[KL*HL];
    int local_width = local_w;
    int local_height = local_h;
    float* res = new float[QL*HL];

#ifdef TIME
    Clk timer("AttentionGlobal");
#endif

    AttentionGlobal(Q, K, V, QL, KL, HL, local_width, local_height, res);

#ifdef TIME
    Clk::printDuration(timer);
#endif

/* TODO: check result.*/

    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] res;
}


int main(int argc, char **argv)
{

    int Batch, Head;
    int QL, KL, HL;
    int local_w, local_h;
    int window_size, window_stride;

    QL = 2048;
    KL = 2048;
    HL = 512;
    local_h = 4;
    local_w = 4;

    test_full_attn(QL, KL, HL);
    test_local_attn(QL, KL, HL, local_w, local_h);

    return 0;
}
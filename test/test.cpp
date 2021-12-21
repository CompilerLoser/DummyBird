
#include "include/attention.h"

#include <iostream>

#define QL 1024
#define KL 1024
#define HL 512



void test_full_attn()
{
    float* Q = new float[QL*HL];
    float* K = new float[KL*HL];
    float* V = new float[KL*HL];

    float* res = new float[QL*HL];
#ifdef TIME
/*timing start.*/
#endif
    AttentionFull(Q, K, V, QL, KL, HL, res);
#ifdef TIME
/*timing end.*/
#endif
/*TODO: check result.*/
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] res;

}


int main()
{
    test_full_attn();
    return 0;
}
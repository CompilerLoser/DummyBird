
#include "include/attention.h"
#include "include/composed_attention.h"
#include "utils/helper.h"
#include <iostream>
//#include "Python.h"
/*
#define QL 1024
#define KL 1024
#define HL 512
*/

void test_full_attn(int QL, int KL, int HL)
{
    float *Q = new float[QL * HL];
    float *K = new float[KL * HL];
    float *V = new float[KL * HL];

    float *res = new float[QL * HL];
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
    float *Q = new float[QL * HL];
    float *K = new float[KL * HL];
    float *V = new float[KL * HL];
    float *res = new float[QL * HL];

#ifdef TIME
    Clk timer("AttentionGlobal");
#endif

    AttentionGlobal(Q, K, V, QL, KL, HL, local_w, local_h, res);

#ifdef TIME
    Clk::printDuration(timer);
#endif

    /* TODO: check result.*/

    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] res;
}
void test_random(int QL, int KL, int HL,
                 int random_size)
{
    float *Q = new float[QL * HL];
    float *K = new float[KL * HL];
    float *V = new float[KL * HL];
    float *res = new float[QL * HL];

#ifdef TIME
    Clk timer("Random");
#endif
    AttentionRandom(Q, K, V, QL, KL, HL, random_size, res);
#ifdef TIME
    Clk::printDuration(timer);
#endif

    /* TODO: check result.*/

    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] res;
}
void test_composed(int QL, int KL, int HL,
                   int local_w, int local_h, int random_size,
                   int window_size, int window_stride, int window_height)
{
    float *Q = new float[QL * HL];
    float *K = new float[KL * HL];
    float *V = new float[KL * HL];
    float *res = new float[QL * HL];

#ifdef TIME
    Clk timer("Composed");
#endif

    AttentionComposed(Q, K, V, QL, KL, HL, local_w, local_h, window_size, window_height, window_stride, random_size, res);
#ifdef TIME
    Clk::printDuration(timer);
#endif

    /* TODO: check result.*/

    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] res;
}

void test_batched_multi_head_attention_random(int QW, int KW, int VW, int QL, int HL, int KL,
                                              int random_size, int BatchSize, int Head)
{
    for (int i = 0; i < BatchSize; ++i)
    {
        float *QS = new float[QL * QW];
        float *KS = new float[KL * KW];
        float *VS = new float[KL * KW];
        float *res = new float[Head * QL * HL];
#ifdef TIME
        Clk timer("MultiHeadAttnRand");
#endif
        MultiHeadAttentionRandom(QS, KS, VS, QL, QW, KL, KW, HL, Head, random_size, res);

#ifdef TIME
        Clk::printDuration(timer);
#endif
        delete[] res;
        delete[] QS;
        delete[] KS;
        delete[] VS;
    }
}

int main(int argc, char **argv)
{

    int Batch, Head;
    int QL, KL, HL;
    int local_w, local_h;
    int window_size, window_stride, window_height;
    int random_size;

    QL = 2048;
    KL = 2048;
    HL = 512;
    local_h = 64;
    local_w = 64;
    window_stride = 4;
    window_size = 64;
    window_height = 16;
    random_size = 16;
    Batch = 1;
    Head = 4;

    test_random(QL, KL, HL, random_size);
    // test_full_attn(QL, KL, HL);
    test_local_attn(QL, KL, HL, local_w, local_h);
    // test_composed(QL, KL, HL, local_h, local_w, random_size, window_size, window_stride, window_height);
    test_batched_multi_head_attention_random(512, 512, 512, QL, HL, KL, random_size, Batch, Head);

    return 0;
}
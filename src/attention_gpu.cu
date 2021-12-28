#include "include/attention_gpu.h"

/**
 * perpare Q/K/V sequences and multi-head project matrixs and call cuda kernel 
 * to compute a batch of multi-head random attentions and write result to res.
 * 
 * @param QS/KS/VS: [BS * QL/KL/VL * QW/KW/VW] Input sequences 
 * @param WQ/WK/WV: [Heads * QW/KW/VW * HL] Project Matrixs
 * @param QL, KL, VL, QW, KW, VW, HL Sizes
 * @param BS, Heads
 * @param random_size 
 * @param res:[BS * Heads * QL * HL]
 * */
void BatchedMultiHeadAttentionRandomCUDA(const float *Q, const float *K, const float *V,
                     int QL, int KL, int HL,
                     int random_size,
                     float *res)
{
    return ;
}
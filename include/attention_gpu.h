#include <cuda.h>

void BatchedMultiHeadAttentionRandomCUDA(const float *Q, const float *K, const float *V,
                     int QL, int KL, int HL,
                     int random_size,
                     float *res);
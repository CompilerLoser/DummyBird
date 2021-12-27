#ifndef COMPOSED_ATTENTION_H
#define COMPOSED_ATTENTION_H
void AttentionComposed(const float *Q, const float *K, const float *V, int QL, int KL, int HL,
                       int local_width, int local_height,
                       int window_size, int window_height, int window_stride,
                       int random_size, float *res);


#endif
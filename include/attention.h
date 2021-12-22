#ifndef ATTENTION_H
#define ATTENTION_H

void attention();

void MultiHeadAttentionFull(const float *Q, const float *K, const float *V,
                            int QL, int QW, int KL, int KW, int HL, int Head,
                            float* res);

void AttentionFull(const float* Q, const float* K, const float* V, 
                    int QL, int KL, int HL, 
                    float* res);

void AttentionGlobal(const float* Q, const float* K, const float* V, 
                    int QL, int KL, int HL, 
                    int local_width, int local_height,
                    float* res);
                    
void AttentionWindow(const float* Q, const float* K, const float* V, 
                    int M, int N, int L, 
                    int Qs, int Ks, int Vs, 
                    int window_blocks, float* res);

void AttentionRandom(const float* Q, const float* K, const float* V, 
                    int M, int N, int L, 
                    int Qs, int Ks, int Vs, 
                    int rand_blocks, float* res);

#endif 
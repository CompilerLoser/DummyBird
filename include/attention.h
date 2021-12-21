#ifndef ATTENTION_H
#define ATTENTION_H

void attention();

void AttentionFull(const float* Q, const float* K, const float* V, 
                    int QL, int KL, int HL, 
                    float* res);

void AttentionLocal(const float* Q, const float* K, const float* V, 
                    int M, int N, int L, 
                    int Qs, int Ks, int Vs, 
                    int local_blocks, float* res);
                    
void AttentionWindow(const float* Q, const float* K, const float* V, 
                    int M, int N, int L, 
                    int Qs, int Ks, int Vs, 
                    int window_blocks, float* res);

void AttentionRandom(const float* Q, const float* K, const float* V, 
                    int M, int N, int L, 
                    int Qs, int Ks, int Vs, 
                    int rand_blocks, float* res);

#endif 
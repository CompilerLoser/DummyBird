#ifndef ATTENTION_H
#define ATTENTION_H

void attention();

template <typename T>
void AttentionFull(const T* Q, const T* K, const T* V, 
                    int QL, int KL, int HL, 
                    T* res);

template <typename T>
void AttentionLocal(const T* Q, const T* K, const T* V, 
                    int M, int N, int L, 
                    int Qs, int Ks, int Vs, 
                    int local_blocks, T* res);
                    
template <typename T>
void AttentionWindow(const T* Q, const T* K, const T* V, 
                    int M, int N, int L, 
                    int Qs, int Ks, int Vs, 
                    int window_blocks, T* res);

template <typename T>
void AttentionRandom(const T* Q, const T* K, const T* V, 
                    int M, int N, int L, 
                    int Qs, int Ks, int Vs, 
                    int rand_blocks, T* res);

#endif 
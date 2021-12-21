#include "include/attention.h"
#include "utils/helper.h"
#include <cmath>
#include <memory.h>

void attention()
{
    foo();
}

/**
 * @brief Given the projected matrixs Q, K, V, compute the full attn scores.
 * add all scores to res.
 * Serialize all SCCs, min fusion.
 * @param Q QL*HL
 * @param K KL*HL
 * @param V KL*HL
 * @param QL length of queries
 * @param KL length of keys
 * @param HL hidden_size
 * @param res QL*HL
 */
template <typename T>
void AttentionFull(const T *Q, const T *K, const T *V,
                   int QL, int KL, int HL,
                   T *res)
{
    T *temp = new T[QL * KL];
    T *line_exp_sum = new T[QL];
    memset(temp, 0, QL * KL * sizeof(T));
    memset(line_exp_sum, 0, QL * sizeof(T));

    double dk_inv = 1.0 / sqrt(HL);

    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < KL; ++j)
            for (int l = 0; l < HL; ++l)
                /*  temp[i][j] = reduce_sum(Q[i][l] * Kt[l][j])
                 *             = reduce_sum(Q[i][l] * K[j][l])   */
                temp[i * KL + j] += Q[i * HL + l] * K[j * HL + l];

    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < QL; ++j)
            temp[i * KL + j] *= dk_inv;

    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < KL; ++j)
            line_exp_sum[i] += exp(temp[i * KL + j]);

    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < KL; ++j)
            temp[i * KL + j] /= line_exp_sum[i];

    for (int i = 0; i < QL; ++i)
        for (int h = 0; h < HL; ++h)
            for (int l = 0; l < KL; ++l)
                /* res[i][h] += temp[i][l] * V[l][h] */
                res[i * HL + h] += temp[i * KL + l] * V[l * HL + h];

    delete[] temp;
    delete[] line_exp_sum;
}
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
 * 
 * @param Q QL*HL
 * @param K KL*HL
 * @param V KL*HL
 * @param QL length of queries
 * @param KL length of keys
 * @param HL hidden_size
 * @param res QL*HL
 */
void AttentionFull(const float *Q, const float *K, const float *V,
                   int QL, int KL, int HL,
                   float *res)
{
    float *temp = new float[QL * KL];
    float *line_exp_sum = new float[QL];
    memset(temp, 0, QL * KL * sizeof(float));
    memset(line_exp_sum, 0, QL * sizeof(float));

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

/**
 * @brief 
 *  
 * @param Q QL*QW -> before projected
 * @param K KL*QK 
 * @param V KL*QK
 * @param QL Q sequence sizes 
 * @param QW Q sequence length 
 * @param KL  
 * @param KW 
 * @param HL projected out dims 
 * @param Head 
 * X@param Batch  
 * @param res 
 */

void MultiHeadAttentionFull(const float *Q, const float *K, const float *V,
                            int QL, int QW, int KL, int KW, int HL, int Head,
                            float* res)
{
    /*
    new WQ[QW*HL*Head] WK[KW*HL*Head] WV[KW*HL*Head]
    for each Heads
        Q = Q @ &WQ[Head*QW*HL]
        ...
        AttentionFull(Q, K, V, QL, KL, HL, &res[Head*QL*HL])
    */
    return;
}

/**
 * @brief 
 * compute global attention patterns 
 * @param local_width local_size * K_blocksize in BigBird paper 
 * @param local_height local_size * Q_blocksize in BigBird paper
 */
void AttentionGlobal(const float* Q, const float* K, const float* V, 
                    int QL, int KL, int HL, 
                    int local_width, int local_height,
                    float* res)
{
    float* temp_top = new float[local_height * KL];
    float* temp_left = new float[(QL - local_width) * local_width];

    /* compute top 
    
}

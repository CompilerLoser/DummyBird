/**
 * This file implement the SINGLE basic attention patterns in BigBird.
 * NOTE when compose different patterns in a attention, the scores matrixs
 * should be merged and then normalized.
 */

#include "include/attention.h"
#include "utils/helper.h"
#include <cmath>
#include <memory.h>

void attention()
{
    foo();
}

#define max(x, y)  x > y ? x : y
#define min(x, y)  x < y ? x : y

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
            temp[i * KL + j] = exp(temp[i * KL + j]) / line_exp_sum[i];

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
                            float *res)
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
 * @param local_width local_size(number of blocks) * K_blocksize in BigBird paper
 * @param local_height local_size * Q_blocksize in BigBird paper
 */
void AttentionGlobal(const float *Q, const float *K, const float *V,
                     int QL, int KL, int HL,
                     int local_width, int local_height,
                     float *res)
{
    float *temp_top = new float[local_height * KL];
    float *temp_left = new float[(QL - local_height) * local_width];
    float *line_exp_sum = new float[QL];
    memset(temp_top, 0, local_height * KL * sizeof(float));
    memset(temp_left, 0, (QL - local_height) * local_width * sizeof(float));
    memset(line_exp_sum, 0, QL * sizeof(float));

    double dk_inv = 1.0 / sqrt(HL);

    /* compute top *local_height* rows scores */
    for (int i = 0; i < local_height; ++i)
        for (int j = 0; j < KL; ++j)
            for (int l = 0; l < HL; ++l)
                /* temp[i][j] = reduce_sum(Q[i][l] * K[j][l]) */
                temp_top[i * KL + j] += Q[i * HL + l] * K[j * HL + l];

    /* compute left *local_width* cols but remove top *local_height* rows. */
    for (int i = local_height; i < QL; ++i)
        for (int j = 0; j < local_width; ++j)
            for (int l = 0; l < HL; ++l)
                temp_left[(i - local_height) * local_width + j] +=
                    Q[i * HL + l] * K[j * HL + l];

    /* compute ∑ef(i) */
    for (int i = 0; i < local_height; ++i)
        for (int j = 0; j < KL; ++j)
        {
            temp_top[i * KL + j] *= dk_inv;
            line_exp_sum[i] += exp(temp_top[i * KL + j]);
        }

    for (int i = local_height; i < QL; ++i)
        for (int j = 0; j < local_width; ++j)
        {
            temp_left[(i - local_height) * local_width + j] *= dk_inv;
            line_exp_sum[i] += exp(temp_left[(i - local_height) * local_width + j]);
        }
    /* softmax */
    for (int i = 0; i < local_height; ++i)
        for (int j = 0; j < KL; ++j)
            temp_top[i * KL + j] = exp(temp_top[i * KL + j]) / line_exp_sum[i];

    for (int i = local_height; i < QL; ++i)
        for (int j = 0; j < local_width; ++j)
            temp_left[(i - local_height) * local_width + j] = /*the Σei should contain all e0 = 1, that is (KL - local_width)*/
                exp(temp_left[(i - local_height) * local_width + j]) / (line_exp_sum[i] + KL - local_width); /* place here avoiding imperfect loop nest.

    /* global score @ V */
    /* part 1 */
    for (int i = 0; i < local_height; ++i)
        for (int j = 0; j < HL; ++j)
            for (int l = 0; l < KL; ++l)
                res[i * HL + j] += temp_top[i * KL + l] * V[l * HL + j];
    /*part 2 */
    for (int i = local_height; i < QL; ++i)
        for (int j = 0; j < HL; ++j)
            for (int l = 0; l < local_width; ++l)
                res[i * HL + j] += temp_left[(i - local_height) * local_width + l] * V[l * HL + j];
}

/**
 * @brief Simple window attention in BigBird.
 * The most naive way: compute scores row by row.
 * @param window_len Window length in a row
 * @param window_height How many rows passed when the window are going to stride.
 * @param window_stride Window stride length
 */
void AttentionWindow(const float *Q, const float *K, const float *V,
                     int QL, int KL, int HL,
                     int window_size, int window_height, int window_stride,
                     float *res)
{
    float *temp = new float[QL * window_size];
    float *line_exp_sum = new float[QL];
    memset(temp, 0, QL * window_size * sizeof(float));
    memset(line_exp_sum, 0, QL * sizeof(float));

    // for now we assume the first line window start from 0.
    int first_line_padding = 0;

    for (int i = 0; i < QL; ++i)
    {
        int window_start = first_line_padding + int(i / window_height) * window_stride; 
        int window_end = window_start + window_size;
        window_start = max(window_start, 0);
        window_end = min(window_end, KL);

        for (int j = window_start; j < window_end; ++j)
            for (int l = 0; l < HL; ++l)
                temp[i * window_size + j - window_start] += Q[i * QL + l] * K[j * KL + l];
        
        /* softmax */
        for(int i = 0; i<QL; ++i)
            for(int j =0; j<window_size; ++j)
                line_exp_sum[i] += exp(temp[i*window_size+j]);
        for(int i=0; i<)

        
         
    }

    return;
}

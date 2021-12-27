#include "utils/helper.h"
#include "include/attention.h"
#include <cmath>
#include <memory.h>

static bool in_rand_cols(int col, int *rand_line, int random_size)
{
    for (int i = 0; i < random_size; ++i)
        if (col == rand_line[i])
            return true;
    return false;
}

static void compute_rand_cols(int length, int random_size, int local_width, int start_padding,
                              int window_size, int KL, const int *window_index, int *rand_cols)
{
    for (int i = 0; i < length; ++i)
        for (int j = 0; j < random_size; ++j)
        {
            rand_cols[j] = random(KL);
            while (
                rand_cols[j] < local_width ||
                (rand_cols[j] < min(window_index[i] + window_size,KL)) && (rand_cols[j] > window_index[i]) ||
                in_rand_cols(rand_cols[j], &rand_cols[i * random_size], random_size))
            {
                rand_cols[j] = random(KL);
            }
        }
}

/**
 * Dummy code of the implementation detials designed in BigBird Appendix.
 * TODO: write the simplified compute pattern notes. 
 * TODO: debug and check result.
 */
void AttentionComposed(const float *Q, const float *K, const float *V, int QL, int KL, int HL,
                       int local_width, int local_height,
                       int window_size, int window_height, int window_stride,
                       int random_size, float *res)
{

    int *rand_cols = new int[(QL - local_height) * (random_size)];
    int *window_index = new int[QL - local_height];
    float *scores_localtop = new float[local_height * KL];
    float *scores = new float[(QL - local_height) * (window_size + random_size + local_width)];
    float *exp_sum = new float[QL];
    memset(scores_localtop, 0, local_height * KL * sizeof(float));
    memset(scores, 0, (QL - local_height) * (window_size + random_size + local_width) * sizeof(float));
    memset(exp_sum, 0, QL * sizeof(float));

    // compute rand_cols
    int start_padding = 0; // :(
    for (int i = 0; i < QL - local_height; ++i)
        window_index[i] = max(0, start_padding + int(i / window_height) * window_stride);
    compute_rand_cols(QL - local_height, random_size, local_width, start_padding, window_size, KL, window_index, rand_cols);

    // compute scores
    /* top local_height rows of score matrix, compute as S[i, j] = Q[i,:] @ KT[:,j], i<local_height. */
    for (int i = 0; i < local_height; ++i)
        for (int j = 0; j < KL; ++j)
            for (int l = 0; l < HL; ++l)
                scores_localtop[i * KL + j] += Q[i * HL + l] * K[i * HL + l];
    /* other scores, depend on the paddings, some score value in window score part may be 0. */
    int score_len = window_size + random_size + local_width;
    for (int i = 0; i < QL - local_height; ++i)
    {
        int idx = i + local_height;
        for (int j = 0; j < local_width; ++j)
            for (int l = 0; l < HL; ++l)
                scores[i * score_len + j] += Q[idx * HL + l] * K[j * HL + l]; /* S[i,j] = Q[i,:] @ KT[:,j], j<local_width, i>local_height */
        for (int j = max(local_width,local_width-window_index[i]); j < j + min(window_size, KL-window_index[i]); ++j) //do not compute padding scores
            for (int l = 0; l < HL; ++l)
                scores[i * score_len + j] += Q[idx * HL + l] * K[(window_index[i] + j - local_width) * HL + l]; /* S[i,j] = Q[i,:] @ KT[:, window_index[i]]*/
        for (int j = local_width + window_size; j < score_len; ++j)
            for (int l = 0; l < HL; ++l)
                scores[i * score_len + j] += Q[idx * HL + l] * K[(rand_cols[i * random_size + j - local_width - window_size]) * HL + l];/* S[i,j] = Q[i,:] @ KT[:,rand_cols[i]]*/
    }

    // softmax
    double dk_inv = 1.0 / sqrt(HL);
    for (int i = 0; i < local_height; ++i)
        for (int j = 0; j < KL; ++j)
            exp_sum[i] += exp(scores_localtop[i * KL + j]);
    for (int i = 0; i < QL - local_height; ++i)
    {
        for (int j = 0; j < score_len; ++j)
            exp_sum[i + local_height] += exp(scores[i * score_len + j]);
    }
    for (int i = 0; i < local_height; ++i)
        for (int j = 0; j < KL; ++j)
            scores_localtop[i * KL + j] = exp(scores_localtop[i * KL + j] / dk_inv) / exp_sum[i];
    for (int i = 0; i < QL - local_height; ++i)
    {
        for (int j = 0; j < score_len; ++j)
            scores[i * score_len + j] = exp(scores[i * score_len + j] / dk_inv) / exp_sum[i + local_height];
    }

    //compute res
    for(int i =0; i<local_height; ++i)
        for(int j =0; j< HL; ++j)
            for(int l = 0; l<KL; ++l)
                res[i*HL +j] += scores_localtop[i*KL+l] * V[l*HL + j];

    for(int i =local_height; i<QL; ++i)
        for(int j=0; j<HL; ++j){
            for(int l=0; l<local_width; ++l)
                res[i*HL+j] += scores[(i-local_height)*score_len+l] * V[l*HL +j];
            for(int l = local_width; l<local_width+window_size; ++l)
                res[i*HL +j] += scores[(i-local_height)*score_len+l] * V[(window_index[i-local_height]+l-local_width)*HL + j];
            for(int l = local_width+window_size; l<score_len; ++l)
                res[i*HL +j] += scores[(i-local_height)*score_len+l] * V[(rand_cols[l-local_width-window_size])*HL +j];
        }

    delete[] rand_cols;
    delete[] window_index;
    delete[] scores_localtop;
    delete[] scores;
    delete[] exp_sum; 
}
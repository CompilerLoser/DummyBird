/**
 * This file implement the SINGLE pattern attention functions come from BigBird.
 */
#include "include/attention.h"
#include "utils/helper.h"

void attention()
{
    foo();
}

/**
 * Given the projected matrixs Q, K, V, compute the full attn scores.
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
 * compute global attention patterns
 * TODO: Support any sequences pair that is symmetric along the diagonal
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
                temp_left[(i - local_height) * local_width + j] += Q[i * HL + l] * K[j * HL + l];

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
                exp(temp_left[(i - local_height) * local_width + j]) /
                (line_exp_sum[i] + KL - local_width); /* No LICM, avoiding create imperfect loop nest.

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

    delete[] temp_top;
    delete[] temp_left;
    delete[] line_exp_sum;
}

/**
 * Simple window attention in BigBird.
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

    double dk_inv = 1.0 / sqrt(HL);

    for (int i = 0; i < QL; ++i)
    {
        int window_start = first_line_padding + int(i / window_height) * window_stride;
        int window_end = window_start + window_size;
        window_start = max(window_start, 0);
        window_end = min(window_end, KL);

        for (int j = window_start; j < window_end; ++j)
            for (int l = 0; l < HL; ++l)
                temp[i * window_size + j - window_start] += Q[i * HL + l] * K[j * HL + l];
    }
    /* softmax */
    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < window_size; ++j)
            line_exp_sum[i] += exp(temp[i * window_size + j]);
    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < window_size; ++j)
            temp[i * window_size + j] = exp(temp[i * window_size + j] * dk_inv) /
                                            line_exp_sum[i] +
                                        (KL - window_size);
    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < HL; ++j)
            for (int l = 0; l < window_size; ++l)
            {
                int col_start = first_line_padding + int(i / window_height) * window_stride;
                res[i * HL + j] += temp[i * window_size + l] * V[max(col_start, 0) * l + j];
            }
    delete[] temp;
    delete[] line_exp_sum;
}

/**
 * For now, assume that random attention can exist between any two tokens.
 * @param random_size the attention number of a query token.
 */
void AttentionRandom(const float *Q, const float *K, const float *V,
                     int QL, int KL, int HL,
                     int random_size,
                     float *res)
{
    int *random_key_pos = new int[QL * random_size];
    float *scores = new float[QL * random_size];
    float *line_exp_sum = new float[QL];
    memset(line_exp_sum, 0, QL * sizeof(float));
    memset(scores, 0, QL * random_size * sizeof(float));

    double dk_inv = 1.0 / sqrt(HL);
    // generate a random matrix used to specify the token attend for each row
    /*
    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < random_size; ++j)
            random_key_pos[i * random_size + j] = random(KL); // may be duplicate
            */
    generate_random_cols(KL, QL, random_size, conflict, random_key_pos);
    // compute scores as normal
    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < random_size; ++j)
            for (int l = 0; l < HL; ++l)
            {
                int col_pos = random_key_pos[i * random_size + j];
                scores[i * random_size + j] += Q[i * HL + l] * K[col_pos * HL + l];
            }
    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < random_size; ++j)
            line_exp_sum[i] += exp(random_key_pos[i * random_size + j]);
    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < random_size; ++j)
            scores[i * random_size + j] = exp(scores[i * random_size + j] * dk_inv) /
                                          (line_exp_sum[i] + KL - random_size);

    // compute attention result
    for (int i = 0; i < QL; ++i)
        for (int j = 0; j < HL; ++j)
            for (int l = 0; l < random_size; ++l)
            {
                int row = random_key_pos[i * random_size + l];
                res[i * HL + j] += scores[i * random_size + l] * V[row * HL + j];
            }
    delete[] random_key_pos;
    delete[] scores;
    delete[] line_exp_sum;
}

static void project(const float *Mat, const float *WMat, int M, int N, int L, float *res)
{
    memset(res, 0, M * L * sizeof(float));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < L; ++j)
            for (int l = 0; l < N; ++l)
                res[i * M + j] += Mat[i * M + l] * WMat[l * N + j];
}

void MultiHeadAttentionRandom(const float *QS, const float *KS, const float *VS,
                              int QL, int QW, int KL, int KW, int HL,
                              int Head, int random_size,
                              float *res)
{

    float *WQ = new float[Head * QW * HL];
    float *WK = new float[Head * KW * HL];
    float *WV = new float[Head * KW * HL];

    float *Q = new float[Head * QL * HL];
    float *K = new float[Head * KL * HL];
    float *V = new float[Head * KL * HL];

    memset(res, 0, Head * QL * HL * sizeof(float));

    rand_init_matrix(100, QW, HL * Head, WQ);
    rand_init_matrix(100, KW, HL * Head, WK);
    rand_init_matrix(100, KW, HL * Head, WV);

    for (int i = 0; i < Head; ++i)
    {
        project(QS, &WQ[i * QW * HL], QL, QW, HL, &Q[i*QL*HL]);
        project(KS, &WK[i * KW * HL], KL, KW, HL, &K[i* KL *HL]);
        project(VS, &WV[i * KW * HL], KL, KW, HL, &V[i *KL *HL]);
    }

    for(int i =0; i < Head; ++i)
    {
        AttentionRandom(&Q[i*QL*HL], &K[i* KL *HL], &V[i*KL*HL], QL, KL, HL, random_size, &res[i*QL*HL]);
    }

    delete[] WQ;
    delete[] WK;
    delete[] WV;
    delete[] Q;
    delete[] K;
    delete[] V;
}
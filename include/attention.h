#ifndef ATTENTION_H
#define ATTENTION_H

#include <string>

class AttnPattern
{
public:
    AttnPattern();

private:
    enum ptn {FULL, GLOBAL, WINDOW, RANDOM, COMPOSED};
};

class AttnConfig
{
public:
    AttnConfig(int QLen, int KLen, int HLen, AttnPattern &pattern) : \
    QL(QLen = 2048), KL(KLen = 2048), HL(HLen = 512), pattern(pattern) {
        initQKV(QL, KL, HL);
    }
    ~AttnConfig() = default;

    std::string device;
private:
    int QL;
    int KL;
    int HL;
    float *Q;
    float *K;
    float *V;
    void initQKV(int QL, int KL, int HL);

    AttnPattern &pattern;
};

class Attention
{
public:
    Attention(AttnConfig& config);
    ~Attention() = default;
    void run();
    virtual void runFull();
    virtual void runGlobal();
    virtual void runWindow();
    virtual void runRandom();
    virtual void runComposed();
protected:
    AttnConfig& conf;
};

class DummyAttention : public Attention
{
public:
    void runFull() final;
};


void MultiHeadAttentionFull(const float *Q, const float *K, const float *V,
                            int QL, int QW, int KL, int KW, int HL, int Head,
                            float *res);

void AttentionFull(const float *Q, const float *K, const float *V,
                   int QL, int KL, int HL,
                   float *res);

void AttentionGlobal(const float *Q, const float *K, const float *V,
                     int QL, int KL, int HL,
                     int local_width, int local_height,
                     float *res);

void AttentionWindow(const float *Q, const float *K, const float *V,
                     int QL, int KL, int HL,
                     int window_size, int window_height, int window_stride,
                     float *res);

void AttentionRandom(const float *Q, const float *K, const float *V,
                     int M, int N, int L,
                     int Qs, int Ks, int Vs,
                     int rand_blocks, float *res);

#endif
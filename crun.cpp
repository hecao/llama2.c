#ifndef CRUN_CPP
#define CRUN_CPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>

using namespace std;

class Config {
public: 
    Config() {}

    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length

    int loadFromFile(const string& filename) {
        FILE *file = fopen(filename.c_str(), "rb");
        if (!file) {
            cerr << "Failed to open file: " << filename << endl;
        }

        if (!fread(&dim, sizeof(dim), 1, file)) {return -1;};
        if (!fread(&hidden_dim, sizeof(hidden_dim), 1, file)) {return -1;};
        if (!fread(&n_layers, sizeof(n_layers), 1, file)) {return -1;};
        if (!fread(&n_heads, sizeof(n_heads), 1, file)) {return -1;};
        if (!fread(&n_kv_heads, sizeof(n_kv_heads), 1, file)) {return -1;};
        if (!fread(&vocab_size, sizeof(vocab_size), 1, file)) {return -1;};
        if (!fread(&seq_len, sizeof(seq_len), 1, file)) {return -1;};
        
        fclose(file);
        return 0;
    }

    void printConfig() const {
        std::cout << "Dimension: " << dim << std::endl;
        std::cout << "Hidden Dimension: " << hidden_dim << std::endl;
        std::cout << "Number of Layers: " << n_layers << std::endl;
        std::cout << "Number of Query Heads: " << n_heads << std::endl;
        std::cout << "Number of Key/Value Heads: " << n_kv_heads << std::endl;
        std::cout << "Vocabulary Size: " << vocab_size << std::endl;
        std::cout << "Sequence Length: " << seq_len << std::endl;
    }
};


class Tensor {
public:
    Tensor(int size) {
        data.resize(size);
        start = data.begin();
        end = data.end();
    }

    Tensor& subTensor(int offset, int length) {
        // cout << offset << "|" << length << "|" << data.size() << endl;
        if (offset + length > data.size()) {
            throw std::out_of_range("Subtensor out of range");
        }
        start = data.begin() + offset;
        end = start + length;
        return *this;
    }

    void reset() {
        start = data.begin();
        end = data.end();
    }

    void copyFrom(Tensor& tensor) {
        for (int i = 0; i < size(); i++) {
            data[i] = tensor[i];
        }
    }

    int readFromFile(FILE *file) {
        int result = fread(data.data(), sizeof(float), data.size(), file);
        start = data.begin();
        end = data.end();
        return result;
    }

    // Function to get the size of the current span
    int size() const {
        return std::distance(start, end);
    }

    // Access element in the span
    float& operator[](int index) {
        return *(start + index);
    }

    std::vector<float> data;
    std::vector<float>::iterator start;
    std::vector<float>::iterator end;
};

class TransformerWeights {
public:
    TransformerWeights(const Config& config) :
            head_size (config.dim / config.n_heads),
            token_embedding_table(Tensor(config.vocab_size * config.dim)),
            rms_att_weight(Tensor(config.n_layers * config.dim)),
            rms_ffn_weight(Tensor(config.n_layers * config.dim)),
            wq(Tensor(config.n_layers * config.dim * config.n_heads * head_size)),
            wk(Tensor(config.n_layers * config.dim * config.n_kv_heads * head_size)),
            wv(Tensor(config.n_layers * config.dim * config.n_kv_heads * head_size)),
            wo(Tensor(config.n_layers * config.n_heads * head_size * config.dim)),
            w1(Tensor(config.n_layers * config.hidden_dim * config.dim)),
            w2(Tensor(config.n_layers * config.dim * config.hidden_dim)),
            w3(Tensor(config.n_layers * config.hidden_dim * config.dim)),
            rms_final_weight(Tensor(config.dim)) {}
    int head_size;
    Tensor token_embedding_table;

    Tensor rms_att_weight;
    Tensor rms_ffn_weight;
    
    Tensor wq;
    Tensor wk;
    Tensor wv;
    Tensor wo;
    
    Tensor w1;
    Tensor w2;
    Tensor w3;

    Tensor rms_final_weight;
    Tensor & wcls = token_embedding_table;   // 当前固定是shared weigths的方式

    bool sample_weight = false;

    int loadFromFile(const string& filename) {
        FILE *file = fopen(filename.c_str(), "rb");
        if (!file) {
            cerr << "Failed to open file: " << filename << endl;
        }

        fseek(file, 28, 0); //Skip Config
        token_embedding_table.readFromFile(file);
        rms_att_weight.readFromFile(file);
        wq.readFromFile(file);
        wk.readFromFile(file);
        wv.readFromFile(file);
        wo.readFromFile(file);
        rms_ffn_weight.readFromFile(file);
        w1.readFromFile(file);
        w2.readFromFile(file);
        w3.readFromFile(file);
        rms_final_weight.readFromFile(file);

        if (sample_weight) {
            cout << "token_embedding_table" << endl;
            cout << token_embedding_table[0] << endl;
            cout << token_embedding_table[1] << endl;
            cout << "rms_att_weight" << endl;
            cout << rms_att_weight[0] << endl;
            cout << "wo" << endl;
            cout << wo[0] << endl;
            cout << "w3" << endl;
            cout << w3[0] << endl;
            cout << "rms_final_weight" << endl;
            cout << rms_final_weight[0] << endl;
            cout << rms_final_weight[rms_final_weight.size() - 1] << endl;
            cout << "wcls" << endl;
            cout << wcls[0] << endl;
        }
        
        return fclose(file);
    }
};


class RunState {
public:
    RunState(Config& c) : 
        x(Tensor(c.dim)),
        xb(Tensor(c.dim)),
        xb2(Tensor(c.dim)),
        hb(Tensor(c.hidden_dim)),
        hb2(Tensor(c.hidden_dim)),
        q(Tensor(c.dim)),
        att(Tensor(c.n_heads * c.seq_len)),
        logits(Tensor(c.vocab_size)),
        key_cache(Tensor(c.n_layers*c.seq_len*c.dim)), 
        value_cache(Tensor(c.n_layers*c.seq_len*c.dim)) {
    }

    // current wave of activations
    Tensor x; // activation at current time stamp (dim,)
    Tensor xb; // same, but inside a residual branch (dim,)
    Tensor xb2; // an additional buffer just for convenience (dim,)
    Tensor hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    Tensor hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    Tensor q; // query (dim,)
    Tensor att; // buffer for scores/attention values (n_heads, seq_len)
    Tensor logits; // output logits
    // kv cache
    Tensor key_cache; // (layer, seq_len, dim)
    Tensor value_cache; // (layer, seq_len, dim)
};

static void rmsnorm(Tensor& o, Tensor& x, Tensor& weight) {
    int size = x.size();
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;    // 避免除0
    ss = 1.0f / sqrt(ss);

    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

static void matmul(Tensor& xout, Tensor& x, Tensor& w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

static void softmax(Tensor& x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

class Transformer {
public:
    Config& config;
    TransformerWeights& weigths;
    RunState* state;

    bool sample_output = false;

    Transformer(Config& cfg, TransformerWeights& w)
        :config(cfg), weigths(w) {
            state = new RunState(config);
        }

    void test() {
        // cout << "Transformer Test" << endl;
        Tensor m1 = Tensor(3);
        m1.data = {1, 2, 3};
        Tensor m2 = Tensor(3);
        m2.data = {4, 5, 6};
        Tensor out2 = Tensor(3);
        rmsnorm(out2, m1, m2);
        // cout << "out2 " << out2[0] << endl;
    }

    Tensor forward(int token, int pos) {
        Tensor& x = state->x;
        int dim = config.dim;
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        int kv_mul = config.n_heads / config.n_kv_heads;
        int hidden_dim = config.hidden_dim;
        int head_size = dim / config.n_heads;

        weigths.token_embedding_table.subTensor(token * dim, dim);
        x.copyFrom(weigths.token_embedding_table);

        if (sample_output) {
            cout << "embeding[0]" << x[0] << ",";
        }

        for (auto l = 0; l < config.n_layers; l++) {

            // attention norm
            weigths.rms_att_weight.subTensor(l * dim, dim);
            rmsnorm(state->xb, x, weigths.rms_att_weight);
            if (sample_output && l == 1) {
                cout << "xb[0]" << state->xb[0] << "," << x[0] << "," << weigths.rms_att_weight[0] << endl;
            }

            // kv cache https://blog.csdn.net/ningyanggege/article/details/134564203
            int loff = l * config.seq_len * kv_dim;
            state->key_cache.subTensor(loff + pos * kv_dim, kv_dim);
            state->value_cache.subTensor(loff + pos * kv_dim, kv_dim);
            weigths.wq.subTensor(l * dim * dim, dim);
            weigths.wk.subTensor(l * dim * kv_dim, kv_dim);
            weigths.wv.subTensor(l * dim * kv_dim, kv_dim);

            matmul(state->q, state->xb, weigths.wq, dim, dim);
            matmul(state->key_cache, state->xb, weigths.wk, dim, kv_dim);
            matmul(state->value_cache, state->xb, weigths.wv, dim, kv_dim);

            if (sample_output && l == 1) {
                cout << "q k v " << state->q[10] << "," << state->key_cache[10] << "," << state->value_cache[10] << endl;
            }

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            // 另一路， PE
            // 原理参考 https://cloud.tencent.com/developer/article/2327751
            for (int i = 0; i < dim; i+=2) {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only

                if (sample_output && i == 0 && l == 0) {
                    cout << "rotn before " <<  state->q[0] << state->key_cache[10];
                }
                for (int v = 0; v < rotn; v++) {
                    Tensor& vec = v == 0 ? state->q : state->key_cache; // the vector to rotate (query or key)
                    float v0 = vec[i];
                    float v1 = vec[i+1];
                    vec[i]   = v0 * fcr - v1 * fci;     // 通过旋转q 、 k 将位置信息嵌入
                    vec[i+1] = v0 * fci + v1 * fcr;
                }

                if (sample_output && i == 0 && l == 0) {
                    cout << "rotn after" <<  state->q[0] << state->key_cache[10] << endl;
                }
            }

            // multihead attention. iterate over all heads
            // 参考 https://blog.csdn.net/v_JULY_v/article/details/130090649 中 1.3.1 和 1.3.2
            for (int h = 0; h < config.n_heads; h++) {
                state->q.subTensor(h * head_size, head_size);
                state->att.subTensor(h * config.seq_len, config.seq_len);

                for (int t = 0; t <= pos; t++) {
                    state->key_cache.subTensor(loff + t * kv_dim + (h / kv_mul) * head_size, head_size);
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += state->q[i] * state->key_cache[i];
                    }
                    score /= sqrtf(head_size);
                    // cout << pos << " " << h << " " << t << " " << score << " " << state->q[0] << " " << state->key_cache[0] << endl;
                    state->att[t] = score;
                }

                softmax(state->att, pos + 1);

                state->xb.subTensor(h * head_size, head_size);
                fill(state->xb.start, state->xb.end, 0.0f);
                for (int t = 0; t <= pos; t++) {
                    state->value_cache.subTensor(loff + t * kv_dim + (h / kv_mul) * head_size, head_size);
                    float a = state->att[t];
                    for (int i = 0; i < head_size; i++) {
                        state->xb[i] += a * state->value_cache[i];
                    }
                    // cout << "mla2 " << pos << " " << h << " " << t << " " << a << " " << state->xb[0] << endl;
                }

            }

            state->q.reset();
            state->xb.reset();
            if (sample_output && l == 0) {
                cout << "after mla: " << state->xb[0] << "," << x[0] << "," << weigths.rms_att_weight[0] << endl;
            }

            matmul(state->xb2, state->xb, weigths.wo.subTensor(l * dim * dim, dim * dim), dim, dim);

            if (sample_output) {
                cout << "" << endl;
                cout << "xb2 " << state->xb2[10] << endl;
            }

            // 残差
            for (int i = 0; i < dim; i++) {
                x[i] += state->xb2[i];
            }
            if (sample_output) {
                cout << "after residual " << x[10] << endl;
            }

            // ffn rmsnorm
            rmsnorm(state->xb, x, weigths.rms_ffn_weight.subTensor(l * dim, dim));

            // ffn
            matmul(state->hb, state->xb, weigths.w1.subTensor(l * dim * hidden_dim, hidden_dim), dim, hidden_dim);
            matmul(state->hb2, state->xb, weigths.w3.subTensor(l * dim * hidden_dim, hidden_dim), dim, hidden_dim);

            // SwiGLU
            for (int i = 0; i < hidden_dim; i++) {
                float val = state-> hb[i];
                val *= (1.0f / (1.0f + expf(-val)));
                val *= state->hb2[i];
                state->hb[i] = val;
            }

            if (sample_output && (l == 0 || l == 1)) {
                cout << "x2b[0]" << state->xb[0] << "," << x[0] << "," << weigths.rms_att_weight[0] << endl;
            }

            matmul(state->xb, state->hb, weigths.w2.subTensor(l* dim * hidden_dim, hidden_dim), hidden_dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += state->xb[i];
            }
            
            if (sample_output && (l == 0 || l == 1)) {
                cout << "x3b[0]" << state->xb[0] << "," << x[0] << "," << weigths.rms_att_weight[0] << endl;
            }

        }

        rmsnorm(x, x, weigths.rms_final_weight);

        state->logits.reset();
        weigths.wcls.reset();
        matmul(state->logits, x, weigths.wcls, config.dim, config.vocab_size);
        return state->logits;
    }

};

class Sampler {
public:
    int vocab_size;
    Sampler(Config& c): vocab_size(c.vocab_size) {

    }
    int sample(Tensor& tensor) {
        return sample_argmax(tensor, vocab_size);   // 目前只实现 temperature = 0的情况
    }

    int sample_argmax(Tensor& tensor, int n) {
        // return the index that has the highest probability
        int max_i = 0;
        float max_p = tensor[0];
        for (int i = 1; i < n; i++) {
            if (tensor[i] > max_p) {
                max_i = i;
                max_p = tensor[i];
            }
        }
        return max_i;
    }
};

class Tokenizer {
public:
    int vocab_size;
    int max_token_length;
    vector<string> vocab;
    vector<float> vocab_scores;
    Tokenizer(int v): vocab_size(v) {
        vocab.resize(vocab_size);
    }

    int loadFromFile(const string& filename) {
        FILE *file = fopen(filename.c_str(), "rb");
        if (!file) {
            cerr << "Failed to open file: " << filename << endl;
        }

        int r = fread(&max_token_length, sizeof(int), 1, file);

        int len;
        float score;
        for (int i = 0; i < vocab_size; i++) {
            if(!fread(&score, sizeof(float), 1, file)) { return -1; };
            if(!fread(&len, sizeof(int), 1, file)) { return -1; };
            string tmp(len, '\0');
            if(!fread(&tmp[0], sizeof(char), len, file)) { return -1; };
            vocab[i] = std::move(tmp);
        }

        fclose(file);
        return 0;
    }

    string decode(int prev_token, int token) {  // TODO
        string result = vocab[token];
        return result;
    }

};

class Generator {
public:
    Transformer& transformer;
    Sampler& sampler;
    Tokenizer& tokenizer;
    int steps = 256;

    Generator(Transformer& t, Sampler& s, Tokenizer& tk): transformer(t), sampler(s), tokenizer(tk) {
    }

    int generate(string prompt) {

        int number_prompt_tokens = 0;
        vector<int> prompt_tokens = {1};

        long start = 0;
        int next;
        int token = prompt_tokens.at(0); // BOS
        int pos = 0;
        while (pos < steps) {
            Tensor logits = transformer.forward(token, pos);

            // prompt 为空
            next = sampler.sample(logits);
            // cout << "token :" << next;
            cout << tokenizer.decode(token, next);
            fflush(stdout);
            pos++;
            if (next == 1) {
                break;
            }
            token = next;
        }
        return 0;
    }
};

/**
 * mkdir build
 * cd build
 * cmake ..
 * make
 * ./crun_test  //test
 * ./crun_cpp //run
 */
#ifndef RUN_TESTS
int main(int argc, char **argv) {
    string checkpoint_path = "/home/caohe/Workspace/hecao/llama2.c/stories42M.bin";
    string tokenizer_path = "/home/caohe/Workspace/hecao/llama2.c/tokenizer.bin";

    Config config;
    config.loadFromFile(checkpoint_path);
    config.printConfig();

    TransformerWeights weights(config);
    weights.loadFromFile(checkpoint_path);

    Transformer transformer(config, weights);
    transformer.test();

    Tokenizer tokenizer(config.vocab_size);
    tokenizer.loadFromFile(tokenizer_path);
    
    Sampler sampler(config);
    Generator g(transformer, sampler, tokenizer);
    g.generate("");
    cout << "end" << endl;
}
#endif

#endif // CRUN_CPP
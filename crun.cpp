#ifndef CRUN_CPP
#define CRUN_CPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

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

        fread(&dim, sizeof(dim), 1, file);
        fread(&hidden_dim, sizeof(hidden_dim), 1, file);
        fread(&n_layers, sizeof(n_layers), 1, file);
        fread(&n_heads, sizeof(n_heads), 1, file);
        fread(&n_kv_heads, sizeof(n_kv_heads), 1, file);
        fread(&vocab_size, sizeof(vocab_size), 1, file);
        fread(&seq_len, sizeof(seq_len), 1, file);
        
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

    void subTensor(int offset, int length) {
        // cout << offset << "|" << length << "|" << data.size() << endl;
        if (offset + length > data.size()) {
            throw std::out_of_range("Subtensor out of range");
        }
        start = data.begin() + offset;
        end = start + length;
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
        // k(Tensor(c.dim)),
        // v(Tensor(c.dim)),
        att(Tensor(c.dim)),
        logits(Tensor(c.dim)),
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
    // Tensor k; // key (dim,)
    // Tensor v; // value (dim,)
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

class Transformer {
public:
    Config& config;
    TransformerWeights& weigths;
    RunState* state;

    bool sample_output = true;

    Transformer(Config& cfg, TransformerWeights& w)
        :config(cfg), weigths(w) {
            state = new RunState(config);
        }

    void test() {
        cout << "Transformer Test" << endl;
        Tensor m1 = Tensor(3);
        m1.data = {1, 2, 3};
        Tensor m2 = Tensor(3);
        m2.data = {4, 5, 6};
        Tensor out2 = Tensor(3);
        rmsnorm(out2, m1, m2);
        cout << "out2 " << out2[0] << endl;
    }

    vector<float> forward(int token, int pos) {
        vector<float> result;

        Tensor& x = state->x;
        int dim = config.dim;
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        int kv_mul = config.n_heads / config.n_kv_heads;
        int hidden_dim = config.hidden_dim;
        int head_size = dim / config.n_heads;

        weigths.token_embedding_table.subTensor(token * dim, dim);
        x.copyFrom(weigths.token_embedding_table);
        weigths.token_embedding_table.reset();

        if (sample_output) {
            cout << "embeding[0]" << x[0] << ",";
        }

        for (auto l = 0; l < config.n_layers; l++) {

            // attention norm
            weigths.rms_att_weight.subTensor(l * dim, dim);
            rmsnorm(state->xb, x, weigths.rms_att_weight);
            if (sample_output && l == 0) {
                cout << "xb[0]" << state->xb[0] << "," << x[0] << "," << weigths.rms_att_weight[0] << endl;
            }
            // kv cache https://blog.csdn.net/ningyanggege/article/details/134564203\

            int loff = l * config.seq_len * kv_dim;
            state->key_cache.subTensor(loff + pos * kv_dim, kv_dim);
            state->value_cache.subTensor(loff + pos * kv_dim, kv_dim);
            weigths.wq.subTensor(l * dim * dim, dim);
            weigths.wk.subTensor(l * dim * kv_dim, kv_dim);
            weigths.wv.subTensor(l * dim * kv_dim, kv_dim);

            matmul(state->q, state->xb, weigths.wq, dim, dim);
            matmul(state->key_cache, state->xb, weigths.wk, dim, kv_dim);
            matmul(state->value_cache, state->xb, weigths.wv, dim, kv_dim);

            if (sample_output && l == 0) {
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
                    cout << "rotn b " <<  state->q[0] << state->key_cache[10];
                }
                for (int v = 0; v < rotn; v++) {
                    Tensor& vec = v == 0 ? state->q : state->key_cache; // the vector to rotate (query or key)
                    float v0 = vec[i];
                    float v1 = vec[i+1];
                    vec[i]   = v0 * fcr - v1 * fci;     // 通过旋转q 、 k 将位置信息嵌入，具体原理不太懂
                    vec[i+1] = v0 * fci + v1 * fcr;
                }

                if (sample_output && i == 0 && l == 0) {
                    cout << "rotn " <<  state->q[0] << state->key_cache[10] << endl;
                }
            }

        }

        return result;
    }

};

class Generator {
public:
    Transformer& transformer;
    int steps = 256;

    Generator(Transformer& t): transformer(t) {
    }

    int generate(string prompt) {

        int number_prompt_tokens = 0;
        vector<int> prompt_tokens;

        long start = 0;
        int next;
        // int token = prompt_tokens.at(0);
        int token = 1; // BOS
        int pos = 0;
        while (pos < steps) {
            vector<float> logits = transformer.forward(token, pos);

            pos++;
        }
        return 0;
    }
};

static void printSTH() {
    
}

/**
 * mkdir build
 * cd build
 * cmake ..
 * make
 * ./crun_test
 */
#ifndef RUN_TESTS

int main(int argc, char **argv) {
    cout << "Hello World" << endl;

    string checkpoint_path = "/home/caohe/Workspace/hecao/llama2.c/stories42M.bin";

    Config config;
    config.loadFromFile(checkpoint_path);
    config.printConfig();

    TransformerWeights weights(config);
    weights.loadFromFile(checkpoint_path);

    Transformer transformer(config, weights);
    transformer.test();
    
    Generator g(transformer);
    g.generate("");
    cout << "end" << endl;

}
#endif

#endif // CRUN_CPP
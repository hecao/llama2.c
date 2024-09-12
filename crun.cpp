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

class TransformerWeights {
public:
    TransformerWeights(const Config& config) {
        int head_size = config.dim / config.n_heads;
        token_embedding_table.resize(config.vocab_size * config.dim);
        rms_att_weight.resize(config.n_layers * config.dim);
        rms_ffn_weight.resize(config.n_layers * config.dim);
        wq.resize(config.n_layers * config.dim * config.n_heads * head_size);
        wk.resize(config.n_layers * config.dim * config.n_kv_heads * head_size);
        wv.resize(config.n_layers * config.dim * config.n_kv_heads * head_size);
        wo.resize(config.n_layers * config.n_heads * head_size * config.dim);
        w1.resize(config.n_layers * config.hidden_dim * config.dim);
        w2.resize(config.n_layers * config.dim * config.hidden_dim);
        w3.resize(config.n_layers * config.hidden_dim * config.dim);
        rms_final_weight.resize(config.dim);
    }

    vector<float> token_embedding_table;

    vector<float> rms_att_weight;
    vector<float> rms_ffn_weight;
    
    vector<float> wq;
    vector<float> wk;
    vector<float> wv;
    vector<float> wo;
    
    vector<float> w1;
    vector<float> w2;
    vector<float> w3;

    vector<float> rms_final_weight;
    vector<float> & wcls = token_embedding_table;   // 当前固定是shared weigths的方式

    bool sample_weight = false;

    int loadFromFile(const string& filename) {
        FILE *file = fopen(filename.c_str(), "rb");
        if (!file) {
            cerr << "Failed to open file: " << filename << endl;
        }

        fseek(file, 28, 0); //Skip Config
        fread(token_embedding_table.data(), sizeof(float), token_embedding_table.size(), file);
        fread(rms_att_weight.data(), sizeof(float), rms_att_weight.size(), file);
        fread(wq.data(), sizeof(float), wq.size(), file);
        fread(wk.data(), sizeof(float), wk.size(), file);
        fread(wv.data(), sizeof(float), wv.size(), file);
        fread(wo.data(), sizeof(float), wo.size(), file);
        fread(rms_ffn_weight.data(), sizeof(float), rms_ffn_weight.size(), file);
        fread(w1.data(), sizeof(float), w1.size(), file);
        fread(w2.data(), sizeof(float), w2.size(), file);
        fread(w3.data(), sizeof(float), w3.size(), file);
        fread(rms_final_weight.data(), sizeof(float), rms_final_weight.size(), file);

        if (sample_weight) {
            cout << "token_embedding_table" << endl;
            cout << token_embedding_table.at(0) << endl;
            cout << token_embedding_table.at(1) << endl;
            cout << "rms_att_weight" << endl;
            cout << rms_att_weight.at(0) << endl;
            cout << "wo" << endl;
            cout << wo.at(0) << endl;
            cout << "w3" << endl;
            cout << w3.at(0) << endl;
            cout << "rms_final_weight" << endl;
            cout << rms_final_weight.at(0) << endl;
            cout << rms_final_weight.at(rms_final_weight.size() - 1) << endl;
            cout << "wcls" << endl;
            cout << wcls.at(0) << endl;
        }
        
        return fclose(file);
    }
};

class RunState {
public:
    RunState() {}

    // current wave of activations
    vector<float> x; // activation at current time stamp (dim,)
    vector<float> xb; // same, but inside a residual branch (dim,)
    vector<float> xb2; // an additional buffer just for convenience (dim,)
    vector<float> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    vector<float> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    vector<float> q; // query (dim,)
    vector<float> k; // key (dim,)
    vector<float> v; // value (dim,)
    vector<float> att; // buffer for scores/attention values (n_heads, seq_len)
    vector<float> logits; // output logits
    // kv cache
    vector<float> key_cache;   // (layer, seq_len, dim)
    vector<float> value_cache; // (layer, seq_len, dim)
};

class Transformer {
public:
    Config& config;
    TransformerWeights& weigths;
    RunState state;

    bool sample_output = true;

    Transformer(Config& cfg, TransformerWeights& w)
        :config(cfg), weigths(w) {
            state = RunState();
        }

    void rmsnorm(std::vector<float>& o, const std::vector<float>& x, const std::vector<float>& weight) {
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
        o.resize(size); // 确保输出向量大小正确
        for (int j = 0; j < size; j++) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }

    void test() {
        cout << "Transformer Test" << endl;
        vector<float> out;
        vector<float> m = {1, 2, 3};
        vector<float> n = {4, 5, 6};
        rmsnorm(out, m, n);
        cout << "out " << out.at(0) << endl;
    }

    vector<float> forward(int token, int pos) {
        vector<float> result;

        vector<float> & x = state.x;
        int dim = config.dim;
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        int kv_mul = config.n_heads / config.n_kv_heads;
        int hidden_dim = config.hidden_dim;
        int head_size = dim / config.n_heads;

        x = {weigths.token_embedding_table.begin() + token * dim, weigths.token_embedding_table.begin() + token * dim + dim};

        if (sample_output) {
            cout << "embeding[0]" << x.at(0) << ",";
        }

        for (auto l = 0; l < config.n_layers; l++) {

            // attention norm
            vector<float> rms_att_w = {weigths.rms_att_weight.begin() + l * dim, weigths.rms_att_weight.begin() + l * dim + dim};
            rmsnorm(state.xb, x, rms_att_w);
            if (sample_output && l == 0) {
                cout << "xb[0]" << state.xb.at(0) << "," << x.at(0) << "," << rms_att_w.at(0);
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
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

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

    cout << "end" << endl;

}
#endif
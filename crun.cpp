#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class Config {
public: 
    Config() {}

    int32_t  dim;
    int32_t  hidden_dim;
    int32_t  n_layers;
    int32_t  n_heads;
    int32_t  n_kv_heads;
    int32_t  vocab_size;
    int32_t  seq_len;

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

    Config c = Config();
    c.loadFromFile(checkpoint_path);
    c.printConfig();

}
#endif
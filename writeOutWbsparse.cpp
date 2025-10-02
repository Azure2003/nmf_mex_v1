#include "writeOutWbsparse.h"
#include <fstream>
#include <vector>
#include <map>
void insert_number_before_wbsparse(std::string& s, int num) {
    const std::string target = ".trajTEMP.wbsparse";
    size_t pos = s.rfind(target);
    if (pos != std::string::npos) {
        std::string insert_str = "_" + std::to_string(num);
        s.insert(pos, insert_str);
    }
}
uint32_t myclamp(const int& x)
{
    if (x >= 1023) return 1023;
    if (x <= 0) return 0;
    return x;
}

uint64_t encodeFibers(int fiber1, int fiber2, int fiber3)
{
    if (fiber1<0) fiber1=0;
    if (fiber2<0) fiber2=0;
    if (fiber3<0) fiber3=0;
    int total = fiber1 + fiber2 + fiber3;
    if (total <= 0)
        throw std::runtime_error("Total fiber count must be positive.");

    double f0 = static_cast<double>(fiber1) / total;
    double f1 = static_cast<double>(fiber2) / total;
    double f2 = 1.0f - f0 - f1;

   /* if (f0 < 0.0f || f1 < 0.0f || f2 < 0.0f || f2 > 1.0f)
        throw std::runtime_error("Invalid fiber fractions derived from counts.");*/

    uint32_t i_f0 = myclamp(static_cast<int>(f0 * 1000.0f + 0.5f));
    uint32_t i_f1 = myclamp(static_cast<int>(f1 * 1000.0f + 0.5f));

    // distance is zero here
    uint32_t temp = (i_f1 << 10) | (i_f0 << 20);

    /*if (temp & (3u << 30))
        throw std::runtime_error("Overflow into reserved bits 30–31.");*/

    uint64_t result = (static_cast<uint64_t>(total) << 32) | temp;
    return result;
}
void write_wbsparse_cpp_from_matrix(Eigen::MatrixXd& W, Eigen::VectorXd& D, Eigen::MatrixXd& H, std::string filepath){
    int k=W.cols();
    const int n = H.cols();
    const int m = W.rows() / 3;
    insert_number_before_wbsparse(filepath, k);
    std::ofstream out(filepath, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open output file");

    // 1. Write 8-byte magic
    char magic[8] = {0, 0, 0, 0, 'c', 's', 't', 0};
    out.write(magic, 8);

    int64_t k64=k;
    int64_t m64=m;
    out.write(reinterpret_cast<const char*>(&m64), sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(&k64), sizeof(int64_t));
    std::vector<std::pair<int64_t, uint64_t>> encoded_entries;
    for(int k_index = 0; k_index<k; k_index++){
        int64_t len=0;
        for(int m_index = 0; m_index<m; m_index++){
            double value1=W(m_index*3,k_index)*D(k_index);
            double value2=W(m_index*3+1,k_index)*D(k_index);
            double value3=W(m_index*3+2,k_index)*D(k_index);
            double result1=0;
            double result2=0;
            double result3=0;
            if (value1 == 0.0 && value2 == 0.0 && value3 == 0.0)
                continue;
            for(int n_index=0; n_index<n; n_index++){
                result1+=value1*H(k_index, n_index);
                result2+=value2*H(k_index, n_index);
                result3+=value3*H(k_index, n_index);
            }
            if(static_cast<int>(result1)>0||static_cast<int>(result2)>0||static_cast<int>(result3)>0){
                uint64_t resultEncoded=encodeFibers(static_cast<int> (result1), static_cast<int> (result2), static_cast<int> (result3));
                len++;
                encoded_entries.emplace_back(static_cast<int64_t>(m_index), resultEncoded);
            }
        }
        out.write(reinterpret_cast<const char*>(&len), sizeof(int64_t));
    }
    for(const auto& entry : encoded_entries){
        out.write(reinterpret_cast<const char*>(&entry.first), sizeof(int64_t));
        out.write(reinterpret_cast<const char*>(&entry.second), sizeof(uint64_t));
    }
    out.close();
}
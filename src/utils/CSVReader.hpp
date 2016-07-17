#pragma once
#include <iostream>
#include <string>
#include <functional>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>

namespace csv {
    
class CSVReader {
public:
    CSVReader(const std::string& filename, char delimiter=';');

    CSVReader(std::istream& instream, char delimiter=';');

    template <typename T>
    std::vector<T> get_column(const std::string& col_name) {
        std::vector<T> res;
        auto& strings = get_column_strings_or_throw(col_name);
        std::transform(std::begin(strings), std::end(strings), std::back_inserter(res), [](const std::string& s) {
            T temp;
            std::stringstream ss(s);
            ss >> temp;
            return temp;
        });
        return res;
    }

private:
    void read_column_headers(std::istream& instream);

    void read_and_store_columns_as_string(std::istream& instream);

    const std::vector<std::string>& get_column_strings_or_throw(const std::string& col_name);

    std::vector<std::string> split_string(const std::string& s);

private:
    std::map<std::string, std::vector<std::string>>     m_string_data;
    char                                                m_delimiter;
    std::vector<std::string>                            m_column_headers;
};

}

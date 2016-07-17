#include <fstream>
#include <stdexcept>
#include <locale>   // for std::isspace
#include <algorithm>
#include "CSVReader.hpp"

namespace csv {

CSVReader::CSVReader(std::istream&& instream, char delimiter)
    : m_delimiter(delimiter)
{
    read_column_headers(instream);
    read_and_store_columns_as_string(instream);
}
CSVReader::CSVReader(const std::string& filename, char delimiter)
    : CSVReader(std::ifstream(filename, std::ios::in), delimiter) { }

void CSVReader::read_column_headers(std::istream& instream) {
    std::string str;
    if (std::getline(instream, str)) {
        m_column_headers = split_string(str);
    } else {
        throw std::runtime_error("unable to read column headers");
    }
}

std::vector<std::string> CSVReader::split_string(const std::string& s) {
    std::vector<std::string> res;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, m_delimiter)) {
        // strip any \r or \n
        item.erase(std::remove_if(std::begin(item), std::end(item), [](char c) {
            return (c=='\n') || (c=='\r');
        }), std::end(item));
        res.push_back(item);
    }
    return res;
}

void CSVReader::read_and_store_columns_as_string(std::istream& instream) {
    std::string cur_line;
    const auto num_columns = m_column_headers.size();
    while (std::getline(instream, cur_line)) {
        if (std::all_of(cur_line.begin(), cur_line.end(), isspace)) {
            continue;
        }

        auto string_parts = split_string(cur_line);
        if (string_parts.size() != num_columns) {
            throw std::runtime_error("mismatch between number of row entries and number of column headers");
        }
        for (size_t i = 0; i < num_columns; i++) {
            m_string_data[m_column_headers[i]].push_back(string_parts[i]);
        }
    }
}

const std::vector<std::string>& CSVReader::get_column_strings_or_throw(const std::string& col_name) {
    if (m_string_data.find(col_name) == std::end(m_string_data)) {
        throw std::runtime_error("no column found");
    }
    return m_string_data[col_name];
}

}   // end namespace

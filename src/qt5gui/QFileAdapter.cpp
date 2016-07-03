#include <QDataStream>
#include <boost/iostreams/stream.hpp>
#include "QFileAdapter.hpp"

namespace qfileadapter {

class InputAdapterInternal {
public:
    typedef char char_type;
    typedef boost::iostreams::source_tag category;

    InputAdapterInternal(QDataStream* const source)
        : m_source(source) { }

    std::streamsize read(char* buffer, std::streamsize n) {
        if (m_source) {
            return m_source->readRawData(buffer, n);
        } else {
            return -1;
        }
    }
private:
    QDataStream* const m_source;
};

InputAdapter::InputAdapter(QFile& inputfile) {
    m_data_stream = std::make_shared<QDataStream>(&inputfile);
    m_internal = std::make_shared<InputAdapterInternal>(m_data_stream.get());
    m_stream = std::make_shared<boost::iostreams::stream<InputAdapterInternal>>(*m_internal);
}

std::istream& InputAdapter::operator()() {
    return *m_stream;
}

}	// end namespace

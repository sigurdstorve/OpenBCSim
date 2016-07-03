#pragma once
#include <QFile>
#include <istream>
#include <memory>

namespace qfileadapter {

class InputAdapterInternal;
class InputAdapter {
public:
    // QFile object must remain in scope the entire time the adapter is used.
    InputAdapter(QFile& inputfile);
   
    std::istream& operator()();

private:
    std::shared_ptr<InputAdapterInternal> m_internal;
    std::shared_ptr<std::istream>		  m_stream;
    std::shared_ptr<QDataStream>		  m_data_stream;
};

}	// end namespace

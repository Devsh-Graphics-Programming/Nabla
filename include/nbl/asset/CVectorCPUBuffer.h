#ifndef _NBL_ASSET_C_VECTOR_CPU_BUFFER_H_INCLUDED_
#define _NBL_ASSET_C_VECTOR_CPU_BUFFER_H_INCLUDED_

#include "nbl/asset/ICPUBuffer.h"

namespace nbl::asset {

template<typename T, typename Allocator>
class CVectorCPUBuffer final : public ICPUBuffer
{
public:
    inline CVectorCPUBuffer(core::vector<T, Allocator>&& _vec) : ICPUBuffer(_vec.size(), _vec.data()), m_vec(std::move(_vec))
    {
    }

protected:
    virtual inline ~CVectorCPUBuffer()
    {
        freeData();
    }
    inline void freeData() override
    {
        m_vec.clear();
        ICPUBuffer::data = nullptr;
    }

    core::vector<T, Allocator> m_vec;
};

}



#endif
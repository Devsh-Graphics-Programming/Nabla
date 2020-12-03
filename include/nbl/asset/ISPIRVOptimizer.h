#ifndef __IRR_I_SPIRV_OPTIMIZER_H_INCLUDED__
#define __IRR_I_SPIRV_OPTIMIZER_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl
{

namespace asset
{

class ISPIRVOptimizer final : public core::IReferenceCounted
{
public:
    core::smart_refctd_ptr<ICPUBuffer> optimize(const uint32_t* _spirv, uint32_t _dwordCount) const;
    core::smart_refctd_ptr<ICPUBuffer> optimize(const ICPUBuffer* _spirv) const;
};

}

}

#endif
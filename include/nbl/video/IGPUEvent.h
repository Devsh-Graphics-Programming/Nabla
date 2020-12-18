#ifndef __NBL_I_GPU_EVENT_H_INCLUDED__
#define __NBL_I_GPU_EVENT_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

namespace nbl {
namespace video
{

class IGPUEvent : public core::IReferenceCounted
{
public:
    enum E_STATUS : uint32_t
    {
        ES_SET,
        ES_RESET,
        ES_FAILURE
    };

protected:
    virtual ~IGPUEvent() = default;
};

}}

#endif
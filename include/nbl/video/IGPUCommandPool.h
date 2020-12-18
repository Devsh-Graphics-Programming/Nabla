#ifndef __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

namespace nbl {
namespace video
{

class IGPUCommandPool : public core::IReferenceCounted
{
public:
    enum E_CREATE_FLAGS : uint32_t
    {
        ECF_TRANSIENT_BIT = 0x01,
        ECF_RESET_COMMAND_BUFFER_BIT = 0x02,
        ECF_PROTECTED_BIT = 0x04
    };
};

}}


#endif
#ifndef __NBL_I_EVENT_H_INCLUDED__
#define __NBL_I_EVENT_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

namespace nbl {
namespace asset
{

class IEvent
{
public:
    enum E_CREATE_FLAGS
    {
        ECF_DEVICE_ONLY_BIT = 0x01
    };

    enum E_STATUS : uint32_t
    {
        ES_SET,
        ES_RESET,
        ES_FAILURE
    };

protected:
    virtual ~IEvent() = default;
};

}}

#endif
#ifndef _NBL_I_EVENT_H_INCLUDED_
#define _NBL_I_EVENT_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"

namespace nbl::asset
{

class IEvent
{
    public:
        enum class CREATE_FLAGS : uint8_t
        {
            NONE = 0x00u,
            DEVICE_ONLY_BIT = 0x01u
        };

        enum class STATUS : uint8_t
        {
            SET,
            RESET,
            FAILURE
        };

        CREATE_FLAGS getFlags() const { return m_flags; }

    protected:
        virtual ~IEvent() = default;

        IEvent(const CREATE_FLAGS _flags) : m_flags(_flags) {}

        const CREATE_FLAGS m_flags;
};

}

#endif
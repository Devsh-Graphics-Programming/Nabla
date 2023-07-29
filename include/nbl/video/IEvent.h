#ifndef _NBL_VIDEO_I_EVENT_H_INCLUDED_
#define _NBL_VIDEO_I_EVENT_H_INCLUDED_


#include "nbl/core/IReferenceCounted.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IEvent : public IBackendObject
{
    public:
        enum class CREATE_FLAGS : uint8_t
        {
            NONE = 0x00u,
            DEVICE_ONLY_BIT = 0x01u
        };
        inline auto getFlags() const { return m_flags; }

        enum class STATUS : uint8_t
        {
            SET,
            RESET,
            FAILURE
        };
        inline STATUS getEventStatus() const
        {
            if (m_flags.hasFlags(CREATE_FLAGS::DEVICE_ONLY_BIT))
                return getEventStatus_impl();
            return STATUS::FAILURE;
        }
        inline STATUS resetEvent()
        {
            if (m_flags.hasFlags(CREATE_FLAGS::DEVICE_ONLY_BIT))
                return resetEvent_impl();
            return STATUS::FAILURE;
        }
        inline STATUS setEvent()
        {
            if (m_flags.hasFlags(CREATE_FLAGS::DEVICE_ONLY_BIT))
                return setEvent_impl();
            return STATUS::FAILURE;
        }

    protected:
        explicit IEvent(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const core::bitflag<CREATE_FLAGS> flags) : IBackendObject(std::move(dev)), m_flags(flags) {}
        virtual ~IEvent() = default;

        virtual STATUS getEventStatus_impl() const =0;
        virtual STATUS resetEvent_impl() =0;
        virtual STATUS setEvent_impl() =0;

        const core::bitflag<CREATE_FLAGS> m_flags;
};

}

#endif

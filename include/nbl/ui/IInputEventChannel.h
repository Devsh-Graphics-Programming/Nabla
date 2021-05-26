#ifndef __NBL_I_INPUT_EVENT_CHANNEL_H_INCLUDED__
#define __NBL_I_INPUT_EVENT_CHANNEL_H_INCLUDED__

#include <mutex>

#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/containers/CCircularBuffer.h"
#include "nbl/core/SRange.h"

namespace nbl {
namespace ui
{

class IInputEventChannel : public core::IReferenceCounted
{
protected:
    virtual ~IInputEventChannel() = default;

public:
    // TODO Any other common (not depending on event struct type) member functions?

    virtual bool empty() const = 0;
};

namespace impl
{
    template <typename EventType>
    class IEventChannelBase : public IInputEventChannel
    {
    protected:
        static inline constexpr size_t MaxEvents = 1024u;

        using cb_t = core::CCompileTimeSizedCircularBuffer<EventType, MaxEvents>;
        using iterator_t = typename cb_t::iterator;
        using range_t = core::SRange<EventType, iterator_t, iterator_t>;

        std::mutex m_bgEventBufMtx;
        cb_t m_bgEventBuf;
        cb_t m_frontEventBuf;

        // Lock while working with background event buffer
        std::unique_lock<std::mutex> lockBackgroundBuffer()
        {
            return std::unique_lock<std::mutex>(m_bgEventBufMtx);
        }

        // Use this within OS-specific impl (Windows callback/XNextEvent loop thread/etc...)
        void pushIntoBackground(EventType&& ev)
        {
            m_bgEventBuf.push_back(std::move(ev));
        }

        // WARNING: Access to getEvents() must be externally synchronized to be safe!
        // (the function returns range of iterators)
        range_t getEvents()
        {
            downloadFromBackgroundIntoFront();
            return range_t(m_frontEventBuf.begin(), m_frontEventBuf.end());
        }

        virtual ~IEventChannelBase() = default;

    private:
        void downloadFromBackgroundIntoFront()
        {
            auto lk = lockBackgroundBuffer();

            while (m_bgEventBuf.size() > 0ull)
            {
                auto& ev = m_bgEventBuf.pop_front();
                m_frontEventBuf.push_back(std::move(ev));
            }
        }

    public:
        bool empty() const override final
        {
            return m_bgEventBuf.size() == 0ull;
        }
    };
}

struct SMouseEvent
{
    // TODO
};

class IMouseEventChannel : public impl::IEventChannelBase<SMouseEvent>
{
protected:
    virtual ~IMouseEventChannel() = default;
};

struct SKeyboardEvent
{
    // TODO
};

class IKeyboardEventChannel : public impl::IEventChannelBase<SKeyboardEvent>
{
protected:
    virtual ~IKeyboardEventChannel() = default;
};

}
}

#endif

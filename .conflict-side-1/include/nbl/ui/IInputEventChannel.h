#ifndef __NBL_I_INPUT_EVENT_CHANNEL_H_INCLUDED__
#define __NBL_I_INPUT_EVENT_CHANNEL_H_INCLUDED__

#include <mutex>
#include <chrono>

#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/containers/CCircularBuffer.h"
#include "nbl/core/SRange.h"

#include "nbl/ui/KeyCodes.h"
#include "nbl/ui/SInputEvent.h"

namespace nbl::ui
{

class IWindow;
class IInputEventChannel : public core::IReferenceCounted
{
    protected:
        NBL_API2 virtual ~IInputEventChannel() = default;

    public:
        // TODO Any other common (not depending on event struct type) member functions?
        enum E_TYPE
        {
            ET_MOUSE,
            ET_KEYBOARD,
            ET_MULTITOUCH,
            ET_GAMEPAD,

            ET_COUNT
        };

        virtual bool empty() const = 0;
        virtual E_TYPE getType() const = 0;
};

namespace impl
{

template <typename EventType>
class IEventChannelBase : public IInputEventChannel
{
    protected:
        inline virtual ~IEventChannelBase() = default;
        inline explicit IEventChannelBase(size_t _circular_buffer_capacity) :
            m_bgEventBuf(_circular_buffer_capacity), m_frontEventBuf(_circular_buffer_capacity)
        {
        }

        using cb_t = core::CConstantRuntimeSizedCircularBuffer<EventType>;
        using iterator_t = typename cb_t::iterator;

        std::mutex m_bgEventBufMtx;
        cb_t m_bgEventBuf;
        cb_t m_frontEventBuf;

    public:
        //
        inline size_t getBackgroundBufferCapacity() const {return m_frontEventBuf.capacity();}
        inline size_t getFrontBufferCapacity() const {return m_frontEventBuf.capacity();}

        // Lock while working with background event buffer
        inline std::unique_lock<std::mutex> lockBackgroundBuffer()
        {
            return std::unique_lock<std::mutex>(m_bgEventBufMtx);
        }

        inline bool empty() const override final
        {
            return m_bgEventBuf.size() == 0ull;
        }

        // Use this within OS-specific impl (Windows callback/XNextEvent loop thread/etc...)
        inline void pushIntoBackground(EventType&& ev)
        {
            m_bgEventBuf.push_back(std::move(ev));
        }

        // WARNING: Access to getEvents() must be externally synchronized to be safe!
        // (the function returns range of iterators)
        using range_t = core::SRange<EventType, iterator_t, iterator_t>;
        inline range_t getEvents()
        {
            downloadFromBackgroundIntoFront();
            return range_t(m_frontEventBuf.begin(), m_frontEventBuf.end());
        }
        
        template<typename F, class ChannelType> requires std::is_base_of_v<IEventChannelBase<EventType>,ChannelType>
        class CChannelConsumer : public core::IReferenceCounted
        {
            public:
                CChannelConsumer(F&& process, core::smart_refctd_ptr<ChannelType>&& channel)
                    : m_process(std::move(process)), m_channel(std::move(channel)) {}

                inline void operator()()
                {
                    auto events = m_channel->getEvents();
                    const auto frontBufferCapacity = m_channel->getFrontBufferCapacity();
                    if (events.size()>consumedCounter+frontBufferCapacity)
                    {
                        m_process.overflow(events.size()-consumedCounter,m_channel.get());
                        consumedCounter = events.size()-frontBufferCapacity;
                    }
                    m_process(range_t(events.begin()+consumedCounter,events.end()),m_channel.get());
                    consumedCounter = events.size();
                }

                const auto* getChannel() const {return m_channel.get();}

            protected:
                F m_process;
                core::smart_refctd_ptr<ChannelType> m_channel;
                uint64_t consumedCounter = 0ull;
        };

    private:
        inline void downloadFromBackgroundIntoFront()
        {
            auto lk = lockBackgroundBuffer();

            while (m_bgEventBuf.size() > 0ull)
            {
                auto ev = m_bgEventBuf.pop_front();
                m_frontEventBuf.push_back(std::move(ev));
            }
        }
};
}

class NBL_API2 IMouseEventChannel : public impl::IEventChannelBase<SMouseEvent>
{
        using base_t = impl::IEventChannelBase<SMouseEvent>;

    protected:
        virtual ~IMouseEventChannel() = default;

    public:
        inline IMouseEventChannel(size_t circular_buffer_capacity) : base_t(circular_buffer_capacity)
        {
        }
        using base_t::base_t;

        inline E_TYPE getType() const override final
        { 
            return ET_MOUSE;
        }

        template<typename F>
        using CChannelConsumer = base_t::CChannelConsumer<F,IMouseEventChannel>;
};

// TODO left/right shift/ctrl/alt kb flags
class NBL_API2 IKeyboardEventChannel : public impl::IEventChannelBase<SKeyboardEvent>
{
        using base_t = impl::IEventChannelBase<SKeyboardEvent>;

    protected:
        virtual ~IKeyboardEventChannel() = default;

    public:
        using base_t::base_t;
        inline IKeyboardEventChannel(size_t circular_buffer_capacity) : base_t(circular_buffer_capacity)
        {
        }

        inline E_TYPE getType() const override final
        {
            return ET_KEYBOARD;
        }

        template<typename F>
        using CChannelConsumer = base_t::CChannelConsumer<F,IKeyboardEventChannel>;
};

}

#endif

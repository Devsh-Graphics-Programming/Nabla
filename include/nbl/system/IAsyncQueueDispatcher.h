#ifndef __NBL_I_ASYNC_QUEUE_DISPATCHER_H_INCLUDED__
#define __NBL_I_ASYNC_QUEUE_DISPATCHER_H_INCLUDED__

#include "nbl/system/IThreadHandler.h"
#include "nbl/core/Types.h"

namespace nbl {
namespace system
{

template <typename QueueElementType, typename InternalStateType>
class IAsyncQueueDispatcher : public IThreadHandler<InternalStateType>
{
    using base_t = system::IThreadHandler<InternalStateType>;

public:
    void enqueue(uint32_t count, const queue_element_t* elements)
    {
        auto raii_handler = base_t::createRAIIDisptachHandler();

        for (uint32_t i = 0u; i < count; ++i)
        {
            auto e = elements[i];
            q.push(std::move(e));
        }
    }

    virtual ~IAsyncQueueDispatcher() = default;

protected:
    using queue_element_t = QueueElementType;

    virtual void processElement(internal_state_t& state, queue_element_t&& e) const = 0;

    void work(lock_t& lock, internal_state_t& state) override final
    {
        // pop item from queue
        auto e = std::move(m_queue.front());
        m_queue.pop();

        lock.unlock();

        processElement(state, e);

        lock.lock();
    }

    bool wakeupPredicate() const override final { return m_queue.size() || base_t::wakeupPredicate(); }
    bool continuePredicate() const override final { return m_queue.size() && base_t::continuePredicate(); }

private:
    core::queue<queue_element_t> m_queue;
};

}
}

#endif

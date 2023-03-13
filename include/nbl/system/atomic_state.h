#ifndef _NBL_SYSTEM_ATOMIC_STATE_H_INCLUDED_
#define _NBL_SYSTEM_ATOMIC_STATE_H_INCLUDED_

#include <atomic>

namespace nbl::system
{

template<class STATE, STATE kInitial=static_cast<STATE>(0u)>
class atomic_state_t
{
        static_assert(std::is_enum_v<STATE>);

    public:
        ~atomic_state_t()
        {
            static_assert(std::atomic_uint32_t::is_always_lock_free);
            // must have been consumed before exit !
            const auto atExit = state.load();
            assert(static_cast<STATE>(atExit)==kInitial);
        }

        //!
        inline STATE query() const {return static_cast<STATE>(state.load());}

        //!
        template<class ConditionLambda>
        inline void wait(ConditionLambda&& cond) const
        {
            uint32_t current;
            while (cond(static_cast<STATE>(current=state.load())))
                state.wait(current);
        }

        //!
        [[nodiscard]] inline bool tryTransition(const STATE to, STATE& expected)
        {
            return state.compare_exchange_strong(reinterpret_cast<uint32_t&>(expected),static_cast<uint32_t>(to));
        }

        //!
        inline void waitTransition(const STATE to, const STATE from)
        {
            STATE expected = from;
            while (!tryTransition(to,expected))
            {
                state.wait(static_cast<uint32_t>(expected));
                expected = from;
            }
            assert(expected==from);
        }
        [[nodiscard]] inline bool waitAbortableTransition(const STATE to, const STATE from, const STATE abortState)
        {
            uint32_t expected = static_cast<uint32_t>(from);
            while (!tryTransition(to,expected))
            {
                state.wait(expected);
                if (expected==static_cast<uint32_t>(abortState))
                    return false;
                expected = from;
            }
            assert(expected==from);
            return true;
        }

        //!
        template<bool all=true>
        [[nodiscard]] inline STATE exchangeNotify(const STATE to)
        {
            const auto prev = state.exchange(static_cast<uint32_t>(to));
            if constexpr (all)
                state.notify_all();
            else
                state.notify_one();
            return static_cast<STATE>(prev);
        }
        template<bool all=true>
        inline void exchangeNotify(const STATE to, const STATE expected)
        {
            const auto prev = exchangeNotify<all>(to);
            assert(static_cast<STATE>(prev)==expected);
        }

    private:
        std::atomic_uint32_t state = static_cast<uint32_t>(kInitial);
};

}

#endif
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_CORE_SINGLE_EVENT_HANDLER_H__
#define __NBL_CORE_CORE_SINGLE_EVENT_HANDLER_H__


#include "nbl/core/decl/Types.h"

namespace nbl::core
{

// TODO: actually implement and test
//#define NBL_EVENT_DEREGISTER_IMPLEMENTATION_READY
class NBL_API SingleEventHandler
{
    public:
        using Function = std::function<void()>;

    protected:
        using FunctionContainerType = core::forward_list<Function>;
        using FunctionContainerIt = typename FunctionContainerType::iterator;

        bool                    mExecuteOnDestroy;
		uint32_t				mFunctionsCount;
        FunctionContainerType   mFunctions;
        FunctionContainerIt     mLastFunction;
#ifdef NBL_EVENT_DEREGISTER_IMPLEMENTATION_READY
        // returns prev and 
        inline std::pair<FunctionContainerIt,FunctionContainerIt> findFunction(const Function& function)
        {
            auto prev = mFunctions.before_begin();
            auto curr = mFunctions.begin();
            while (prev!=mLastFunction)
            {
                if (*curr==function)
                    break;
                prev = curr++;
            }
            return {prev,curr};
        }
#endif
    public:
        SingleEventHandler(bool executeEventsOnDestroy) : mExecuteOnDestroy(executeEventsOnDestroy), mFunctionsCount(0u)
        {
            mLastFunction = mFunctions.before_begin();
        }

        virtual ~SingleEventHandler()
        {
            if (mExecuteOnDestroy)
            for (auto& func : mFunctions)
                func();
        }

        //
        inline auto getFunctionCount() const { return mFunctionsCount; }

        //
        inline void registerFunction(Function&& function)
        {
            mLastFunction = mFunctions.emplace_after(mLastFunction,std::forward<Function>(function));
            mFunctionsCount++;
        }
#ifdef NBL_EVENT_DEREGISTER_IMPLEMENTATION_READY
 // no comparison operator for std::function<> so no find
        //! does not call the operator()
        inline void deregisterFunction(const Function& function)
        {
            auto found = findFunction(function);
            if (found.first!=mLastFunction)
            {
                if (found.second==mLastFunction)
                    mLastFunction = found.first;
                mFunctions.erase_after(found.first);
            }
        }

        inline void swapFunctions(const Function& oldFunction, Function&& newFunction)
        {
            auto found = findFunction(oldFunction);
            if (found.second!=mFunctions.end())
                found.second->swap(newFunction);
        }
#endif
        //
        inline void execute()
        {
            for (auto& func : mFunctions)
                func();
            mFunctionsCount = 0u;
            mFunctions.clear();
            mLastFunction = mFunctions.before_begin();
        }
};

//
class NBL_API QuitSignalling
{
    public:
        inline void registerOnQuit(SingleEventHandler::Function&& function)
        {
            quitEventHandler.registerFunction(std::move(function));
        }
#ifdef NBL_EVENT_DEREGISTER_IMPLEMENTATION_READY
        //! does not call the operator()
        inline void deregisterOnQuit(const SingleEventHandler::Function& function)
        {
            quitEventHandler.deregisterFunction(function);
        }
#endif
    protected:
        QuitSignalling() : quitEventHandler(false) {}
        virtual ~QuitSignalling() {assert(!quitEventHandler.getFunctionCount());}

        SingleEventHandler quitEventHandler;
};

#ifdef NBL_EVENT_DEREGISTER_IMPLEMENTATION_READY
//
template<class T>
class NBL_API FactoryAndStaticSafeST
{
        T data = {};
        QuitSignalling* factory = nullptr;

    protected:
        virtual void preemptiveDestruction()
        {
            data = T();
            factory = nullptr;
        }

    public:
        FactoryAndStaticSafeST() = default;
        ~FactoryAndStaticSafeST()
        {
            assert(!factory);
        }

        T& getData(QuitSignalling* _factory)
        {
            if (_factory!=factory)
            {
                std::function<void()> func(preemptiveDestruction);
                if (factory)
                    factory->deregisterOnQuit(func);
                _factory->registerOnQuit(std::move(func));
                factory = _factory;
            }
            return data;
        }
};

//
template<class T, class Lockable=std::mutex>
class NBL_API FactoryAndStaticSafeMT : protected FactoryAndStaticSafeST<T>
{
        static_assert(std::is_standard_layout<Lockable>::value, "Lock class is not standard layout");
        Lockable lock;

    protected:
        inline void preemptiveDestruction() override
        {
            lock.lock();
            FactoryAndStaticSafeST<T>::preemptiveDestruction();
            lock.unlock();
        }

    public:
        FactoryAndStaticSafeMT() = default;
        ~FactoryAndStaticSafeMT() {}
        
        std::pair<T&,std::unique_lock<Lockable>> getData(QuitSignalling* _factory)
        {
            std::unique_lock lockFirst(lock);
            return {FactoryAndStaticSafeST<T>::getData(),std::move(lockFirst)};
        }
};
#endif

}

#endif





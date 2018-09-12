// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_ADDRESS_ALLOCATOR_STATE_H_INCLUDED__
#define __IRR_ADDRESS_ALLOCATOR_STATE_H_INCLUDED__

#include "IrrCompileConfig.h"

namespace irr
{
namespace core
{

template<class AddressAlloc>
class AddressAllocatorState : public AddressAlloc, public IReferenceCounted
{
        uint8_t* const                          bufferStart;
    protected:
        virtual ~AddressAllocatorState() {}
    public:

        template<typename... Args>
        AddressAllocatorState(void* buffer, Args&&... args) noexcept :
                    AddressAlloc(reinterpret_cast<size_t>(buffer)&(AddressAlloc::max_alignment-1u),std::forward<Args>(args)...),
                    bufferStart(reinterpret_cast<uint8_t*>(buffer)) {}

        inline uint8_t*                         getBufferStart() noexcept {return bufferStart;}
};


//! B is an IBuffer derived type, S is some AddressAllocatorState
template<class M, class S>
class AllocatorStateDriverMemoryAdaptor : public S
{
        M* const    memory;
    protected:
        virtual ~AllocatorStateDriverMemoryAdaptor()
        {
            if (memory) // move constructor compatibility
                memory->drop();
        }
    public:
        AllocatorStateDriverMemoryAdaptor(M* mem) : S(mem->getMappedPointer(),mem->getAllocationSize()), memory(mem)
        {
            memory->grab();
        }
};

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_STATE_H_INCLUDED__


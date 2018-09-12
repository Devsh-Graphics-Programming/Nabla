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
        const typename AddressAlloc::size_type  reservedSpace;
        uint8_t* const                          bufferStart;
    protected:
        virtual ~AddressAllocatorState() {}
    public:

#define CALC_RESERVED_SPACE (AddressAlloc::reserved_size(reinterpret_cast<size_t>(buffer),bufSz,std::forward<Args>(args)...))
        template<typename... Args>
        AddressAllocatorState(void* buffer, typename AddressAlloc::size_type bufSz, Args&&... args) noexcept :
                AddressAlloc(buffer, reinterpret_cast<typename AddressAlloc::size_type>(reinterpret_cast<uint8_t*>(buffer)+CALC_RESERVED_SPACE), bufSz-CALC_RESERVED_SPACE, std::forward<Args>(args)...),
                        reservedSpace(CALC_RESERVED_SPACE), bufferStart(reinterpret_cast<uint8_t*>(buffer)+reservedSpace)
        {
        }
#undef CALC_RESERVED_SPACE

        inline uint8_t*                         getBufferStart() noexcept {return bufferStart;}
};


//! I is an IReferenceCounted derived type
template<class I, class AddressAlloc>
class AllocatorStateRefCountedAdaptor : public AddressAllocatorState<AddressAlloc>
{
        I* const    associatedObj;
    protected:
        virtual ~AllocatorStateRefCountedAdaptor()
        {
            if (associatedObj) // move constructor compatibility
                associatedObj->drop();
        }
    public:
        template<typename... Args>
        AllocatorStateRefCountedAdaptor(I* assocObjToBuffmem, void* buffmem, typename AddressAlloc::size_type bufSz, Args&&... args) noexcept :
                    AddressAllocatorState<AddressAlloc>(buffmem,bufSz,std::forward<Args>(args)...), associatedObj(assocObjToBuffmem)
        {
            associatedObj->grab();
        }
};

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_STATE_H_INCLUDED__


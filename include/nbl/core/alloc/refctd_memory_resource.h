// Copyright (C) 2019-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_CORE_ALLOC_REFCTD_MEMORY_RESOURCE_INCLUDED_
#define _NBL_CORE_ALLOC_REFCTD_MEMORY_RESOURCE_INCLUDED_

#include "BuildConfigOptions.h"
#include "nbl/core/IReferenceCounted.h"

#include <memory_resource>

using namespace nbl;

namespace nbl::core
{

class refctd_memory_resource : public core::IReferenceCounted
{
    public:
        refctd_memory_resource(std::pmr::memory_resource* pmr) : m_pmr(pmr) {};

        void* allocate(size_t bytes, size_t alignment) {
            return m_pmr->allocate(bytes, alignment);
        }

        void deallocate(void* p, size_t bytes, size_t alignment) {
            return m_pmr->deallocate(p, bytes, alignment);
        }

    private:
        std::pmr::memory_resource* m_pmr;
};

NBL_API2 smart_refctd_ptr<refctd_memory_resource> getNullMemoryResource();
NBL_API2 smart_refctd_ptr<refctd_memory_resource> getDefaultMemoryResource();
NBL_API2 void setDefaultMemoryResource(smart_refctd_ptr<refctd_memory_resource> memoryResource);

}

#endif
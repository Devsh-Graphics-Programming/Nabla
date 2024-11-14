// Copyright (C) 2019-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_CORE_ALLOC_VECTOR_VIEW_NULL_MEMORY_RESOURCE_INCLUDED_
#define _NBL_CORE_ALLOC_VECTOR_VIEW_NULL_MEMORY_RESOURCE_INCLUDED_

#include <memory_resource>

using namespace nbl;

namespace nbl::core
{

class VectorViewNullMemoryResource : public std::pmr::memory_resource
{
    public:
        // only create the resource from an already sized vector
        VectorViewNullMemoryResource(core::vector<uint8_t>&& buffer) : buffer(buffer)
        {
            assert(buffer.size());
        };

        void* data() {
            return buffer.data();
        }

    protected:
        void* do_allocate(size_t bytes, size_t alignment) override {
            assert(bytes <= buffer.size());
            return buffer.data();
        }

        void do_deallocate(void* p, size_t bytes, size_t alignment) override {}

        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            return this == &other;
        }

    private:
        core::vector<uint8_t> buffer;
};

}

#endif
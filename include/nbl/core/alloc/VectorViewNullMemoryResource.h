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
        VectorViewNullMemoryResource(core::vector<uint8_t>&& buffer) : buffer(std::move(buffer)), already_called(false)
        {
            assert(buffer.size());
        }

        void* data() {
            return buffer.data();
        }

    protected:
        void* do_allocate(size_t bytes, size_t alignment) override
        {
            if (already_called || bytes > buffer.size() || !core::is_aligned_to(buffer.data(), alignment))
                return nullptr;
            already_called = true;
            return buffer.data();
        }

        void do_deallocate(void* p, size_t bytes, size_t alignment) override
        {
            assert(p == buffer.data());
            already_called = false;
        }

        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
        {
            return this == &other;
        }

    private:
        core::vector<uint8_t> buffer;
        bool already_called;
};

}

#endif
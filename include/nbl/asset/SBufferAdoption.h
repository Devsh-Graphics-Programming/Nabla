// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_BUFFER_ADOPTION_H_INCLUDED_
#define _NBL_ASSET_S_BUFFER_ADOPTION_H_INCLUDED_
#include <concepts>
#include <ranges>
#include <type_traits>
#include <utility>
#include "nbl/asset/ICPUBuffer.h"
namespace nbl::asset
{
namespace impl
{
// Owns contiguous storage that can be adopted by a CPU buffer. Views like std::span are rejected.
template<typename Storage>
concept AdoptedBufferStorage =
    std::ranges::contiguous_range<std::remove_reference_t<Storage>> &&
    std::ranges::sized_range<std::remove_reference_t<Storage>> &&
    (!std::ranges::view<std::remove_cvref_t<Storage>>) &&
    requires(std::remove_reference_t<Storage>& storage)
    {
        typename std::ranges::range_value_t<std::remove_reference_t<Storage>>;
        { std::ranges::data(storage) } -> std::same_as<std::ranges::range_value_t<std::remove_reference_t<Storage>>*>;
    };
}
// Generic CPU-buffer adoption helper for owning contiguous storage such as std::vector or core::vector.
class SBufferAdoption
{
    public:
        template<impl::AdoptedBufferStorage Storage>
        static inline core::smart_refctd_ptr<ICPUBuffer> create(Storage&& data)
        {
            using storage_t = std::remove_cvref_t<Storage>;
            using value_t = std::ranges::range_value_t<storage_t>;

            if (std::ranges::empty(data))
                return nullptr;

            auto backer = core::make_smart_refctd_ptr<core::adoption_memory_resource<storage_t>>(std::forward<Storage>(data));
            auto& storage = backer->getBacker();
            const size_t byteCount = std::ranges::size(storage) * sizeof(value_t);
            return ICPUBuffer::create(
                { { byteCount }, std::ranges::data(storage), core::smart_refctd_ptr<core::refctd_memory_resource>(std::move(backer)), alignof(value_t) },
                core::adopt_memory);
        }
};
}
#endif

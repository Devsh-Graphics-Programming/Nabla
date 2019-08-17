#ifndef __IRR_DYNAMIC_ARRAY_H_INCLUDED__
#define __IRR_DYNAMIC_ARRAY_H_INCLUDED__

#include "irr/core/Types.h"//for core::allocator

namespace irr { namespace core
{

template<typename T, class allocator = core::allocator<T>>
class dynamic_array
{
public:
    using allocator_type = allocator;
    using value_type = T;
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;
    using iterator = T*;
    using const_iterator = const T*;

protected:
    size_t item_count;
    allocator alctr;
    pointer contents;

public:
    dynamic_array(size_t _length, const allocator& _alctr = allocator()) : item_count(_length), alctr(_alctr), contents(alctr.allocate(item_count))
    {
        for (size_t i = 0ull; i < item_count; ++i)
            std::allocator_traits<allocator>::construct(alctr, contents+i);
    }
    dynamic_array(size_t _length, const T& _val, const allocator& _alctr = allocator()) : item_count(_length), alctr(_alctr), contents(alctr.allocate(item_count))
    {
        for (size_t i = 0ull; i < item_count; ++i)
            std::allocator_traits<allocator>::construct(alctr, contents+i, _val);
    }
    dynamic_array(std::initializer_list<T> _contents, const allocator& _alctr = allocator()) : item_count(_contents.size()), alctr(_alctr), contents(alctr.allocate(item_count))
    {
        for (size_t i = 0ull; i < item_count; ++i)
            std::allocator_traits<allocator>::construct(alctr, contents+i, *(_contents.begin()+i));
    }

    virtual ~dynamic_array()
    {
        for (size_t i = 0ull; i < item_count; ++i)
            std::allocator_traits<allocator>::destroy(alctr, contents+i);
        if (contents)
            alctr.deallocate(contents, item_count);
    }

    bool operator!=(const dynamic_array<T, allocator>& _other) const
    {
        if (size() != _other.size())
            return true;
        for (size_t i = 0u; i < size(); ++i)
            if ((*this)[i] != _other[i])
                return true;
        return false;
    }
    bool operator==(const dynamic_array<T, allocator>& _other) const
    {
        return !((*this) != _other);
    }

    iterator begin() noexcept { return contents; }
    const_iterator begin() const noexcept { return contents; }
    iterator end() noexcept { return contents+item_count; }
    const_iterator end() const noexcept { return contents+item_count; }
    const_iterator cend() const noexcept { return contents+item_count; }
    const_iterator cbegin() const noexcept { return contents; }

    size_t size() const noexcept { return item_count; }
    bool empty() const noexcept { return !size(); }

    const T& operator[](size_t ix) const noexcept { return contents[ix]; }
    T& operator[](size_t ix) noexcept { return contents[ix]; }

    T& front() noexcept { return *begin(); }
    const T& front() const noexcept { return *begin(); }
    T& back() noexcept { return *(end()-1); }
    const T& back() const noexcept { return *(end()-1); }
    pointer data() noexcept { return contents; }
    const_pointer data() const noexcept { return contents; }
};

}}

#endif
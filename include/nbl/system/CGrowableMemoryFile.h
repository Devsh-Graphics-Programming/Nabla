// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_SYSTEM_C_GROWABLE_MEMORY_FILE_H_INCLUDED_
#define _NBL_SYSTEM_C_GROWABLE_MEMORY_FILE_H_INCLUDED_

#include "nbl/system/IFile.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace nbl::system
{

namespace impl
{

struct CImmediateFileIoResultSetter final : ISystem::IFutureManipulator
{
    using ISystem::IFutureManipulator::set_result;
};

struct CNoopMutex
{
    inline void lock() {}
    inline void unlock() {}
};

class CGrowableMemoryFileStorage
{
    public:
        constexpr static inline size_t InitialGrowthBytes = 1ull << 20; // 1 MiB

        inline size_t size() const
        {
            return m_storage.size();
        }

        inline size_t capacity() const
        {
            return m_storage.capacity();
        }

        inline void reserve(const size_t reservedSize)
        {
            m_storage.reserve(reservedSize);
        }

        inline void clear()
        {
            m_storage.clear();
        }

        inline const std::byte* data() const
        {
            return m_storage.empty() ? nullptr : m_storage.data();
        }

        inline std::byte* data()
        {
            return m_storage.empty() ? nullptr : m_storage.data();
        }

        inline std::vector<std::byte> copyData() const
        {
            return m_storage;
        }

        inline size_t read(void* const buffer, const size_t offset, const size_t sizeToRead) const
        {
            if (offset >= m_storage.size())
                return 0ull;

            const size_t clampedRead = std::min(sizeToRead, m_storage.size() - offset);
            std::memcpy(buffer, m_storage.data() + offset, clampedRead);
            return clampedRead;
        }

        inline size_t write(const void* const buffer, const size_t offset, const size_t sizeToWrite)
        {
            const size_t requiredSize = offset + sizeToWrite;
            if (requiredSize > m_storage.capacity())
                reserve(growCapacity(requiredSize));
            if (requiredSize > m_storage.size())
                m_storage.resize(requiredSize);
            std::memcpy(m_storage.data() + offset, buffer, sizeToWrite);
            return sizeToWrite;
        }

    private:
        inline size_t growCapacity(const size_t requiredSize) const
        {
            size_t currentCapacity = m_storage.capacity();
            if (currentCapacity == 0ull)
                currentCapacity = InitialGrowthBytes;

            size_t nextCapacity = currentCapacity;
            while (nextCapacity < requiredSize)
            {
                const size_t growth = std::max(nextCapacity, InitialGrowthBytes);
                if (nextCapacity > std::numeric_limits<size_t>::max() - growth)
                    return requiredSize;
                nextCapacity += growth;
            }
            return nextCapacity;
        }

        std::vector<std::byte> m_storage;
};

template<typename MutexType>
class IGrowableMemoryFile : public IFile
{
    protected:
        using mutex_t = MutexType;

        inline explicit IGrowableMemoryFile(path&& filename, const size_t reservedSize = 0ull, const time_point_t initialModified = std::chrono::utc_clock::now())
            : IFile(std::move(filename), core::bitflag<E_CREATE_FLAGS>(E_CREATE_FLAGS::ECF_READ_WRITE), initialModified)
        {
            reserve(reservedSize);
        }

        template<typename Fn>
        inline decltype(auto) withLockedStorage(Fn&& fn)
        {
            std::lock_guard<mutex_t> lock(m_mutex);
            return std::forward<Fn>(fn)(m_storage);
        }

        template<typename Fn>
        inline decltype(auto) withLockedStorage(Fn&& fn) const
        {
            std::lock_guard<mutex_t> lock(m_mutex);
            return std::forward<Fn>(fn)(m_storage);
        }

    public:
        inline size_t getSize() const override
        {
            return withLockedStorage([](const CGrowableMemoryFileStorage& storage) {
                return storage.size();
            });
        }

        inline size_t capacity() const
        {
            return withLockedStorage([](const CGrowableMemoryFileStorage& storage) {
                return storage.capacity();
            });
        }

        //! Optional capacity hint for callers that can estimate the final serialized size.
        /** The internal storage already uses an adaptive growth policy, so this is only a performance hint. */
        inline void reserve(const size_t reservedSize)
        {
            withLockedStorage([reservedSize](CGrowableMemoryFileStorage& storage) {
                storage.reserve(reservedSize);
            });
        }

        inline void clear()
        {
            withLockedStorage([](CGrowableMemoryFileStorage& storage) {
                storage.clear();
            });
            setLastWriteTime();
        }

        inline std::vector<std::byte> copyData() const
        {
            return withLockedStorage([](const CGrowableMemoryFileStorage& storage) {
                return storage.copyData();
            });
        }

    protected:
        inline void* getMappedPointer_impl() override
        {
            return nullptr;
        }

        inline const void* getMappedPointer_impl() const override
        {
            return nullptr;
        }

        inline void unmappedRead(ISystem::future_t<size_t>& fut, void* buffer, size_t offset, size_t sizeToRead) override
        {
            static const CImmediateFileIoResultSetter resultSetter = {};
            const size_t processed = withLockedStorage([buffer, offset, sizeToRead](const CGrowableMemoryFileStorage& storage) {
                return storage.read(buffer, offset, sizeToRead);
            });
            resultSetter.set_result(fut, processed);
        }

        inline void unmappedWrite(ISystem::future_t<size_t>& fut, const void* buffer, size_t offset, size_t sizeToWrite) override
        {
            static const CImmediateFileIoResultSetter resultSetter = {};
            const size_t processed = withLockedStorage([buffer, offset, sizeToWrite](CGrowableMemoryFileStorage& storage) {
                return storage.write(buffer, offset, sizeToWrite);
            });
            resultSetter.set_result(fut, processed);
        }

    private:
        mutable mutex_t m_mutex;
        CGrowableMemoryFileStorage m_storage;
};

}

//! A lightweight growable in-memory implementation of `system::IFile`.
/**
    This class stores file contents in a dynamically growing byte buffer while preserving the regular
    Nabla file-oriented API. It is useful in flows that want `IFile*` interoperability without
    forcing an obligatory round-trip through the host filesystem.

    Representative use-cases include:
    - serialization roundtrip validation
    - benchmark or profiling harnesses that want to separate codec work from storage latency
    - tool pipelines that need a temporary serialized representation but do not require a persistent file

    The object grows on demand during writes and can later be consumed by APIs that read from
    `system::IFile*`, for example `IAssetManager::getAsset(system::IFile*, supposedFilename, ...)`.

    Allocation policy:
    - storage growth is handled internally
    - capacity expansion is geometric rather than exact-size-only
    - the first growth step uses a minimum allocation quantum of `1 MiB`
    - callers may still provide an explicit `reserve(...)` hint if they already know the likely output size

    This keeps the common case simple for callers while reducing the amount of repeated reallocation
    and copying that would otherwise happen during long sequential write streams.

    Important notes:
    - reads and writes are positional and operate on the current logical size
    - `getMappedPointer()` intentionally returns `nullptr`
      The storage is growable, so exposing a stable mapped pointer would be misleading
    - this class is not thread-safe
      Concurrent read, write, reserve, clear, or direct `data()` access on the same object requires external synchronization
*/
class CGrowableMemoryFile final : public impl::IGrowableMemoryFile<impl::CNoopMutex>
{
        using base_t = impl::IGrowableMemoryFile<impl::CNoopMutex>;

    public:
        using base_t::capacity;
        using base_t::clear;
        using base_t::copyData;
        using base_t::reserve;

        inline explicit CGrowableMemoryFile(path&& filename, const size_t reservedSize = 0ull, const time_point_t initialModified = std::chrono::utc_clock::now())
            : base_t(std::move(filename), reservedSize, initialModified)
        {
        }

        inline const std::byte* data() const
        {
            return withLockedStorage([](const impl::CGrowableMemoryFileStorage& storage) {
                return storage.data();
            });
        }

        inline std::byte* data()
        {
            return withLockedStorage([](impl::CGrowableMemoryFileStorage& storage) {
                return storage.data();
            });
        }
};

//! A synchronized growable in-memory implementation of `system::IFile`.
/**
    This variant serializes internal operations with a mutex. It is intended for cases where the same
    memory-backed file object may be touched from multiple threads and external synchronization is not
    desirable or not available.

    The synchronized variant intentionally does not expose raw `data()` accessors. A raw pointer would
    not carry any lifetime relationship to the internal lock and would therefore invite accidental use
    after another thread mutates or reallocates the storage. Callers that need to inspect the contents
    can either:
    - take a snapshot with `copyData()`
    - use `withLockedData(...)` and keep any pointer or span-like view strictly inside the callback
*/
class CSynchronizedGrowableMemoryFile final : public impl::IGrowableMemoryFile<std::mutex>
{
        using base_t = impl::IGrowableMemoryFile<std::mutex>;

    public:
        using base_t::capacity;
        using base_t::clear;
        using base_t::copyData;
        using base_t::reserve;

        inline explicit CSynchronizedGrowableMemoryFile(path&& filename, const size_t reservedSize = 0ull, const time_point_t initialModified = std::chrono::utc_clock::now())
            : base_t(std::move(filename), reservedSize, initialModified)
        {
        }

        template<typename Fn>
        inline decltype(auto) withLockedData(Fn&& fn)
        {
            return withLockedStorage([&fn](impl::CGrowableMemoryFileStorage& storage) -> decltype(auto) {
                return std::forward<Fn>(fn)(storage.data(), storage.size());
            });
        }

        template<typename Fn>
        inline decltype(auto) withLockedData(Fn&& fn) const
        {
            return withLockedStorage([&fn](const impl::CGrowableMemoryFileStorage& storage) -> decltype(auto) {
                return std::forward<Fn>(fn)(storage.data(), storage.size());
            });
        }
};

}

#endif

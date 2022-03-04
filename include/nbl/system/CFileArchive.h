// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_SYSTEM_C_FILE_ARCHIVE_H_INCLUDED_
#define _NBL_SYSTEM_C_FILE_ARCHIVE_H_INCLUDED_

#include "nbl/system/IFileArchive.h"
#include "nbl/system/CFileView.h"
#include "nbl/system/IFileViewAllocator.h"

namespace nbl::system
{

// these files are cached and created "on demand", and their storage is pooled
template<typename T>
class CInnerArchiveFile : public CFileView<T>
{
		std::atomic_flag* alive;
	public:
		CInnerArchiveFile(CFileView<T>* arch, std::atomic_flag* _flag) : CFileView<T>(std::move(*arch)), alive(_flag)
		{
		}
		~CInnerArchiveFile() = default;

		static void* operator new(size_t size) noexcept
		{
			assert(false);
			return nullptr;
		}
		static void* operator new[](size_t size) noexcept
		{
			assert(false);
			return nullptr;
		}
		static void* operator new(size_t size, void* ptr, std::atomic_flag* alive)
		{
			alive->test_and_set();
			return ::operator new(size, ptr);
		}
		static void operator delete(void* ptr) noexcept
		{
			static_cast<CInnerArchiveFile*>(ptr)->alive->clear();
			static_cast<CInnerArchiveFile*>(ptr)->alive->notify_one();
		}
		static void  operator delete[](void* ptr) noexcept
		{
			assert(false);
		}
};


//!
class CFileArchive : public IFileArchive
{
		static inline constexpr size_t SIZEOF_INNER_ARCHIVE_FILE = std::max(sizeof(CInnerArchiveFile<CPlainHeapAllocator>), sizeof(CInnerArchiveFile<VirtualMemoryAllocator>));
		static inline constexpr size_t ALIGNOF_INNER_ARCHIVE_FILE = std::max(alignof(CInnerArchiveFile<CPlainHeapAllocator>), alignof(CInnerArchiveFile<VirtualMemoryAllocator>));

	public:
		core::smart_refctd_ptr<IFile> readFile(const SOpenFileParams& params)
		{
			auto index = getIndexByPath(params.filename);
			if (index == -1) return nullptr;
			switch (listAssets()[index].allocatorType)
			{
			case EAT_NULL:
				return getFile_impl<CNullAllocator>(params, index);
				break;
			case EAT_MALLOC:
				return getFile_impl<CPlainHeapAllocator>(params, index);
				break;
			case EAT_VIRTUAL_ALLOC:
				return getFile_impl<VirtualMemoryAllocator>(params, index);
				break;
			}
			assert(false);
			return nullptr;
		}
		virtual core::smart_refctd_ptr<IFile> readFile_impl(const SOpenFileParams& params) = 0;
		int32_t getIndexByPath(const system::path& p)
		{
			for (int i = 0; i < m_files.size(); ++i)
			{
				if (p == m_files[i].fullName) return i;
			}
			return -1;
		}
		template<class Allocator>
		core::smart_refctd_ptr<CInnerArchiveFile<Allocator>> getFile_impl(const SOpenFileParams& params, const uint32_t index)
		{
			std::unique_lock lock(fileMutex);

			auto* file = reinterpret_cast<CInnerArchiveFile<Allocator>*>(m_filesBuffer + index * SIZEOF_INNER_ARCHIVE_FILE);
			//  intentionally calling grab() on maybe-not-existing object
			const auto oldRefcount = file->grab();

			if (oldRefcount == 0) // need to construct
			{
				m_fileFlags[index].wait(true); //what should the param of wait be?
				new (file, &m_fileFlags[index]) CInnerArchiveFile<Allocator>(static_cast<CFileView<Allocator>*>(readFile_impl(params).get()), &m_fileFlags[index]);
			}
			return core::smart_refctd_ptr<CInnerArchiveFile<Allocator>>(file, core::dont_grab);
		}

	protected:
		CFileArchive(core::smart_refctd_ptr<IFile>&& file, system::logger_opt_smart_ptr&& logger) : IFileArchive(std::move(file),std::move(logger)) {}
		~CFileArchive()
		{ 
			_NBL_ALIGNED_FREE(m_filesBuffer);
			_NBL_ALIGNED_FREE(m_fileFlags);
		}

		void setFlagsVectorSize(size_t fileCount);

		std::mutex fileMutex;
		std::atomic_flag* m_fileFlags = nullptr;
		std::byte* m_filesBuffer = nullptr;
};


} // end namespace nbl::system

#endif


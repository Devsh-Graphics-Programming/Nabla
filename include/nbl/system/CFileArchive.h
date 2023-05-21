// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifndef _NBL_SYSTEM_C_FILE_ARCHIVE_H_INCLUDED_
#define _NBL_SYSTEM_C_FILE_ARCHIVE_H_INCLUDED_


#include "nbl/system/IFileArchive.h"
#include "nbl/system/CFileView.h"
#include "nbl/system/IFileViewAllocator.h"

#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/CFileViewAPKAllocator.h"
#endif

namespace nbl::system
{

// these files are cached and created "on demand", and their storage is pooled
template<typename T>
class CInnerArchiveFile : public CFileView<T>
{
		std::atomic_flag* alive;
	public:
		template<typename... Args>
		CInnerArchiveFile(std::atomic_flag* _flag, Args&&... args) : CFileView<T>(std::forward<Args>(args)...), alive(_flag)
		{
		}
		~CInnerArchiveFile() = default;

		static void* operator new(size_t size) noexcept
		{
			assert(false);
			exit(-0x45);
			return nullptr;
		}
		static void* operator new[](size_t size) noexcept
		{
			assert(false);
			exit(-0x45);
			return nullptr;
		}
		static void* operator new(size_t size, void* ptr, std::atomic_flag* alive) noexcept
		{
			alive->test_and_set();
			return ::operator new(size,ptr);
		}

		//
		static void operator delete(void* ptr) noexcept
		{
			static_cast<CInnerArchiveFile*>(ptr)->alive->clear();
			static_cast<CInnerArchiveFile*>(ptr)->alive->notify_one();
		}
		static void  operator delete[](void* ptr) noexcept
		{
			assert(false);
			exit(-0x45);
		}

		// make compiler shut up about initizaliation throwing exceptions
		static void operator delete(void* dummy, void* ptr, std::atomic_flag* alive) noexcept
		{
			::operator delete(ptr);
		}
};


//!
class CFileArchive : public IFileArchive
{
		static inline constexpr size_t SIZEOF_INNER_ARCHIVE_FILE = std::max(sizeof(CInnerArchiveFile<CPlainHeapAllocator>), sizeof(CInnerArchiveFile<VirtualMemoryAllocator>));
		static inline constexpr size_t ALIGNOF_INNER_ARCHIVE_FILE = std::max(alignof(CInnerArchiveFile<CPlainHeapAllocator>), alignof(CInnerArchiveFile<VirtualMemoryAllocator>));

	public:
		inline core::smart_refctd_ptr<IFile> getFile(const path& pathRelativeToArchive, const std::string_view& password) override
		{
			const auto* item = getItemFromPath(pathRelativeToArchive);
			if (!item)
				return nullptr;
			
			switch (item->allocatorType)
			{
				case EAT_NULL:
					return getFile_impl<CNullAllocator>(item);
					break;
				case EAT_MALLOC:
					return getFile_impl<CPlainHeapAllocator>(item);
					break;
				case EAT_VIRTUAL_ALLOC:
					return getFile_impl<VirtualMemoryAllocator>(item);
					break;
				case EAT_APK_ALLOCATOR:
					#ifdef _NBL_PLATFORM_ANDROID_
					return getFile_impl<CFileViewAPKAllocator>(item);
					#else
					assert(false);
					#endif
					break;
				default: // directory or something
					break;
			}
			return nullptr;
		}

	protected:
		CFileArchive(path&& _defaultAbsolutePath, system::logger_opt_smart_ptr&& logger, core::vector<SFileList::SEntry> _items) :
			IFileArchive(std::move(_defaultAbsolutePath),std::move(logger))
		{
			//should _items be rvalue reference?
			auto itemsSharedPtr = std::make_shared<core::vector<IFileArchive::SFileList::SEntry>>(std::move(_items));
			std::sort(itemsSharedPtr->begin(), itemsSharedPtr->end());
			m_items.store(itemsSharedPtr);

			const auto fileCount = itemsSharedPtr->size();
			m_filesBuffer = (std::byte*)_NBL_ALIGNED_MALLOC(fileCount*SIZEOF_INNER_ARCHIVE_FILE, ALIGNOF_INNER_ARCHIVE_FILE);
			m_fileFlags = (std::atomic_flag*)_NBL_ALIGNED_MALLOC(fileCount*sizeof(std::atomic_flag), alignof(std::atomic_flag));
			for (size_t i=0u; i<fileCount; i++)
				m_fileFlags[i].clear();
			memset(m_filesBuffer,0,fileCount*SIZEOF_INNER_ARCHIVE_FILE);
		}
		~CFileArchive()
		{ 
			_NBL_ALIGNED_FREE(m_filesBuffer);
			_NBL_ALIGNED_FREE(m_fileFlags);
		}
		
		template<class Allocator>
		inline core::smart_refctd_ptr<CInnerArchiveFile<Allocator>> getFile_impl(const IFileArchive::SFileList::SEntry* item)
		{
			auto* file = reinterpret_cast<CInnerArchiveFile<Allocator>*>(m_filesBuffer+item->ID*SIZEOF_INNER_ARCHIVE_FILE);
			// NOTE: Intentionally calling grab() on maybe-not-existing object!
			const auto oldRefcount = file->grab();

			if (oldRefcount==0) // need to construct (previous refcount was 0)
			{
				const auto fileBuffer = getFileBuffer(item);
				// Might have barged inbetween a refctr drop and finish of a destructor + delete,
				// need to wait for the "alive" flag to become `false` which tells us `operator delete` has finished.
				m_fileFlags[item->ID].wait(true);
				// coast is clear, do placement new
				new (file, &m_fileFlags[item->ID]) CInnerArchiveFile<Allocator>(
					m_fileFlags+item->ID,
					getDefaultAbsolutePath()/item->pathRelativeToArchive,
					IFile::ECF_READ, // TODO: stay like this until we allow write access to archived files
					fileBuffer.buffer,
					fileBuffer.size,
					Allocator(fileBuffer.allocatorState) // no archive uses stateful allocators yet
				);
			}
			// don't grab because we've already grabbed
			return core::smart_refctd_ptr<CInnerArchiveFile<Allocator>>(file,core::dont_grab);
		}

		// this function will return a buffer that needs to be deallocated with an allocator matching `item->allocatorType`
		struct file_buffer_t
		{
			void* buffer;
			size_t size;
			void* allocatorState;
		};
		virtual file_buffer_t getFileBuffer(const IFileArchive::SFileList::SEntry* item) = 0;

		std::atomic_flag* m_fileFlags = nullptr;
		std::byte* m_filesBuffer = nullptr;
};


} // end namespace nbl::system

#endif


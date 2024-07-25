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

	protected:
		inline CFileArchive(path&& _defaultAbsolutePath, system::logger_opt_smart_ptr&& logger, std::shared_ptr<core::vector<SFileList::SEntry>> _items) :
			IFileArchive(std::move(_defaultAbsolutePath),std::move(logger))
		{
			setItemList(_items);

			const auto fileCount = _items->size();
			m_filesBuffer = (std::byte*)_NBL_ALIGNED_MALLOC(fileCount*SIZEOF_INNER_ARCHIVE_FILE, ALIGNOF_INNER_ARCHIVE_FILE);
			m_fileFlags = (std::atomic_flag*)_NBL_ALIGNED_MALLOC(fileCount*sizeof(std::atomic_flag), alignof(std::atomic_flag));
			for (size_t i=0u; i<fileCount; i++)
				m_fileFlags[i].clear();
			memset(m_filesBuffer,0,fileCount*SIZEOF_INNER_ARCHIVE_FILE);
		}
		virtual inline ~CFileArchive()
		{ 
			_NBL_ALIGNED_FREE(m_filesBuffer);
			_NBL_ALIGNED_FREE(m_fileFlags);
		}
		
		inline core::smart_refctd_ptr<IFile> getFile_impl(const SFileList::found_t& found, const core::bitflag<IFile::E_CREATE_FLAGS> flags, const std::string_view& password) override
		{
			switch (found->allocatorType)
			{
				case EAT_NULL:
					return getFile_impl<CNullAllocator>(found,flags);
					break;
				case EAT_MALLOC:
					return getFile_impl<CPlainHeapAllocator>(found,flags);
					break;
				case EAT_VIRTUAL_ALLOC:
					return getFile_impl<VirtualMemoryAllocator>(found,flags);
					break;
				case EAT_APK_ALLOCATOR:
					#ifdef _NBL_PLATFORM_ANDROID_
					return getFile_impl<CFileViewAPKAllocator>(found,flags);
					#else
					assert(false);
					#endif
					break;
				default: // directory or something
					break;
			}
			return nullptr;
		}
		
		template<class Allocator>
		inline core::smart_refctd_ptr<CInnerArchiveFile<Allocator>> getFile_impl(const SFileList::found_t& found, core::bitflag<IFile::E_CREATE_FLAGS> flags)
		{
			// TODO: figure out a new system of cached allocations which can handle files being added/removed from an archive,
			// which will also allow for changing the flags that a File View is created with.
			if (flags.hasFlags(IFile::ECF_MAPPABLE))
			{
				m_logger.log("Overriding file flags for %s, creating it as mappable anyway.",ILogger::ELL_INFO,found->pathRelativeToArchive.c_str());
				flags |= IFile::ECF_MAPPABLE;
			}
			// IFileArchive should have already checked for this, stay like this until we allow write access to archived files
			assert(!flags.hasFlags(IFile::ECF_WRITE));

			auto* file = reinterpret_cast<CInnerArchiveFile<Allocator>*>(m_filesBuffer+found->ID*SIZEOF_INNER_ARCHIVE_FILE);
			// NOTE: Intentionally calling grab() on maybe-not-existing object!
			const auto oldRefcount = file->grab();

			if (oldRefcount==0) // need to construct (previous refcount was 0)
			{
				const auto fileBuffer = getFileBuffer(found);
				// Might have barged inbetween a refctr drop and finish of a destructor + delete,
				// need to wait for the "alive" flag to become `false` which tells us `operator delete` has finished.
				m_fileFlags[found->ID].wait(true);
				// coast is clear, do placement new
				new (file, &m_fileFlags[found->ID]) CInnerArchiveFile<Allocator>(
					m_fileFlags+found->ID,
					getDefaultAbsolutePath()/found->pathRelativeToArchive,
					flags,
					fileBuffer.initialModified,
					fileBuffer.buffer,
					fileBuffer.size,
					Allocator(fileBuffer.allocatorState) // no archive uses stateful allocators yet
				);
			}
			// don't grab because we've already grabbed
			return core::smart_refctd_ptr<CInnerArchiveFile<Allocator>>(file,core::dont_grab);
		}

		// this function will return a buffer that needs to be deallocated with an allocator matching `found->allocatorType`
		struct file_buffer_t
		{
			void* buffer;
			size_t size;
			void* allocatorState;
			// TODO: Implement this !!!
			IFileBase::time_point_t initialModified = std::chrono::utc_clock::now();
		};
		virtual file_buffer_t getFileBuffer(const SFileList::found_t& found) = 0;

		std::atomic_flag* m_fileFlags = nullptr;
		std::byte* m_filesBuffer = nullptr;
};


} // end namespace nbl::system

#endif


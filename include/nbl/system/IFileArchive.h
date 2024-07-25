// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_SYSTEM_I_FILE_ARCHIVE_H_INCLUDED_
#define _NBL_SYSTEM_I_FILE_ARCHIVE_H_INCLUDED_

#include "nbl/core/SRange.h"

#include "nbl/system/path.h"
#include "nbl/system/ILogger.h"
#include "nbl/system/IFileBase.h"

#include <string_view>
#include <algorithm>
#include <span>

namespace nbl::system
{

class IFile;

//! The FileArchive manages archives and provides access to files inside them.
class IFileArchive : public core::IReferenceCounted
{
	public:
		enum E_ALLOCATOR_TYPE
		{
			EAT_NONE = 0,
			EAT_NULL, // read directly from archive's underlying mapped file
			EAT_VIRTUAL_ALLOC, // decompress to RAM (with sparse paging)
			EAT_APK_ALLOCATOR, // specialization to be able to call `AAsset_close`
			EAT_MALLOC // decompress to RAM
		};
		//! An entry in a list of items, can be a folder or a file.
		class SFileList
		{
			public:
				struct SEntry
				{
					//same stuff as `SListEntry` right now
					//! The name of the file including the path relative to archive root
					system::path pathRelativeToArchive;

					//! The size of the file in bytes
					size_t size;

					//! FileOffset inside an archive
					size_t offset;

					//! The ID of the file in an archive, it maps it to a memory pool entry for CFileView
					uint32_t ID;

					// `EAT_NONE` for directories
					IFileArchive::E_ALLOCATOR_TYPE allocatorType;

					//! The == operator is provided so that CFileList can slowly search the list!
					inline bool operator==(const struct SEntry& other) const
					{
						return pathRelativeToArchive.string() == other.pathRelativeToArchive.string();
					}

					//! The < operator is provided so that CFileList can sort and quickly search the list.
					inline bool operator<(const struct SEntry& other) const
					{
						return pathRelativeToArchive < other.pathRelativeToArchive;
					}
				};
				using refctd_storage_t = std::shared_ptr<const core::vector<SEntry>>;

				class found_t final
				{
						refctd_storage_t m_backingStorage = nullptr;

					public:
						using type = refctd_storage_t::element_type::const_pointer;

						inline found_t() = default;
						inline found_t(refctd_storage_t&& _storage, type _iter) :	m_backingStorage(_storage), m_iter(_iter) {}

						explicit inline operator bool() const {return m_iter;}

						inline const SEntry& operator*() const {return *m_iter;}
						inline const SEntry* operator->() const {return m_iter;}

						type m_iter = nullptr;
				};

				/*deprecated*/ using range_t = std::span<const SEntry>;
				using span_t = std::span<const SEntry>;
				inline operator span_t() const {return m_span;}

				inline SFileList(const SFileList&) = default;
				inline SFileList(SFileList&&) = default;
				inline SFileList& operator=(const SFileList&) = default;
				inline SFileList& operator=(SFileList&&) = default;

			private:
				// default ctor full range
				inline SFileList(refctd_storage_t _data) : m_data(_data), m_span(m_data->data(),m_data->data()+m_data->size()) {}

				friend class IFileArchive;
				refctd_storage_t m_data;
				span_t m_span;
		};

		//
		virtual inline SFileList listAssets() const
		{
			return { m_items.load() };
		}

		// List all files and directories in a specific dir of the archive
		NBL_API2 SFileList listAssets(path pathRelativeToArchive) const;

		//
		inline core::smart_refctd_ptr<IFile> getFile(const path& pathRelativeToArchive, const core::bitflag<IFileBase::E_CREATE_FLAGS> flags, const std::string_view& password)
		{
			const auto item = getItemFromPath(pathRelativeToArchive);
			if (!item)
				return nullptr;

			if (flags.hasFlags(IFileBase::ECF_WRITE))
			{
				m_logger.log("Cannot open file %s with WRITE flag, we don't support writing to archives yet!",ILogger::ELL_ERROR,pathRelativeToArchive.c_str());
				return nullptr;
			}
			return getFile_impl(item,flags,password);
		}

		//
		inline const path& getDefaultAbsolutePath() const {return m_defaultAbsolutePath;}

	protected:
		inline IFileArchive(path&& _defaultAbsolutePath, system::logger_opt_smart_ptr&& logger) :
			m_defaultAbsolutePath(std::move(_defaultAbsolutePath.make_preferred())), m_logger(std::move(logger)) {}
		virtual inline ~IFileArchive() = default;

		//
		virtual core::smart_refctd_ptr<IFile> getFile_impl(const SFileList::found_t& found, const core::bitflag<IFileBase::E_CREATE_FLAGS> flags, const std::string_view& password) = 0;

		inline const SFileList::found_t getItemFromPath(const system::path& pathRelativeToArchive) const
		{
            const SFileList::SEntry itemToFind = { pathRelativeToArchive };
			// calling `listAssets` makes sure any "update list" overload can kick in
			auto items = listAssets();
			const auto span = SFileList::span_t(items);
			const auto found = std::lower_bound(span.begin(),span.end(),itemToFind);
			if (found==span.end() || found->pathRelativeToArchive!=pathRelativeToArchive)
				return {};
			return SFileList::found_t(std::move(items.m_data),&(*found));
		}

		const path m_defaultAbsolutePath;
		system::logger_opt_smart_ptr m_logger;

		inline void setItemList(std::shared_ptr<core::vector<SFileList::SEntry>> _items) const
		{	
			std::sort(_items->begin(),_items->end());
			m_items.store(_items);
		}

	private:
		// files and directories
		mutable std::atomic<SFileList::refctd_storage_t> m_items;
};


class IArchiveLoader : public core::IReferenceCounted
{
	public:
		IArchiveLoader(system::logger_opt_smart_ptr&& logger) : m_logger(std::move(logger)) {}

		//! Check if the file might be loaded by this class
		/** This check may look into the file.
		\param file File handle to check.
		\return True if file seems to be loadable. */
		virtual bool isALoadableFileFormat(IFile* file) const = 0;

		//! Returns an array of string literals terminated by nullptr
		virtual const char** getAssociatedFileExtensions() const = 0;

		//! Creates an archive from the file
		/** \param file File handle to use.
		\return Pointer to newly created archive, or 0 upon error. */
		core::smart_refctd_ptr<IFileArchive> createArchive(core::smart_refctd_ptr<IFile>&& file, const std::string_view& password = "") const;

	protected:
		virtual core::smart_refctd_ptr<IFileArchive> createArchive_impl(core::smart_refctd_ptr<IFile>&& file, const std::string_view& password) const = 0;

		system::logger_opt_smart_ptr m_logger;
};

} // end namespace nbl::system

#endif


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
		struct SListEntry
		{
			//! The name of the file including the path relative to archive root
			system::path pathRelativeToArchive;

			//! The size of the file in bytes
			size_t size;

			//! FileOffset inside an archive
			size_t offset;

			//! The ID of the file in an archive, it maps it to a memory pool entry for CFileView
			uint32_t ID;

			// `EAT_NONE` for directories
			E_ALLOCATOR_TYPE allocatorType;

			//! The == operator is provided so that CFileList can slowly search the list!
			inline bool operator==(const struct SListEntry& other) const
			{
				return pathRelativeToArchive.string()==other.pathRelativeToArchive.string();
			}

			//! The < operator is provided so that CFileList can sort and quickly search the list.
			inline bool operator<(const struct SListEntry& other) const
			{
				return pathRelativeToArchive<other.pathRelativeToArchive;
			}
		};

		//
		virtual core::SRange<const SListEntry> listAssets() const {return {m_items.data(),m_items.data()+m_items.size()};}

		// List all files and directories in a specific dir of the archive
		virtual core::SRange<const SListEntry> listAssets(const path& asset_path) const;

		//
		virtual core::smart_refctd_ptr<IFile> getFile(const path& pathRelativeToArchive, const std::string_view& password) = 0;

		//
		const path& getDefaultAbsolutePath() const {return m_defaultAbsolutePath;}

	protected:
		IFileArchive(path&& _defaultAbsolutePath, system::logger_opt_smart_ptr&& logger) :
			m_defaultAbsolutePath(std::move(_defaultAbsolutePath)), m_logger(std::move(logger)) {}

		inline const SListEntry* getItemFromPath(const system::path& pathRelativeToArchive) const
		{
            const IFileArchive::SListEntry itemToFind = { pathRelativeToArchive };
			const auto found = std::lower_bound(m_items.begin(),m_items.end(),itemToFind);
			if (found==m_items.end() || found->pathRelativeToArchive!=pathRelativeToArchive)
				return nullptr;
			return &(*found);
		}

		mutable std::mutex itemMutex; // TODO: update to readers writer lock
		path m_defaultAbsolutePath;
		// files and directories
		mutable core::vector<SListEntry> m_items;
		//
		system::logger_opt_smart_ptr m_logger;
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


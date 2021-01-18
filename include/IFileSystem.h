// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_FILE_SYSTEM_H_INCLUDED__
#define __NBL_I_FILE_SYSTEM_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "IFileArchive.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/core/core.h"
#include "nbl/asset/compile_config.h"

#ifdef _NBL_EMBED_BUILTIN_RESOURCES_
#include "nbl/builtin/common.h"
#endif
namespace nbl
{
namespace video
{
	class IVideoDriver;
} // end namespace video
namespace io
{

class IReadFile;
class IWriteFile;
class IFileList;
class IXMLWriter;


//! The FileSystem manages files and archives and provides access to them.
/** It manages where files are, so that modules which use the the IO do not
need to know where every file is located. A file could be in a .zip-Archive or
as file on disk, using the IFileSystem makes no difference to this. */
class IFileSystem : public virtual core::IReferenceCounted
{
	public:
		//! Opens a file for read access.
		/** \param filename: Name of file to open.
		\return Pointer to the created file interface.
		The returned pointer should be dropped when no longer needed.
		See IReferenceCounted::drop() for more information. */
		virtual IReadFile* createAndOpenFile(const path& filename) =0;

		//! Creates an IReadFile interface for accessing memory like a file.
		/** This allows you to use a pointer to memory where an IReadFile is requested.
		\param memory: A pointer to the start of the file in memory
		\param len: The length of the memory in bytes
		\param fileName: The name given to this file
		\param deleteMemoryWhenDropped: True if the memory should be deleted
		along with the IReadFile when it is dropped.
		\return Pointer to the created file interface. 
		The returned pointer should be dropped when no longer needed.
		See IReferenceCounted::drop() for more information.
		*/
		virtual IReadFile* createMemoryReadFile(const void* contents, size_t len, const io::path& fileName) = 0;

		//! Creates an IReadFile interface for accessing files inside files.
		/** This is useful e.g. for archives.
		\param fileName: The name given to this file
		\param alreadyOpenedFile: Pointer to the enclosing file
		\param pos: Start of the file inside alreadyOpenedFile
		\param areaSize: The length of the file
		\return A pointer to the created file interface.
		The returned pointer should be dropped when no longer needed.
		See IReferenceCounted::drop() for more information.
		*/
		virtual IReadFile* createLimitReadFile(const path& fileName,
				IReadFile* alreadyOpenedFile, const size_t& pos, const size_t& areaSize) =0;

		//! Creates an IWriteFile interface for accessing memory like a file.
		/** This allows you to use a pointer to memory where an IWriteFile is requested.
			You are responsible for allocating enough memory.
		\param memory: A pointer to the start of the file in memory (allocated by you)
		\param len: The length of the memory in bytes
		\param fileName: The name given to this file
		\param deleteMemoryWhenDropped: True if the memory should be deleted
		along with the IWriteFile when it is dropped.
		\return Pointer to the created file interface.
		The returned pointer should be dropped when no longer needed.
		See IReferenceCounted::drop() for more information.
		*/
		virtual IWriteFile* createMemoryWriteFile(size_t len, const io::path& fileName) =0;


		//! Opens a file for write access.
		/** \param filename: Name of file to open.
		\param append: If the file already exist, all write operations are
		appended to the file.
		\return Pointer to the created file interface. 0 is returned, if the
		file could not created or opened for writing.
		The returned pointer should be dropped when no longer needed.
		See IReferenceCounted::drop() for more information. */
		virtual IWriteFile* createAndWriteFile(const path& filename, bool append=false) =0;

		//! Adds an archive to the file system.
		/** After calling this, the Irrlicht Engine will also search and open
		files directly from this archive. This is useful for hiding data from
		the end user, speeding up file access and making it possible to access
		for example Quake3 .pk3 files, which are just renamed .zip files. By
		default Irrlicht supports ZIP, PAK, TAR, PNK, and directories as
		archives. You can provide your own archive types by implementing
		IArchiveLoader and passing an instance to addArchiveLoader.
		Irrlicht supports AES-encrypted zip files, and the advanced compression
		techniques lzma and bzip2.
		\param filename: Filename of the archive to add to the file system.
		\param archiveType: If no specific E_FILE_ARCHIVE_TYPE is selected then
		the type of archive will depend on the extension of the file name. If
		you use a different extension then you can use this parameter to force
		a specific type of archive.
		\param password An optional password, which is used in case of encrypted archives.
		\param retArchive A pointer that will be set to the archive that is added.
		\return True if the archive was added successfully, false if not. */
		virtual bool addFileArchive(const path& filename,
				E_FILE_ARCHIVE_TYPE archiveType=EFAT_UNKNOWN,
				const core::stringc& password="",
				IFileArchive** retArchive=0) =0;

		//! Adds an archive to the file system.
		/** After calling this, the Irrlicht Engine will also search and open
		files directly from this archive. This is useful for hiding data from
		the end user, speeding up file access and making it possible to access
		for example Quake3 .pk3 files, which are just renamed .zip files. By
		default Irrlicht supports ZIP, PAK, TAR, PNK, and directories as
		archives. You can provide your own archive types by implementing
		IArchiveLoader and passing an instance to addArchiveLoader.
		Irrlicht supports AES-encrypted zip files, and the advanced compression
		techniques lzma and bzip2.
		If you want to add a directory as an archive, prefix its name with a
		slash in order to let Irrlicht recognize it as a folder mount (mypath/).
		Using this technique one can build up a search order, because archives
		are read first, and can be used more easily with relative filenames.
		\param file: Archive to add to the file system.
		\param archiveType: If no specific E_FILE_ARCHIVE_TYPE is selected then
		the type of archive will depend on the extension of the file name. If
		you use a different extension then you can use this parameter to force
		a specific type of archive.
		\param password An optional password, which is used in case of encrypted archives.
		\param retArchive A pointer that will be set to the archive that is added.
		\return True if the archive was added successfully, false if not. */
		virtual bool addFileArchive(IReadFile* file,
				E_FILE_ARCHIVE_TYPE archiveType=EFAT_UNKNOWN,
				const core::stringc& password="",
				IFileArchive** retArchive=0) =0;

		//! Adds an archive to the file system.
		/** \param archive: The archive to add to the file system.
		\return True if the archive was added successfully, false if not. */
		virtual bool addFileArchive(IFileArchive* archive) =0;

		//! Get the number of archives currently attached to the file system
		virtual uint32_t getFileArchiveCount() const =0;

		//! Removes an archive from the file system.
		/** This will close the archive and free any file handles, but will not
		close resources which have already been loaded and are now cached, for
		example textures and meshes.
		\param index: The index of the archive to remove
		\return True on success, false on failure */
		virtual bool removeFileArchive(uint32_t index) =0;

		//! Removes an archive from the file system.
		/** This will close the archive and free any file handles, but will not
		close resources which have already been loaded and are now cached, for
		example textures and meshes. Note that a relative filename might be
		interpreted differently on each call, depending on the current working
		directory. In case you want to remove an archive that was added using
		a relative path name, you have to change to the same working directory
		again. This means, that the filename given on creation is not an
		identifier for the archive, but just a usual filename that is used for
		locating the archive to work with.
		\param filename The archive pointed to by the name will be removed
		\return True on success, false on failure */
		virtual bool removeFileArchive(const path& filename) =0;

		//! Removes an archive from the file system.
		/** This will close the archive and free any file handles, but will not
		close resources which have already been loaded and are now cached, for
		example textures and meshes.
		\param archive The archive to remove.
		\return True on success, false on failure */
		virtual bool removeFileArchive(const IFileArchive* archive) =0;

		//! Changes the search order of attached archives.
		/**
		\param sourceIndex: The index of the archive to change
		\param relative: The relative change in position, archives with a lower index are searched first */
		virtual bool moveFileArchive(uint32_t sourceIndex, int32_t relative) =0;

		//! Get the archive at a given index.
		virtual IFileArchive* getFileArchive(uint32_t index) =0;

		//! Adds an external archive loader to the engine.
		/** Use this function to add support for new archive types to the
		engine, for example proprietary or encrypted file storage. */
		virtual void addArchiveLoader(IArchiveLoader* loader) =0;

		//! Gets the number of archive loaders currently added
		virtual uint32_t getArchiveLoaderCount() const = 0;

		//! Retrieve the given archive loader
		/** \param index The index of the loader to retrieve. This parameter is an 0-based
		array index.
		\return A pointer to the specified loader, 0 if the index is incorrect. */
		virtual IArchiveLoader* getArchiveLoader(uint32_t index) const = 0;

		//! Get the current working directory.
		/** \return Current working directory as a string. */
		virtual const path& getWorkingDirectory() =0;

		//! Changes the current working directory.
		/** \param newDirectory: A string specifying the new working directory.
		The string is operating system dependent. Under Windows it has
		the form "<drive>:\<directory>\<sudirectory>\<..>". An example would be: "C:\Windows\"
		\return True if successful, otherwise false. */
		virtual bool changeWorkingDirectoryTo(const path& newDirectory) =0;

		//! Converts a relative path to an absolute (unique) path, resolving symbolic links if required
		/** \param filename Possibly relative file or directory name to query.
		\result Absolute filename which points to the same file. */
		virtual path getAbsolutePath(const path& filename) const =0;

		//! Get the relative filename, relative to the given directory
		virtual path getRelativeFilename(const path& filename, const path& directory) const =0;

		//! Creates a list of files and directories in the current working directory and returns it.
		/** \return a Pointer to the created IFileList is returned. After the list has been used
		it has to be deleted using its IFileList::drop() method.
		See IReferenceCounted::drop() for more information. */
		virtual IFileList* createFileList() =0;

		//! Creates an empty filelist
		/** \return a Pointer to the created IFileList is returned. After the list has been used
		it has to be deleted using its IFileList::drop() method.
		See IReferenceCounted::drop() for more information. */
		virtual IFileList* createEmptyFileList(const io::path& path) =0;

		//! Set the active type of file system.
		virtual EFileSystemType setFileListSystem(EFileSystemType listType) =0;

		//! Determines if a file exists and could be opened.
		/** \param filename is the string identifying the file which should be tested for existence.
		\return True if file exists, and false if it does not exist or an error occured. */
		virtual bool existFile(const path& filename) const =0;


		//! Run-time resource ID, `builtinPath` includes the "nbl/builtin" prefix
		inline core::smart_refctd_ptr<asset::ICPUBuffer> loadBuiltinData(const std::string& builtinPath)
		{
			#ifdef _NBL_EMBED_BUILTIN_RESOURCES_
				std::pair<const uint8_t*, size_t> found = nbl::builtin::get_resource_runtime(builtinPath);
				if (found.first && found.second)
				{
					auto returnValue = core::make_smart_refctd_ptr<asset::ICPUBuffer>(found.second);
					memcpy(returnValue->getPointer(), found.first, returnValue->getSize());
					return returnValue;
				}
				return nullptr;
			#else
				constexpr auto pathPrefix = "nbl/builtin/";
				auto pos = builtinPath.find(pathPrefix);
				std::string path;
				if (pos!=std::string::npos)
					path = builtinResourceDirectory+builtinPath.substr(pos+strlen(pathPrefix));
				else
					path = builtinResourceDirectory+builtinPath;

				auto file = this->createAndOpenFile(path.c_str());
				if (file)
				{
					auto retval = core::make_smart_refctd_ptr<asset::ICPUBuffer>(file->getSize());
					file->read(retval->getPointer(), file->getSize());
					file->drop();
					return retval;
				}
				return nullptr;
			#endif
		}
		//! Compile time resource ID
		template<typename StringUniqueType>
		inline core::smart_refctd_ptr<asset::ICPUBuffer> loadBuiltinData()
		{
			#ifdef _NBL_EMBED_BUILTIN_RESOURCES_
				std::pair<const uint8_t*, size_t> found = nbl::builtin::get_resource<StringUniqueType>();
				if (found.first && found.second)
				{
					auto returnValue = core::make_smart_refctd_ptr<asset::ICPUBuffer>(found.second);
					memcpy(returnValue->getPointer(), found.first, returnValue->getSize());
					return returnValue;
				}
				return nullptr;
			#else
				return loadBuiltinData(StringUniqueType::value);
			#endif
		}


		//! Get the directory a file is located in.
		/** \param filename: The file to get the directory from.
		\return String containing the directory of the file. */
		static inline path getFileDir(const path& filename)
		{
			// find last forward or backslash
			int32_t lastSlash = filename.findLast('/');
			const int32_t lastBackSlash = filename.findLast('\\'); //! Just remove those '\' on Linux
			lastSlash = core::max<int32_t>(lastSlash, lastBackSlash);

			if ((uint32_t)lastSlash < filename.size())
				return filename.subString(0, lastSlash);
			else
				return path(".");
		}

		//! flatten a path and file name for example: "/you/me/../." becomes "/you"
		static inline path flattenFilename(const path& _directory, const path& root="/")
		{
			auto directory(_directory);
			handleBackslashes(&directory);

			io::path dir;
			io::path subdir;

			int32_t lastpos = 0;
			int32_t pos = 0;
			bool lastWasRealDir=false;

			auto process = [&]() -> void
			{
				subdir = directory.subString(lastpos, pos - lastpos + 1);

				if (subdir == "../")
				{
					if (lastWasRealDir)
					{
						deletePathFromPath(dir, 2);
						lastWasRealDir = (dir.size() != 0);
					}
					else
					{
						dir.append(subdir);
						lastWasRealDir = false;
					}
				}
				else if (subdir == "/")
				{
					dir = root;
				}
				else if (subdir != "./")
				{
					dir.append(subdir);
					lastWasRealDir = true;
				}

				lastpos = pos + 1;
			};
			while ((pos = directory.findNext('/', lastpos)) >= 0)
			{
				process();
			}
			if (directory.lastChar() != '/')
			{
				pos = directory.size();
				process();
			}
			return dir;
		}

		//! Get the base part of a filename, i.e. the name without the directory part.
		/** If no directory is prefixed, the full name is returned.
		\param filename: The file to get the basename from
		\param keepExtension True if filename with extension is returned otherwise everything
		after the final '.' is removed as well. */
		static inline path getFileBasename(const path& filename, bool keepExtension=true)
		{
			// find last forward or backslash
			int32_t lastSlash = filename.findLast('/');
			const int32_t lastBackSlash = filename.findLast('\\'); //! Just remove those '\' on Linux
			lastSlash = core::max<int32_t>(lastSlash, lastBackSlash);

			// get number of chars after last dot
			int32_t end = 0;
			if (!keepExtension)
			{
				// take care to search only after last slash to check only for
				// dots in the filename
				end = filename.findLast('.'); //! Use a reverse search with iterators to give a limit on how far back to search
				if (end == -1 || end < lastSlash)
					end=0;
				else
					end = filename.size()-end;
			}

			if ((uint32_t)lastSlash < filename.size())
				return filename.subString(lastSlash+1, filename.size()-lastSlash-1-end);
			else if (end != 0)
				return filename.subString(0, filename.size()-end);
			else
				return filename;
		}

	protected:
		IFileSystem(std::string&& _builtinResourceDirectory) : builtinResourceDirectory(_builtinResourceDirectory) {}

		const std::string builtinResourceDirectory;
};


} // end namespace io
} // end namespace nbl

#endif


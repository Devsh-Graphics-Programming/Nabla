// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"
#include <list>
#include "CFileSystem.h"
#include "CReadFile.h"
#include "IWriteFile.h"
#include "CZipReader.h"
#include "CMountPointReader.h"
#include "CPakReader.h"
#include "CNPKReader.h"
#include "CTarReader.h"
#include "CWADReader.h"
#include "CFileList.h"
#include "stdio.h"
#include "os.h"
#include "CMemoryFile.h"
#include "CLimitReadFile.h"


#if defined (_IRR_WINDOWS_API_)
	#if !defined ( _WIN32_WCE )
		#include <direct.h> // for _chdir
		#include <io.h> // for _access
		#include <tchar.h>
	#endif
#else
	#if (defined(_IRR_POSIX_API_) || defined(_IRR_OSX_PLATFORM_))
		#include <stdio.h>
		#include <stdlib.h>
		#include <string.h>
		#include <limits.h>
		#include <sys/types.h>
		#include <dirent.h>
		#include <sys/stat.h>
		#include <unistd.h>
	#endif
#endif

namespace irr
{
namespace io
{

//! constructor
CFileSystem::CFileSystem()
{
	#ifdef _IRR_DEBUG
	setDebugName("CFileSystem");
	#endif

	setFileListSystem(FILESYSTEM_NATIVE);
	//! reset current working directory
	getWorkingDirectory();

#ifdef __IRR_COMPILE_WITH_PAK_ARCHIVE_LOADER_
	ArchiveLoader.push_back(new CArchiveLoaderPAK(this));
#endif

#ifdef __IRR_COMPILE_WITH_NPK_ARCHIVE_LOADER_
	ArchiveLoader.push_back(new CArchiveLoaderNPK(this));
#endif

#ifdef __IRR_COMPILE_WITH_TAR_ARCHIVE_LOADER_
	ArchiveLoader.push_back(new CArchiveLoaderTAR(this));
#endif

#ifdef __IRR_COMPILE_WITH_WAD_ARCHIVE_LOADER_
	ArchiveLoader.push_back(new CArchiveLoaderWAD(this));
#endif

#ifdef __IRR_COMPILE_WITH_MOUNT_ARCHIVE_LOADER_
	ArchiveLoader.push_back(new CArchiveLoaderMount(this));
#endif

#ifdef __IRR_COMPILE_WITH_ZIP_ARCHIVE_LOADER_
	ArchiveLoader.push_back(new CArchiveLoaderZIP(this));
#endif

}


//! destructor
CFileSystem::~CFileSystem()
{
	uint32_t i;

	for ( i=0; i < FileArchives.size(); ++i)
	{
		FileArchives[i]->drop();
	}

	for ( i=0; i < ArchiveLoader.size(); ++i)
	{
		ArchiveLoader[i]->drop();
	}
}


//! opens a file for read access
IReadFile* CFileSystem::createAndOpenFile(const io::path& filename)
{
	IReadFile* file = 0;
	uint32_t i;

	for (i=0; i< FileArchives.size(); ++i)
	{
		file = FileArchives[i]->createAndOpenFile(filename);
		if (file)
			return file;
	}

	// Create the file using an absolute path so that it matches
	// the scheme used by CNullDriver::getTexture().
    file = new CReadFile(getAbsolutePath(filename));
    if (static_cast<CReadFile*>(file)->isOpen())
        return file;

    file->drop();
    return 0;
}


//! Creates an IReadFile interface for treating memory like a file.
IReadFile* CFileSystem::createMemoryReadFile(const void* contents, size_t len, const io::path& fileName)
{
	if (!contents)
		return nullptr;
	else
		return new CMemoryReadFile(contents, len, fileName);
}


//! Creates an IReadFile interface for reading files inside files
IReadFile* CFileSystem::createLimitReadFile(const io::path& fileName,
		IReadFile* alreadyOpenedFile, const size_t& pos, const size_t& areaSize)
{
	if (!alreadyOpenedFile)
		return 0;
	else
		return new CLimitReadFile(alreadyOpenedFile, pos, areaSize, fileName);
}


//! Creates an IReadFile interface for treating memory like a file.
IWriteFile* CFileSystem::createMemoryWriteFile(size_t len, const io::path& fileName)
{
    return new CMemoryWriteFile(len, fileName);
}


//! Opens a file for write access.
IWriteFile* CFileSystem::createAndWriteFile(const io::path& filename, bool append)
{
	return createWriteFile(filename, append);
}


//! Adds an external archive loader to the engine.
void CFileSystem::addArchiveLoader(IArchiveLoader* loader)
{
	if (!loader)
		return;

	loader->grab();
	ArchiveLoader.push_back(loader);
}

//! Returns the total number of archive loaders added.
uint32_t CFileSystem::getArchiveLoaderCount() const
{
	return ArchiveLoader.size();
}

//! Gets the archive loader by index.
IArchiveLoader* CFileSystem::getArchiveLoader(uint32_t index) const
{
	if (index < ArchiveLoader.size())
		return ArchiveLoader[index];
	else
		return 0;
}

//! move the hirarchy of the filesystem. moves sourceIndex relative up or down
bool CFileSystem::moveFileArchive(uint32_t sourceIndex, int32_t relative)
{
	bool r = false;
	const int32_t dest = (int32_t) sourceIndex + relative;
	const int32_t dir = relative < 0 ? -1 : 1;
	const int32_t sourceEnd = ((int32_t) FileArchives.size() ) - 1;
	IFileArchive *t;

	for (int32_t s = (int32_t) sourceIndex;s != dest; s += dir)
	{
		if (s < 0 || s > sourceEnd || s + dir < 0 || s + dir > sourceEnd)
			continue;

		t = FileArchives[s + dir];
		FileArchives[s + dir] = FileArchives[s];
		FileArchives[s] = t;
		r = true;
	}
	return r;
}


//! Adds an archive to the file system.
bool CFileSystem::addFileArchive(const io::path& filename, bool ignoreCase,
			  bool ignorePaths, E_FILE_ARCHIVE_TYPE archiveType,
			  const core::stringc& password,
			  IFileArchive** retArchive)
{
	IFileArchive* archive = 0;
	bool ret = false;

	// see if archive is already added
	if (changeArchivePassword(filename, password, retArchive))
		return true;

	int32_t i;

	// do we know what type it should be?
	if (archiveType == EFAT_UNKNOWN || archiveType == EFAT_FOLDER)
	{
		// try to load archive based on file name
		for (i = ArchiveLoader.size()-1; i >=0 ; --i)
		{
			if (ArchiveLoader[i]->isALoadableFileFormat(filename))
			{
				archive = ArchiveLoader[i]->createArchive(filename, ignoreCase, ignorePaths);
				if (archive)
					break;
			}
		}

		// try to load archive based on content
		if (!archive)
		{
			io::IReadFile* file = createAndOpenFile(filename);
			if (file)
			{
				for (i = ArchiveLoader.size()-1; i >= 0; --i)
				{
					file->seek(0);
					if (ArchiveLoader[i]->isALoadableFileFormat(file))
					{
						file->seek(0);
						archive = ArchiveLoader[i]->createArchive(file, ignoreCase, ignorePaths);
						if (archive)
							break;
					}
				}
				file->drop();
			}
		}
	}
	else
	{
		// try to open archive based on archive loader type

		io::IReadFile* file = 0;

		for (i = ArchiveLoader.size()-1; i >= 0; --i)
		{
			if (ArchiveLoader[i]->isALoadableFileFormat(archiveType))
			{
				// attempt to open file
				if (!file)
					file = createAndOpenFile(filename);

				// is the file open?
				if (file)
				{
					// attempt to open archive
					file->seek(0);
					if (ArchiveLoader[i]->isALoadableFileFormat(file))
					{
						file->seek(0);
						archive = ArchiveLoader[i]->createArchive(file, ignoreCase, ignorePaths);
						if (archive)
							break;
					}
				}
				else
				{
					// couldn't open file
					break;
				}
			}
		}

		// if open, close the file
		if (file)
			file->drop();
	}

	if (archive)
	{
		FileArchives.push_back(archive);
		if (password.size())
			archive->Password=password;
		if (retArchive)
			*retArchive = archive;
		ret = true;
	}
	else
	{
		os::Printer::log("Could not create archive for", std::string(filename.c_str()), ELL_ERROR);
	}

	return ret;
}

// don't expose!
bool CFileSystem::changeArchivePassword(const path& filename,
		const core::stringc& password,
		IFileArchive** archive)
{
	for (int32_t idx = 0; idx < (int32_t)FileArchives.size(); ++idx)
	{
		// TODO: This should go into a path normalization method
		// We need to check for directory names with trailing slash and without
		const path absPath = getAbsolutePath(filename);
		const path arcPath = FileArchives[idx]->getFileList()->getPath();
		if ((absPath == arcPath) || ((absPath+_IRR_TEXT("/")) == arcPath))
		{
			if (password.size())
				FileArchives[idx]->Password=password;
			if (archive)
				*archive = FileArchives[idx];
			return true;
		}
	}

	return false;
}

bool CFileSystem::addFileArchive(IReadFile* file, bool ignoreCase,
		bool ignorePaths, E_FILE_ARCHIVE_TYPE archiveType,
		const core::stringc& password, IFileArchive** retArchive)
{
	if (!file || archiveType == EFAT_FOLDER)
		return false;

	if (file)
	{
		if (changeArchivePassword(file->getFileName(), password, retArchive))
			return true;

		IFileArchive* archive = 0;
		int32_t i;

		if (archiveType == EFAT_UNKNOWN)
		{
			// try to load archive based on file name
			for (i = ArchiveLoader.size()-1; i >=0 ; --i)
			{
				if (ArchiveLoader[i]->isALoadableFileFormat(file->getFileName()))
				{
					archive = ArchiveLoader[i]->createArchive(file, ignoreCase, ignorePaths);
					if (archive)
						break;
				}
			}

			// try to load archive based on content
			if (!archive)
			{
				for (i = ArchiveLoader.size()-1; i >= 0; --i)
				{
					file->seek(0);
					if (ArchiveLoader[i]->isALoadableFileFormat(file))
					{
						file->seek(0);
						archive = ArchiveLoader[i]->createArchive(file, ignoreCase, ignorePaths);
						if (archive)
							break;
					}
				}
			}
		}
		else
		{
			// try to open archive based on archive loader type
			for (i = ArchiveLoader.size()-1; i >= 0; --i)
			{
				if (ArchiveLoader[i]->isALoadableFileFormat(archiveType))
				{
					// attempt to open archive
					file->seek(0);
					if (ArchiveLoader[i]->isALoadableFileFormat(file))
					{
						file->seek(0);
						archive = ArchiveLoader[i]->createArchive(file, ignoreCase, ignorePaths);
						if (archive)
							break;
					}
				}
			}
		}

		if (archive)
		{
			FileArchives.push_back(archive);
			if (password.size())
				archive->Password=password;
			if (retArchive)
				*retArchive = archive;
			return true;
		}
		else
		{
			os::Printer::log("Could not create archive for", file->getFileName().c_str(), ELL_ERROR);
		}
	}

	return false;
}


//! Adds an archive to the file system.
bool CFileSystem::addFileArchive(IFileArchive* archive)
{
	for (uint32_t i=0; i < FileArchives.size(); ++i)
	{
		if (archive == FileArchives[i])
			return false;
	}
	FileArchives.push_back(archive);
	return true;
}


//! removes an archive from the file system.
bool CFileSystem::removeFileArchive(uint32_t index)
{
	bool ret = false;
	if (index < FileArchives.size())
	{
	    auto it = FileArchives.begin()+index;
		(*it)->drop();
		FileArchives.erase(it);
		ret = true;
	}

	return ret;
}


//! removes an archive from the file system.
bool CFileSystem::removeFileArchive(const io::path& filename)
{
	const path absPath = getAbsolutePath(filename);
	for (uint32_t i=0; i < FileArchives.size(); ++i)
	{
		if (absPath == FileArchives[i]->getFileList()->getPath())
			return removeFileArchive(i);
	}

	return false;
}


//! Removes an archive from the file system.
bool CFileSystem::removeFileArchive(const IFileArchive* archive)
{
	for (uint32_t i=0; i < FileArchives.size(); ++i)
	{
		if (archive == FileArchives[i])
			return removeFileArchive(i);
	}

	return false;
}


//! gets an archive
uint32_t CFileSystem::getFileArchiveCount() const
{
	return FileArchives.size();
}


IFileArchive* CFileSystem::getFileArchive(uint32_t index)
{
	return index < getFileArchiveCount() ? FileArchives[index] : 0;
}


//! Returns the string of the current working directory
const io::path& CFileSystem::getWorkingDirectory()
{
	EFileSystemType type = FileSystemType;

	if (type != FILESYSTEM_NATIVE)
	{
		type = FILESYSTEM_VIRTUAL;
	}
	else
	{
		#if defined(_IRR_WINDOWS_API_)
			char tmp[_MAX_PATH];
			#if defined(_IRR_WCHAR_FILESYSTEM )
				_wgetcwd(tmp, _MAX_PATH);
				WorkingDirectory[FILESYSTEM_NATIVE] = tmp;
			#else
				_getcwd(tmp, _MAX_PATH);
				WorkingDirectory[FILESYSTEM_NATIVE] = tmp;
			#endif
            handleBackslashes(&WorkingDirectory[FILESYSTEM_NATIVE]);
		#endif

		#if (defined(_IRR_POSIX_API_) || defined(_IRR_OSX_PLATFORM_))

			// getting the CWD is rather complex as we do not know the size
			// so try it until the call was successful
			// Note that neither the first nor the second parameter may be 0 according to POSIX

			#if defined(_IRR_WCHAR_FILESYSTEM )
				uint32_t pathSize=256;
				wchar_t *tmpPath = new wchar_t[pathSize];
				while ((pathSize < (1<<16)) && !(wgetcwd(tmpPath,pathSize)))
				{
					delete [] tmpPath;
					pathSize *= 2;
					tmpPath = new char[pathSize];
				}
				if (tmpPath)
				{
					WorkingDirectory[FILESYSTEM_NATIVE] = tmpPath;
					delete [] tmpPath;
				}
			#else
				uint32_t pathSize=256;
				char *tmpPath = new char[pathSize];
				while ((pathSize < (1<<16)) && !(getcwd(tmpPath,pathSize)))
				{
					delete [] tmpPath;
					pathSize *= 2;
					tmpPath = new char[pathSize];
				}
				if (tmpPath)
				{
					WorkingDirectory[FILESYSTEM_NATIVE] = tmpPath;
					delete [] tmpPath;
				}
			#endif
		#endif

		WorkingDirectory[type].validate();
	}

	return WorkingDirectory[type];
}


//! Changes the current Working Directory to the given string.
bool CFileSystem::changeWorkingDirectoryTo(const io::path& newDirectory)
{
	bool success=false;

	if (FileSystemType != FILESYSTEM_NATIVE)
	{
		WorkingDirectory[FILESYSTEM_VIRTUAL] = newDirectory;
		// is this empty string constant really intended?
		flattenFilename(WorkingDirectory[FILESYSTEM_VIRTUAL], _IRR_TEXT(""));
		success = true;
	}
	else
	{
		WorkingDirectory[FILESYSTEM_NATIVE] = newDirectory;

#if defined(_MSC_VER)
	#if defined(_IRR_WCHAR_FILESYSTEM)
		success = (_wchdir(newDirectory.c_str()) == 0);
	#else
		success = (_chdir(newDirectory.c_str()) == 0);
	#endif
#else
    #if defined(_IRR_WCHAR_FILESYSTEM)
		success = (_wchdir(newDirectory.c_str()) == 0);
    #else
        success = (chdir(newDirectory.c_str()) == 0);
    #endif
#endif
	}

	return success;
}


io::path CFileSystem::getAbsolutePath(const io::path& filename) const
{
#if defined(_IRR_WINDOWS_API_)
	char *p=0;
	char fpath[_MAX_PATH];
	#if defined(_IRR_WCHAR_FILESYSTEM )
		p = _wfullpath(fpath, filename.c_str(), _MAX_PATH);
		core::stringw tmp(p);
	#else
		p = _fullpath(fpath, filename.c_str(), _MAX_PATH);
		core::stringc tmp(p);
	#endif
	handleBackslashes(&tmp);
	return tmp;
#elif (defined(_IRR_POSIX_API_) || defined(_IRR_OSX_PLATFORM_))
	char* p=0;
	char fpath[4096];
	fpath[0]=0;
	p = realpath(filename.c_str(), fpath);
	if (!p)
	{
		// content in fpath is unclear at this point
		if (!fpath[0]) // seems like fpath wasn't altered, use our best guess
		{
			io::path tmp(filename);
			return flattenFilename(tmp);
		}
		else
			return io::path(fpath);
	}
	if (filename[filename.size()-1]=='/')
		return io::path(p)+_IRR_TEXT("/");
	else
		return io::path(p);
#else
	return io::path(filename);
#endif
}



/*
	template<class container>
	uint32_t split(container& ret, const T* const c, uint32_t count=1, bool ignoreEmptyTokens=true, bool keepSeparators=false) const
	{
		if (!c)
			return 0;

		const uint32_t oldSize=ret.size();
		uint32_t lastpos = 0;
		bool lastWasSeparator = false;
		for (uint32_t i=0; i<used; ++i)
		{
			bool foundSeparator = false;
			for (uint32_t j=0; j<count; ++j)
			{
				if (array[i] == c[j])
				{
					if ((!ignoreEmptyTokens || i - lastpos != 0) &&
							!lastWasSeparator)
						ret.push_back(string<T,TAlloc>(&array[lastpos], i - lastpos));
					foundSeparator = true;
					lastpos = (keepSeparators ? i : i + 1);
					break;
				}
			}
			lastWasSeparator = foundSeparator;
		}
		if ((used - 1) > lastpos)
			ret.push_back(string<T,TAlloc>(&array[lastpos], (used - 1) - lastpos));
		return ret.size()-oldSize;
	}
*/

//! Get the relative filename, relative to the given directory
path CFileSystem::getRelativeFilename(const path& filename, const path& directory) const
{
	if ( filename.empty() || directory.empty() )
		return filename;

	io::path path1, file, ext;
	core::splitFilename(getAbsolutePath(filename), &path1, &file, &ext);
	io::path path2(getAbsolutePath(directory));
	core::list<io::path> list1, list2;
	path1.split(list1, _IRR_TEXT("/\\"), 2);
	path2.split(list2, _IRR_TEXT("/\\"), 2);
	uint32_t i=0;
	core::list<io::path>::const_iterator it1,it2;
	it1=list1.begin();
	it2=list2.begin();

	#if defined (_IRR_WINDOWS_API_)
	char partition1 = 0, partition2 = 0;
	io::path prefix1, prefix2;
	if ( it1 != list1.end() )
		prefix1 = *it1;
	if ( it2 != list2.end() )
		prefix2 = *it2;
	if ( prefix1.size() > 1 && prefix1[1] == _IRR_TEXT(':') )
		partition1 = core::locale_lower(prefix1[0]);
	if ( prefix2.size() > 1 && prefix2[1] == _IRR_TEXT(':') )
		partition2 = core::locale_lower(prefix2[0]);

	// must have the same prefix or we can't resolve it to a relative filename
	if ( partition1 != partition2 )
	{
		return filename;
	}
	#endif


	for (; i<list1.size() && i<list2.size()
#if defined (_IRR_WINDOWS_API_)
		&& (io::path(*it1).make_lower()==io::path(*it2).make_lower())
#else
		&& (*it1==*it2)
#endif
		; ++i)
	{
		++it1;
		++it2;
	}
	path1=_IRR_TEXT("");
	for (; i<list2.size(); ++i)
		path1 += _IRR_TEXT("../");
	while (it1 != list1.end())
	{
		path1.append(*it1++);
		path1.append(_IRR_TEXT('/'));
	}
	path1 += file;
	if (ext.size())
	{
		path1.append(_IRR_TEXT('.'));
		path1 += ext;
	}
	return path1;
}


//! Sets the current file systen type
EFileSystemType CFileSystem::setFileListSystem(EFileSystemType listType)
{
	EFileSystemType current = FileSystemType;
	FileSystemType = listType;
	return current;
}


//! Creates a list of files and directories in the current working directory
IFileList* CFileSystem::createFileList()
{
	CFileList* r = 0;
	io::path Path = getWorkingDirectory();
	handleBackslashes(&Path);
	if (Path.lastChar() != '/')
		Path.append('/');

	//! Construct from native filesystem
	if (FileSystemType == FILESYSTEM_NATIVE)
	{
		// --------------------------------------------
		//! Windows version
		#ifdef _IRR_WINDOWS_API_
		#if !defined ( _WIN32_WCE )

		r = new CFileList(Path, true, false);

		// TODO: Should be unified once mingw adapts the proper types
#if defined(__GNUC__)
		long hFile; //mingw return type declaration
#else
		intptr_t hFile;
#endif

		struct _tfinddata_t c_file;
		if( (hFile = _tfindfirst( _T("*"), &c_file )) != -1L )
		{
			do
			{
				r->addItem(Path + c_file.name, 0, c_file.size, (_A_SUBDIR & c_file.attrib) != 0, 0);
			}
			while( _tfindnext( hFile, &c_file ) == 0 );

			_findclose( hFile );
		}
		#endif

		//TODO add drives
		//entry.Name = "E:\\";
		//entry.isDirectory = true;
		//Files.push_back(entry);
		#endif

		// --------------------------------------------
		//! Linux version
		#if (defined(_IRR_POSIX_API_) || defined(_IRR_OSX_PLATFORM_))


		r = new CFileList(Path, false, false);

		r->addItem(Path + _IRR_TEXT(".."), 0, 0, true, 0);

		//! We use the POSIX compliant methods instead of scandir
		DIR* dirHandle=opendir(Path.c_str());
		if (dirHandle)
		{
			struct dirent *dirEntry;
			while ((dirEntry=readdir(dirHandle)))
			{
				uint32_t size = 0;
				bool isDirectory = false;

				if((strcmp(dirEntry->d_name, ".")==0) ||
				   (strcmp(dirEntry->d_name, "..")==0))
				{
					continue;
				}
				struct stat buf;
				if (stat(dirEntry->d_name, &buf)==0)
				{
					size = buf.st_size;
					isDirectory = S_ISDIR(buf.st_mode);
				}
				#if !defined(_IRR_SOLARIS_PLATFORM_) && !defined(__CYGWIN__) && !defined(__LSB_VERSION__)
				// only available on some systems
				else
				{
					isDirectory = dirEntry->d_type == DT_DIR;
				}
				#endif

				r->addItem(Path + dirEntry->d_name, 0, size, isDirectory, 0);
			}
			closedir(dirHandle);
		}
		#endif
	}
	else
	{
		//! create file list for the virtual filesystem
		r = new CFileList(Path, false, false);

		//! add relative navigation
		SFileListEntry e2;
		SFileListEntry e3;

		//! PWD
		r->addItem(Path + _IRR_TEXT("."), 0, 0, true, 0);

		//! parent
		r->addItem(Path + _IRR_TEXT(".."), 0, 0, true, 0);

		//! merge archives
		for (uint32_t i=0; i < FileArchives.size(); ++i)
		{
			const IFileList *merge = FileArchives[i]->getFileList();

			auto files = merge->getFiles();
			for (auto it=files.begin(); it!=files.end(); it++)
			{
				if (core::isInSameDirectory(Path, it->FullName) == 0)
					r->addItem(it->FullName, it->Offset, it->Size, it->IsDirectory, 0);
			}
		}
	}

	return r;
}

//! Creates an empty filelist
IFileList* CFileSystem::createEmptyFileList(const io::path& path, bool ignoreCase, bool ignorePaths)
{
	return new CFileList(path, ignoreCase, ignorePaths);
}


//! determines if a file exists and would be able to be opened.
bool CFileSystem::existFile(const io::path& filename) const
{
	for (uint32_t i=0; i < FileArchives.size(); ++i)
	{
        auto _list = FileArchives[i]->getFileList();
        auto files = _list->getFiles();
		if (_list->findFile(files.begin(),files.end(),filename)!=files.end())
			return true;
	}

#if defined(_MSC_VER)
    #if defined(_IRR_WCHAR_FILESYSTEM)
        return (_waccess(filename.c_str(), 0) != -1);
    #else
        return (_access(filename.c_str(), 0) != -1);
    #endif
#elif defined(F_OK)
    #if defined(_IRR_WCHAR_FILESYSTEM)
        return (_waccess(filename.c_str(), F_OK) != -1);
    #else
        return (access(filename.c_str(), F_OK) != -1);
	#endif
#else
    return (access(filename.c_str(), 0) != -1);
#endif
}



//! creates a filesystem which is able to open files from the ordinary file system,
//! and out of zipfiles, which are able to be added to the filesystem.
IFileSystem* createFileSystem()
{
	return new CFileSystem();
}


} // end namespace irr
} // end namespace io


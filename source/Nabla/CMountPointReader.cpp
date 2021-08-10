// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifdef NEW_FILESYSTEM
#include "CMountPointReader.h"

#ifdef __NBL_COMPILE_WITH_MOUNT_ARCHIVE_LOADER_

#include "CReadFile.h"
#include "nbl_os.h"

namespace nbl
{
namespace io
{

//! Constructor
CArchiveLoaderMount::CArchiveLoaderMount( io::IFileSystem* fs)
: FileSystem(fs)
{
	#ifdef _NBL_DEBUG
	setDebugName("CArchiveLoaderMount");
	#endif
}


//! returns true if the file maybe is able to be loaded by this class
bool CArchiveLoaderMount::isALoadableFileFormat(const std::filesystem::path& filename) const
{
	std::filesystem::path fname(filename);
	core::deletePathFromFilename(fname);

	if (!fname.string().size())
		return true;
	IFileList* list = FileSystem->createFileList();
	bool ret = false;
	if (list)
	{
	    auto files = list->getFiles();
		// check if name is found as directory
		if (list->findFile(files.begin(),files.end(),filename, true)!=files.end())
			ret=true;
		list->drop();
	}
	return ret;
}

//! Check to see if the loader can create archives of this type.
bool CArchiveLoaderMount::isALoadableFileFormat(E_FILE_ARCHIVE_TYPE fileType) const
{
	return fileType == EFAT_FOLDER;
}

//! Check if the file might be loaded by this class
bool CArchiveLoaderMount::isALoadableFileFormat(io::IReadFile* file) const
{
	return false;
}

//! Creates an archive from the filename
IFileArchive* CArchiveLoaderMount::createArchive(const std::filesystem::path& filename) const
{
	IFileArchive *archive = 0;

	EFileSystemType current = FileSystem->setFileListSystem(FILESYSTEM_NATIVE);

	const std::filesystem::path save = FileSystem->getWorkingDirectory();
	std::filesystem::path fullPath = io::IFileSystem::flattenFilename(std::filesystem::absolute(filename));

	if (FileSystem->changeWorkingDirectoryTo(fullPath))
	{
		archive = new CMountPointReader(FileSystem, fullPath);
	}

	FileSystem->changeWorkingDirectoryTo(save);
	FileSystem->setFileListSystem(current);

	return archive;
}

//! creates/loads an archive from the file.
//! \return Pointer to the created archive. Returns 0 if loading failed.
IFileArchive* CArchiveLoaderMount::createArchive(io::IReadFile* file) const
{
	return 0;
}

//! compatible Folder Architecture
CMountPointReader::CMountPointReader(IFileSystem * parent, const std::filesystem::path& basename)
	: CFileList(basename), Parent(parent)
{
	//! ensure CFileList path ends in a slash
	if (*Path.string().rbegin() != '/' )
		Path += '/';

	const std::filesystem::path& work = Parent->getWorkingDirectory();

	Parent->changeWorkingDirectoryTo(basename);
	buildDirectory();
	Parent->changeWorkingDirectoryTo(work);
}


//! returns the list of files
const IFileList* CMountPointReader::getFileList() const
{
	return this;
}

void CMountPointReader::buildDirectory()
{
	IFileList * list = Parent->createFileList();
	if (!list)
		return;

	auto files = list->getFiles();
	for (auto it=files.begin(); it!=files.end(); it++)
	{
		std::filesystem::path full = it->FullName;
		full = full.string().substr(Path.string().size(), full.string().size() - Path.string().size());

		if (!it->IsDirectory)
		{
			addItem(full, it->Offset, it->Size, false, RealFileNames.size());
			RealFileNames.push_back(it->FullName);
		}
		else
		{
			const std::filesystem::path rel = it->Name;
			RealFileNames.push_back(it->FullName);

			std::filesystem::path pwd  = Parent->getWorkingDirectory();
			if (*pwd.string().rbegin() != '/')
				pwd.string() += '/';
			pwd += rel;

			if ( rel != "." && rel != ".." )
			{
				addItem(full, 0, 0, true, 0);
				Parent->changeWorkingDirectoryTo(pwd);
				buildDirectory();
				Parent->changeWorkingDirectoryTo("..");
			}
		}
	}

	list->drop();
}

//! opens a file by file name
IReadFile* CMountPointReader::createAndOpenFile(const std::filesystem::path& filename)
{
    auto found = findFile(Files.begin(),Files.end(),filename,false);
	if (found != Files.end())
    {
        return Parent->createAndOpenFile(RealFileNames[found->ID]);
    }
	
	return nullptr;
}


} // io
} // nbl

#endif // __NBL_COMPILE_WITH_MOUNT_ARCHIVE_LOADER_
#endif
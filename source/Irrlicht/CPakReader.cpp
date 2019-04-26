// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// Code contributed by skreamz

#include "CPakReader.h"

#ifdef __IRR_COMPILE_WITH_PAK_ARCHIVE_LOADER_

#include "os.h"
#include "coreutil.h"
#include "CLimitReadFile.h"

namespace irr
{
namespace io
{

namespace
{

inline bool isHeaderValid(const SPAKFileHeader& header)
{
	const char* tag = header.tag;
	return tag[0] == 'P' &&
		   tag[1] == 'A' &&
		   tag[2] == 'C' &&
		   tag[3] == 'K';
}

} // end namespace

//! Constructor
CArchiveLoaderPAK::CArchiveLoaderPAK( io::IFileSystem* fs)
: FileSystem(fs)
{
#ifdef _IRR_DEBUG
	setDebugName("CArchiveLoaderPAK");
#endif
}


//! returns true if the file maybe is able to be loaded by this class
bool CArchiveLoaderPAK::isALoadableFileFormat(const io::path& filename) const
{
	return core::hasFileExtension(filename, "pak");
}

//! Check to see if the loader can create archives of this type.
bool CArchiveLoaderPAK::isALoadableFileFormat(E_FILE_ARCHIVE_TYPE fileType) const
{
	return fileType == EFAT_PAK;
}

//! Creates an archive from the filename
/** \param file File handle to check.
\return Pointer to newly created archive, or 0 upon error. */
IFileArchive* CArchiveLoaderPAK::createArchive(const io::path& filename, bool ignoreCase, bool ignorePaths) const
{
	IFileArchive *archive = 0;
	io::IReadFile* file = FileSystem->createAndOpenFile(filename);

	if (file)
	{
		archive = createArchive(file, ignoreCase, ignorePaths);
		file->drop ();
	}

	return archive;
}

//! creates/loads an archive from the file.
//! \return Pointer to the created archive. Returns 0 if loading failed.
IFileArchive* CArchiveLoaderPAK::createArchive(io::IReadFile* file, bool ignoreCase, bool ignorePaths) const
{
	IFileArchive *archive = 0;
	if ( file )
	{
		file->seek ( 0 );
		archive = new CPakReader(file, ignoreCase, ignorePaths);
	}
	return archive;
}


//! Check if the file might be loaded by this class
/** Check might look into the file.
\param file File handle to check.
\return True if file seems to be loadable. */
bool CArchiveLoaderPAK::isALoadableFileFormat(io::IReadFile* file) const
{
	SPAKFileHeader header;

	file->read(&header, sizeof(header));

	return isHeaderValid(header);
}


/*!
	PAK Reader
*/
CPakReader::CPakReader(IReadFile* file, bool ignoreCase, bool ignorePaths)
: CFileList((file ? file->getFileName() : io::path("")), ignoreCase, ignorePaths), File(file)
{
#ifdef _IRR_DEBUG
	setDebugName("CPakReader");
#endif

	if (File)
	{
		File->grab();
		scanLocalHeader();
	}
}


CPakReader::~CPakReader()
{
	if (File)
		File->drop();
}


const IFileList* CPakReader::getFileList() const
{
	return this;
}

bool CPakReader::scanLocalHeader()
{
	SPAKFileHeader header;

	// Read and validate the header
	File->read(&header, sizeof(header));
	if (!isHeaderValid(header))
		return false;

	// Seek to the table of contents
	File->seek(header.offset);

	const int numberOfFiles = header.length / sizeof(SPAKFileEntry);

	// Loop through each entry in the table of contents
	for(int i = 0; i < numberOfFiles; i++)
	{
		// read an entry
		SPAKFileEntry entry;
		File->read(&entry, sizeof(entry));

#ifdef _IRR_DEBUG
		os::Printer::log(entry.name);
#endif

		addItem(io::path(entry.name), entry.offset, entry.length, false );
	}
	return true;
}


//! opens a file by file name
IReadFile* CPakReader::createAndOpenFile(const io::path& filename)
{
    auto it = findFile(Files.begin(),Files.end(),filename,false);
	if (it!=Files.end())
        return new CLimitReadFile(File, it->Offset, it->Size, it->FullName);

	return 0;
}
} // end namespace io
} // end namespace irr

#endif // __IRR_COMPILE_WITH_PAK_ARCHIVE_LOADER_


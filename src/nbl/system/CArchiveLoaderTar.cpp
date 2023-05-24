#include "nbl/system/CArchiveLoaderTar.h"


enum E_TAR_LINK_INDICATOR
{
	ETLI_REGULAR_FILE_OLD = 0,
	ETLI_REGULAR_FILE = '0',
	ETLI_LINK_TO_ARCHIVED_FILE = '1', // Hard link
	ETLI_SYMBOLIC_LINK = '2',
	ETLI_CHAR_SPECIAL_DEVICE = '3',
	ETLI_BLOCK_SPECIAL_DEVICE = '4',
	ETLI_DIRECTORY = '5',
	ETLI_FIFO_SPECIAL_FILE = '6',
	ETLI_CONTIGUOUS_FILE = '7'
};

// byte-align structures
#include "nbl/nblpack.h"
struct STarHeader
{
	char FileName[100];
	char FileMode[8];
	char UserID[8];
	char GroupID[8];
	char Size[12];
	char ModifiedTime[12];
	char Checksum[8];
	char Link;
	char LinkName[100];
	char Magic[6];
	char USTARVersion[2];
	char UserName[32];
	char GroupName[32];
	char DeviceMajor[8];
	char DeviceMinor[8];
	char FileNamePrefix[155];
} PACK_STRUCT;
#include "nbl/nblunpack.h"


using namespace nbl;
using namespace nbl::system;

CFileArchive::file_buffer_t CArchiveLoaderTar::CArchive::getFileBuffer(const IFileArchive::SFileList::SEntry* item)
{
	assert(item->allocatorType==EAT_NULL);
	return {reinterpret_cast<uint8_t*>(m_file->getMappedPointer())+item->offset,item->size,nullptr};
}


bool CArchiveLoaderTar::isALoadableFileFormat(IFile* file) const
{
	// TAR files consist of blocks of 512 bytes
	// if it isn't a multiple of 512 then it's not a TAR file.
	if (file->getSize()%BlockSize)
		return false;

	STarHeader fHead;
	IFile::success_t success;
	file->read(success,&fHead,0ull,sizeof(fHead));
	if (!success)
		return false;

	uint32_t checksum = 0;
	sscanf(fHead.Checksum,"%o",&checksum);
	
	// verify checksum

	// some old TAR writers assume that chars are signed, others assume unsigned
	// USTAR archives have a longer header, old TAR archives end after linkname

	uint32_t checksum1=0;
	int32_t checksum2=0;

	// remember to blank the checksum field!
	memset(fHead.Checksum, ' ', 8);

	// old header
	for (uint8_t* p = (uint8_t*)(&fHead); p < (uint8_t*)(&fHead.Magic[0]); ++p)
	{
		checksum1 += *p;
		checksum2 += char(*p);
	}

	if (!strncmp(fHead.Magic, "ustar", 5))
	{
		for (uint8_t* p = (uint8_t*)(&fHead.Magic[0]); p < (uint8_t*)(&fHead) + sizeof(fHead); ++p)
		{
			checksum1 += *p;
			checksum2 += char(*p);
		}
	}
	return checksum1 == checksum || checksum2 == (int32_t)checksum;
}

core::smart_refctd_ptr<IFileArchive> CArchiveLoaderTar::createArchive_impl(core::smart_refctd_ptr<system::IFile>&& file, const std::string_view& password) const
{
	if (!file || !(file->getFlags()&IFileBase::ECF_MAPPABLE))
		return nullptr;

	std::vector<IFileArchive::SFileList::SEntry> items;
	for (size_t pos=0ull; true; )
	{
		STarHeader fHead;
		{
			IFile::success_t success;
			file->read(success,&fHead,pos,sizeof(fHead));
			if (!success)
				break;
		}
		// only add standard files for now
		switch (fHead.Link)
		{
			case ETLI_REGULAR_FILE:
				[[fallthrough]];
			case ETLI_REGULAR_FILE_OLD:
			{
				std::string fullPath = "";
				fullPath.reserve(256);

				// USTAR archives have a filename prefix
				// may not be null terminated, copy carefully!
				if (!strncmp(fHead.Magic, "ustar", 5))
				{
					char* np = fHead.FileNamePrefix;
					while (*np && (np - fHead.FileNamePrefix) < 155)
						fullPath += *np;
					np++;
				}

				// append the file name
				char* np = fHead.FileName;
				while (*np && (np - fHead.FileName) < 100)
				{
					fullPath += *np;
					np++;
				}

				// get size
				std::string sSize = "";
				sSize.reserve(12);
				np = fHead.Size;
				while (*np && (np - fHead.Size) < 12)
				{
					sSize += *np;
					np++;
				}

				// TODO: this is horrible, replace
				const size_t size = strtoul(sSize.c_str(), NULL, 8);
				if (errno == ERANGE)
					m_logger.log("File %s is too large", ILogger::ELL_WARNING, fullPath.c_str());

				// save start position
				const uint32_t offset = pos + BlockSize;

				// move to next file header block
				pos = offset + core::roundUp(size,BlockSize);

				// add file to list
				auto& item = items.emplace_back();
				item.pathRelativeToArchive = fullPath;
				item.size = size;
				item.offset = offset;
				item.ID = items.size()-1u;
				item.allocatorType = IFileArchive::EAT_NULL;
				break;
			}
			// TODO: ETLI_DIRECTORY, ETLI_LINK_TO_ARCHIVED_FILE
			default:
				// move to next block
				pos += BlockSize;
				break;
		}
	}
	if (items.empty())
		return nullptr;

	return core::make_smart_refctd_ptr<CArchive>(std::move(file),core::smart_refctd_ptr(m_logger.get()),std::move(items));
}
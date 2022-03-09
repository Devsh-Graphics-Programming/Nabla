#include "nbl/system/IFileViewAllocator.h"
#include "nbl/system/CArchiveLoaderZip.h"


#include <aesGladman/fileenc.h>

#include <zconf.h>
#include <zlib/zlib.h>

#include <bzip2/bzlib.h>


#include "nbl/nblpack.h"
struct SZIPFileCentralDirFileHeader
{
	uint32_t Sig;	// 'PK0102' (0x02014b50)
	uint16_t VersionMadeBy;
	uint16_t VersionToExtract;
	uint16_t GeneralBitFlag;
	uint16_t CompressionMethod;
	uint16_t LastModFileTime;
	uint16_t LastModFileDate;
	uint32_t CRC32;
	uint32_t CompressedSize;
	uint32_t UncompressedSize;
	uint16_t FilenameLength;
	uint16_t ExtraFieldLength;
	uint16_t FileCommentLength;
	uint16_t DiskNumberStart;
	uint16_t InternalFileAttributes;
	uint32_t ExternalFileAttributes;
	uint32_t RelativeOffsetOfLocalHeader;

	// filename (variable size)
	// extra field (variable size)
	// file comment (variable size)

} PACK_STRUCT;
struct SZIPFileCentralDirEnd
{
	uint32_t Sig;			// 'PK0506' end_of central dir signature			// (0x06054b50)
	uint16_t NumberDisk;		// number of this disk
	uint16_t NumberStart;	// number of the disk with the start of the central directory
	uint16_t TotalDisk;		// total number of entries in the central dir on this disk
	uint16_t TotalEntries;	// total number of entries in the central dir
	uint32_t Size;			// size of the central directory
	uint32_t Offset;			// offset of start of centraldirectory with respect to the starting disk number
	uint16_t CommentLength;	// zipfile comment length
	// zipfile comment (variable size)
} PACK_STRUCT;
struct SZipFileAESExtraData
{
	int16_t Version;
	uint8_t Vendor[2];
	uint8_t EncryptionStrength;
	int16_t CompressionMode;
} PACK_STRUCT;
struct SGZIPMemberHeader
{
	uint16_t sig; // 0x8b1f
	uint8_t  compressionMethod; // 8 = deflate
	uint8_t  flags;
	uint32_t time;
	uint8_t  extraFlags; // slow compress = 2, fast compress = 4
	uint8_t  operatingSystem;
} PACK_STRUCT;
#include "nbl/nblunpack.h"

enum E_GZIP_FLAGS
{
	EGZF_TEXT_DAT = 1,
	EGZF_CRC16 = 2,
	EGZF_EXTRA_FIELDS = 4,
	EGZF_FILE_NAME = 8,
	EGZF_COMMENT = 16
};
struct SZipFileExtraHeader
{
	int16_t ID;
	int16_t Size;
};

// set if the file is encrypted
constexpr int16_t ZIP_FILE_ENCRYPTED = 0x0001;
// the fields crc-32, compressed size and uncompressed size are set to
// zero in the local header
constexpr int16_t ZIP_INFO_IN_DATA_DESCRIPTOR = 0x0008;


using namespace nbl;
using namespace nbl::system;


core::smart_refctd_ptr<IFileArchive> CArchiveLoaderZip::createArchive_impl(core::smart_refctd_ptr<system::IFile>&& file, const std::string_view& password) const
{
	if (!file)
		return nullptr;

	uint16_t sig;
	{
		IFile::success_t success;
		file->read(success,&sig,0,sizeof(sig));
		if (!success)
			return nullptr;
	}

	core::vector<IFileArchive::SListEntry> items;
	core::vector<SZIPFileHeader> itemsMetadata;
	// load file entries
	{
		const bool isGZip = sig==0x8b1fu;
		size_t offset = 0ull;
		std::string filename;
		filename.reserve(ISystem::MAX_FILENAME_LENGTH);
		while (true)
		{
			SZIPFileHeader header;
			memset(&header,0,sizeof(SZIPFileHeader));

			const auto& zipHeader = header;
			static_assert(sizeof(SGZIPMemberHeader)<sizeof(header));
			const auto& gzipHeader = reinterpret_cast<SGZIPMemberHeader&>(header);

			IFile::success_t headerReadSuccess;
			file->read(headerReadSuccess,&header,offset,isGZip ? sizeof(SGZIPMemberHeader):sizeof(header));
			if (!headerReadSuccess)
				break;
			offset += headerReadSuccess.getSizeToProcess();

			IFileArchive::SListEntry item;
			if (isGZip)
			{
				//! The gzip file format seems to think that there can be multiple files in a gzip file
				//! TODO: But OLD Irrlicht Impl doesn't honor it!?
				if (gzipHeader.sig!=0x8b1fu)
					break;
			
				// now get the file info
				if (gzipHeader.flags&EGZF_EXTRA_FIELDS)
				{
					// read lenth of extra data
					uint16_t dataLen;
					IFile::success_t success;
					file->read(success,&dataLen,offset,sizeof(dataLen));
					if (!success)
						break;
					offset += success.getSizeToProcess();
					// skip the extra data
					offset += dataLen;
				}
				//
				filename.clear();
				if (gzipHeader.flags&EGZF_FILE_NAME)
				{
					char c = 0x45; // make sure we start with non-zero char
					while (c)
					{
						IFile::success_t success;
						file->read(success,&c,offset,sizeof(c));
						if (!success)
							break;
						offset += success.getSizeToProcess();
						filename.push_back(c);
					}
					// if string is not null terminated, something went wrong reading the file
					if (c)
						break;
				}
				//
				if (gzipHeader.flags&EGZF_COMMENT)
				{
					char c = 0x45; // make sure we start with non-zero char
					while (c)
					{
						IFile::success_t success;
						file->read(success,&c,offset,sizeof(c));
						if (!success)
							break;
						offset += success.getSizeToProcess();
					}
					// if string is not null terminated, something went wrong reading the file
					if (c)
						break;
				}
				// skip crc16
				if (gzipHeader.flags&EGZF_CRC16)
					offset += 2;

				header.FilenameLength = filename.length();
				header.CompressionMethod = gzipHeader.compressionMethod;
				header.DataDescriptor.CompressedSize = file->getSize()-(offset+sizeof(uint64_t));

				item.offset = offset;
			
				offset += header.DataDescriptor.CompressedSize;
				// read CRC
				{
					IFile::success_t success;
					file->read(success,&header.DataDescriptor.CRC32,offset,sizeof(header.DataDescriptor.CRC32));
					if (!success)
						break;
					offset += success.getSizeToProcess();
				}
				// read uncompressed size
				{
					IFile::success_t success;
					file->read(success,&header.DataDescriptor.UncompressedSize,offset,sizeof(header.DataDescriptor.UncompressedSize));
					if (!success)
						break;
					offset += success.getSizeToProcess();
					item.size = header.DataDescriptor.UncompressedSize;
				}
			}
			else
			{	if (zipHeader.Sig!=0x04034b50u)
					break;

				filename.resize(zipHeader.FilenameLength);
				{
					IFile::success_t success;
					file->read(success,filename.data(),offset,zipHeader.FilenameLength);
					if (!success)
						break;
					offset += success.getSizeToProcess();
				}

				// AES encryption
				if ((header.GeneralBitFlag&ZIP_FILE_ENCRYPTED) && (header.CompressionMethod==99))
				{
/*/
					#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
		int16_t restSize = entry.header.ExtraFieldLength;
		SZipFileExtraHeader extraHeader;
		while (restSize)
		{
			{
				system::future<size_t> fut;
				m_file->read(fut, &extraHeader, offset, sizeof(extraHeader));
				fut.get();
				offset += sizeof(extraHeader);
			}
			restSize -= sizeof(extraHeader);
			if (extraHeader.ID == (int16_t)0x9901)
			{
				SZipFileAESExtraData data;
				{
					system::future<size_t> fut;
					m_file->read(fut, &data, offset, sizeof(data));
					fut.get();
					offset += sizeof(data);
				}

				restSize -= sizeof(data);
				if (data.Vendor[0] == 'A' && data.Vendor[1] == 'E')
				{
					// encode values into Sig
					// AE-Version | Strength | ActualMode
					entry.header.Sig =
						((data.Version & 0xff) << 24) |
						(data.EncryptionStrength << 16) |
						(data.CompressionMode);
					offset += restSize;
					break;
				}
			}
		}
					#else
*/
					break; // no support, can't decrypt
//					#endif
				}
				else
					offset += header.ExtraFieldLength;
		





				item.offset = offset;
				// move forward length of data
				offset += zipHeader.DataDescriptor.CompressedSize;
			}

			// we need to have a filename or we skip
			if (filename.empty())
				continue;

			item.pathRelativeToArchive = filename;
			item.ID = items.size();
			item.allocatorType = header.CompressionMethod ? IFileArchive::EAT_VIRTUAL_ALLOC:IFileArchive::EAT_NULL;
			items.push_back(item);
			itemsMetadata.push_back(header);
		}
	}

	assert(items.size()==itemsMetadata.size());
	if (items.empty())
		return nullptr;

	return core::make_smart_refctd_ptr<CArchive>(std::move(file),core::smart_refctd_ptr(m_logger.get()),std::move(items),std::move(itemsMetadata));
}

#if 0
bool CFileArchiveZip::scanZipHeader(size_t& offset, bool ignoreGPBits)
{

	// if bit 3 was set, use CentralDirectory for setup
	if (!ignoreGPBits && entry.header.GeneralBitFlag & ZIP_INFO_IN_DATA_DESCRIPTOR)
	{
		SZIPFileCentralDirEnd dirEnd;
		m_fileInfo.clear();
		m_files.clear();
		// First place where the end record could be stored
		offset = m_file->getSize() - 22;
		const char endID[] = { 0x50, 0x4b, 0x05, 0x06, 0x0 };
		char tmp[5] = { '\0' };
		bool found = false;
		// search for the end record ID
		while (!found && offset > 0)
		{
			int seek = 8;
			{
				system::future<size_t> fut;
				m_file->read(fut, tmp, offset, 4);
				fut.get();
				offset += 4;
			}
			switch (tmp[0])
			{
			case 0x50:
				if (!strcmp(endID, tmp))
				{
					seek = 4;
					found = true;
				}
				break;
			case 0x4b:
				seek = 5;
				break;
			case 0x05:
				seek = 6;
				break;
			case 0x06:
				seek = 7;
				break;
			}
			offset -= seek;
		}
		{
			system::future<size_t> fut;
			m_file->read(fut, &dirEnd, offset, sizeof(dirEnd));
			fut.get();
			offset += sizeof(dirEnd);
		}
		m_fileInfo.reserve(dirEnd.TotalEntries);
		offset = dirEnd.Offset;
		while (scanCentralDirectoryHeader(offset)) {}
		return false;
	}
}

bool CFileArchiveZip::scanCentralDirectoryHeader(size_t& offset)
{
	std::filesystem::path ZipFileName = "";
	SZIPFileCentralDirFileHeader entry;
	{
		system::future<size_t> fut;
		m_file->read(fut, &entry, offset, sizeof(SZIPFileCentralDirFileHeader));
		fut.get();
		offset += sizeof(SZIPFileCentralDirFileHeader);
	}

	if (entry.Sig != 0x02014b50)
		return false; // central dir headers end here.

	const long pos = offset;
	offset = entry.RelativeOffsetOfLocalHeader;
	scanZipHeader(offset, true);
	offset = pos + entry.FilenameLength + entry.ExtraFieldLength + entry.FileCommentLength;
	m_fileInfo.back().header.DataDescriptor.CompressedSize = entry.CompressedSize;
	m_fileInfo.back().header.DataDescriptor.UncompressedSize = entry.UncompressedSize;
	m_fileInfo.back().header.DataDescriptor.CRC32 = entry.CRC32;
	m_files.back().size = entry.UncompressedSize;
	return true;
}

core::smart_refctd_ptr<IFile> CFileArchiveZip::readFile_impl(const SOpenFileParams& params)
{

		const SZipFileEntry& e = m_fileInfo[found->ID];
		wchar_t buf[64];

		uint8_t* decryptedBuf = 0;
		uint32_t decryptedSize = e.header.DataDescriptor.CompressedSize;
#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
		if ((e.header.GeneralBitFlag & ZIP_FILE_ENCRYPTED) && (e.header.CompressionMethod == 99))
		{
			size_t readOffset;

			uint8_t salt[16] = { 0 };
			const uint16_t saltSize = (((e.header.Sig & 0x00ff0000) >> 16) + 1) * 4;
			{
				readOffset = e.Offset;
				read_blocking(m_file.get(), salt, readOffset, saltSize);
				readOffset += saltSize;
			}
			char pwVerification[2];
			char pwVerificationFile[2];
			{
				read_blocking(m_file.get(), pwVerification, readOffset, 2);
				readOffset += 2;
			}
			fcrypt_ctx zctx; // the encryption context
			int rc = fcrypt_init(
				(e.header.Sig & 0x00ff0000) >> 16,
				(const unsigned char*)m_password.c_str(), // the password
				m_password.size(), // number of bytes in password
				salt, // the salt
				(unsigned char*)pwVerificationFile, // on return contains password verifier
				&zctx); // encryption context
			if (strncmp(pwVerificationFile, pwVerification, 2))
			{
				m_logger.log("Wrong password", ILogger::ELL_ERROR);
				return 0;
			}
			decryptedSize = e.header.DataDescriptor.CompressedSize - saltSize - 12;
			decryptedBuf = new uint8_t[decryptedSize];
			//
			constexpr uint32_t kChunkSize = 0x8000u;
			uint32_t c = 0;
			while ((c + kChunkSize) <= decryptedSize)
			{
				{
					read_blocking(m_file.get(), decryptedBuf + c, readOffset, kChunkSize);
					readOffset += kChunkSize;
				}
				fcrypt_decrypt(
					decryptedBuf + c, // pointer to the data to decrypt
					kChunkSize,   // how many bytes to decrypt
					&zctx); // decryption context
				c += kChunkSize;
			}
			{
				read_blocking(m_file.get(), decryptedBuf + c, readOffset, decryptedSize - c);
				readOffset += decryptedSize - c;
			}
			fcrypt_decrypt(
				decryptedBuf + c, // pointer to the data to decrypt
				decryptedSize - c,   // how many bytes to decrypt
				&zctx); // decryption context

			char fileMAC[10];
			char resMAC[10];
			rc = fcrypt_end(
				(unsigned char*)resMAC, // on return contains the authentication code
				&zctx); // encryption context
			if (rc != 10)
			{
				m_logger.log("Error on encryption closing", ILogger::ELL_ERROR);
				delete[] decryptedBuf;
				return 0;
			}
			{
				read_blocking(m_file.get(), fileMAC, readOffset, 10);
				readOffset += 10;
			}
			if (strncmp(fileMAC, resMAC, 10))
			{
				m_logger.log("Error on encryption check", ILogger::ELL_ERROR);
				delete[] decryptedBuf;
				return 0;
			}
			decrypted = core::make_smart_refctd_ptr<CFileView<VirtualAllocator>>(core::smart_refctd_ptr<ISystem>(m_system), found->fullName, IFile::ECF_READ_WRITE, decryptedSize);

			{
				write_blocking(decrypted.get(), decryptedBuf, 0, decryptedSize);
			}
			actualCompressionMethod = (e.header.Sig & 0xffff);
#if 0
			if ((e.header.Sig & 0xff000000) == 0x01000000)
			{
			}
			else if ((e.header.Sig & 0xff000000) == 0x02000000)
			{
			}
			else
			{
				m_logger.log("Unknown encryption method", ILogger::ELL_ERROR);
				return 0;
			}
#endif
		}
#endif
}
#endif

CFileArchive::file_buffer_t CArchiveLoaderZip::CArchive::getFileBuffer(const IFileArchive::SListEntry* item)
{
	CFileArchive::file_buffer_t retval = {nullptr,item->size,nullptr};

	// Nabla supports 0, 8, 12, 14, 99
	//0 - The file is stored (no compression)
	//1 - The file is Shrunk
	//2 - The file is Reduced with compression factor 1
	//3 - The file is Reduced with compression factor 2
	//4 - The file is Reduced with compression factor 3
	//5 - The file is Reduced with compression factor 4
	//6 - The file is Imploded
	//7 - Reserved for Tokenizing compression algorithm
	//8 - The file is Deflated
	//9 - Reserved for enhanced Deflating
	//10 - PKWARE Date Compression Library Imploding
	//12 - bzip2 - Compression Method from libbz2, WinZip 10
	//14 - LZMA - Compression Method, WinZip 12
	//96 - Jpeg compression - Compression Method, WinZip 12
	//97 - WavPack - Compression Method, WinZip 11
	//98 - PPMd - Compression Method, WinZip 10
	//99 - AES encryption, WinZip 9
	int16_t actualCompressionMethod = m_itemsMetadata[item->ID].CompressionMethod;

	//
	void* decrypted = nullptr;
	size_t decryptedSize = 0ull;
	auto freeOnFail = core::makeRAIIExiter([&actualCompressionMethod,&retval,&decrypted,&decryptedSize](){
		if (decrypted && retval.buffer!=decrypted)
		{
			if (actualCompressionMethod)
				CPlainHeapAllocator(nullptr).dealloc(decrypted,decryptedSize);
			else
				VirtualMemoryAllocator(nullptr).dealloc(decrypted,decryptedSize);
		}
	});
	//
	void* decompressed = nullptr;
	auto freeMMappedOnFail = core::makeRAIIExiter([item,&retval,&decompressed](){
		if (decompressed && retval.buffer!=decompressed)
			VirtualMemoryAllocator(nullptr).dealloc(decompressed,item->size);
	});

	const auto* const cFile = m_file.get();
	void* const filePtr = const_cast<void*>(cFile->getMappedPointer());
	std::byte* const mmapPtr = reinterpret_cast<std::byte*>(filePtr)+item->offset;

	// decrypt
	if (false)
	{
		if (actualCompressionMethod)
			decrypted = CPlainHeapAllocator(nullptr).alloc(decryptedSize);
		else
			decrypted = VirtualMemoryAllocator(nullptr).alloc(decryptedSize);
	}
	//
	if (actualCompressionMethod)
	{
		// TODO
		//const uint32_t uncompressedSize = e.header.DataDescriptor.UncompressedSize;
		decompressed = VirtualMemoryAllocator(nullptr).alloc(item->size);
		if (!decompressed)
		{
			m_logger.log("Not enough memory for decompressing %s",ILogger::ELL_ERROR,item->pathRelativeToArchive.string().c_str());
			return retval;
		}
	}
	switch (actualCompressionMethod)
	{
		case 0: // no compression
			if (decrypted)
				retval.buffer = decrypted;
			else
				retval.buffer = mmapPtr;
			break;
		case 8:
		{
		#ifdef _NBL_COMPILE_WITH_ZLIB_
			// Setup the inflate stream.
			z_stream stream;
			stream.next_in = (Bytef*)(decrypted ? decrypted:mmapPtr);
			stream.avail_in = (uInt)decryptedSize;
			stream.next_out = (Bytef*)decompressed;
			stream.avail_out = item->size;
			stream.zalloc = (alloc_func)0;
			stream.zfree = (free_func)0;

			// Perform inflation. wbits < 0 indicates no zlib header inside the data.
			int32_t err = inflateInit2(&stream, -MAX_WBITS);
			if (err==Z_OK)
			{
				err = inflate(&stream,Z_FINISH);
				inflateEnd(&stream);
				if (err==Z_STREAM_END)
					err = Z_OK;
				err = Z_OK;
				inflateEnd(&stream);
			}

			if (err==Z_OK)
				retval.buffer = decompressed;
		#else
			m_logger.log("ZLIB decompression not supported. File cannot be read.",ILogger::ELL_ERROR);
		#endif
			break;
		}
		case 12:
		{
		#ifdef _NBL_COMPILE_WITH_BZIP2_
			bz_stream bz_ctx = { 0 };
			// use BZIP2's default memory allocation
			//bz_ctx->bzalloc = NULL;
			//bz_ctx->bzfree  = NULL;
			//bz_ctx->opaque  = NULL;
			int err = BZ2_bzDecompressInit(&bz_ctx, 0, 0);
			if (err==BZ_OK)
			{
				bz_ctx.next_in = (char*)(decrypted ? decrypted:mmapPtr);
				bz_ctx.avail_in = decryptedSize;
				bz_ctx.next_out = (char*)decompressed;
				bz_ctx.avail_out = item->size;
				err = BZ2_bzDecompress(&bz_ctx);
				err = BZ2_bzDecompressEnd(&bz_ctx);
			}
			
			if (err==BZ_OK)
				retval.buffer = decompressed;
		#else
			m_logger.log("bzip2 decompression not supported. File cannot be read.", ILogger::ELL_ERROR);
		#endif
			break;
		}
		case 14:
		{
		#ifdef _NBL_COMPILE_WITH_LZMA_
			ELzmaStatus status;
			SizeT tmpDstSize = item->size;
			SizeT tmpSrcSize = decryptedSize;

			const Byte* pcData = decrypted ? reinterpret_cast<std::byte*>(decrypted):mmapPtr;
			const uint32_t propSize = (uint32_t(pcData[3])<<8)+pcData[2];
			int err = LzmaDecode(
				(Byte*)decompressed,
				&tmpDstSize,
				pcData + sizeof(uint32_t) + propSize,
				&tmpSrcSize,
				pcData + sizeof(uint32_t), propSize,
				(m_itemsMetadata[item->ID].GeneralBitFlag&0x1u) ? LZMA_FINISH_END:LZMA_FINISH_ANY, &status,
				&lzmaAlloc
			);

			if (err==SZ_OK)
			{
				retval.buffer = decompressed;
				retval.size = tmpDstSize; // may be different to expected value
			}
		#else
					m_logger.log("lzma decompression not supported. File cannot be read.", ILogger::ELL_ERROR);
		#endif
			break;
		}
		case 99:
			// If we come here with an encrypted file, decryption support is missing
			m_logger.log("Decryption support not enabled. File cannot be read.",ILogger::ELL_ERROR);
			break;
		default:
			m_logger.log("File has unsupported compression method.",ILogger::ELL_ERROR);
			break;
	}

	if (!retval.buffer)
	{
		if (actualCompressionMethod)
			m_logger.log("Error decompressing %s",ILogger::ELL_ERROR,item->pathRelativeToArchive.string().c_str());
		else
			m_logger.log("Unknown error opening file %s from ZIP archive",ILogger::ELL_ERROR,item->pathRelativeToArchive.string().c_str());
	}

	return retval;
}



#ifdef _NBL_COMPILE_WITH_LZMA_
//! Used for LZMA decompression. The lib has no default memory management
namespace
{
	void *SzAlloc(void *p, size_t size) { p = p; return _NBL_ALIGNED_MALLOC(size,_NBL_SIMD_ALIGNMENT); }
	void SzFree(void *p, void *address) { p = p; _NBL_ALIGNED_FREE(address); }
	ISzAlloc lzmaAlloc = { SzAlloc, SzFree };
}
#endif
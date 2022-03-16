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
	static inline constexpr uint32_t ExpectedSig = 0x06054b50u;

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
	uint16_t ID;
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

		//
		auto addItem = [&items,&itemsMetadata](const std::string& _path, const size_t offset, const SZIPFileHeader& meta) -> void
		{
			// we need to have a filename or we skip
			if (_path.empty())
				return;

			auto& item = items.emplace_back();
			item.pathRelativeToArchive = _path;
			item.size = meta.DataDescriptor.UncompressedSize;
			item.offset = offset;
			item.ID = itemsMetadata.size();
			item.allocatorType = meta.CompressionMethod ? IFileArchive::EAT_VIRTUAL_ALLOC:IFileArchive::EAT_NULL;
			itemsMetadata.push_back(meta);
		};

		//
		size_t offset = 0ull;
		auto readStringFromFile = [&file,&offset](auto charCallback) -> bool
		{
			char c = 0x45; // make sure we start with non-zero char
			while (c)
			{
				IFile::success_t success;
				file->read(success,&c,offset,sizeof(c));
				if (!success)
					return false;
				offset += success.getSizeToProcess();
				charCallback(c);
			}
			// if string is not null terminated, something went wrong reading the file
			return !c;
		};
		
		//
		std::string filename;
		filename.reserve(ISystem::MAX_FILENAME_LENGTH);
		if (isGZip)
		{
			SGZIPMemberHeader gzipHeader;
			{
				IFile::success_t success;
				file->read(success,&gzipHeader,0ull,sizeof(gzipHeader));
				if (!success)
					return nullptr;
				offset += success.getSizeToProcess();
			}

			//! The gzip file format seems to think that there can be multiple files in a gzip file
			//! TODO: But OLD Irrlicht Impl doesn't honor it!?
			if (gzipHeader.sig!=0x8b1fu)
				return nullptr;
			
			// now get the file info
			if (gzipHeader.flags&EGZF_EXTRA_FIELDS)
			{
				// read lenth of extra data
				uint16_t dataLen;
				IFile::success_t success;
				file->read(success,&dataLen,offset,sizeof(dataLen));
				if (!success)
					return nullptr;
				offset += success.getSizeToProcess();
				// skip the extra data
				offset += dataLen;
			}
			//
			if (gzipHeader.flags&EGZF_FILE_NAME)
			{
				filename.clear();
				if (!readStringFromFile([&](const char c){filename.push_back(c);}))
					return nullptr;
			}
			//
			if (gzipHeader.flags&EGZF_COMMENT)
			{
				if (!readStringFromFile([](const char c){}))
					return nullptr;
			}
			// skip crc16
			if (gzipHeader.flags&EGZF_CRC16)
				offset += 2;


			SZIPFileHeader header;
			memset(&header,0,sizeof(SZIPFileHeader));
			header.FilenameLength = filename.length();
			header.CompressionMethod = gzipHeader.compressionMethod;
			header.DataDescriptor.CompressedSize = file->getSize()-(offset+sizeof(uint64_t));

			const size_t itemOffset = offset;
			
			offset += header.DataDescriptor.CompressedSize;
			// read CRC
			{
				IFile::success_t success;
				file->read(success,&header.DataDescriptor.CRC32,offset,sizeof(header.DataDescriptor.CRC32));
				if (!success)
					return nullptr;
				offset += success.getSizeToProcess();
			}
			// read uncompressed size
			{
				IFile::success_t success;
				file->read(success,&header.DataDescriptor.UncompressedSize,offset,sizeof(header.DataDescriptor.UncompressedSize));
				if (!success)
					return nullptr;
				offset += success.getSizeToProcess();
			}

			//
			addItem(filename,itemOffset,header);
		}
		else
		{
			while (true)
			{
				SZIPFileHeader zipHeader;
				{
					IFile::success_t success;
					file->read(success,&zipHeader,offset,sizeof(zipHeader));
					if (!success)
						break;
					offset += success.getSizeToProcess();
				}

				if (zipHeader.Sig!=0x04034b50u)
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
				if ((zipHeader.GeneralBitFlag&ZIP_FILE_ENCRYPTED) && (zipHeader.CompressionMethod==99))
				{
					SZipFileExtraHeader extraHeader;
					SZipFileAESExtraData data;

					size_t localOffset = offset;
					offset += zipHeader.ExtraFieldLength;
					while (true)
					{
						{
							IFile::success_t success;
							file->read(success,&extraHeader,localOffset,sizeof(extraHeader));
							if (!success)
								break;
							localOffset += success.getSizeToProcess();
							if (localOffset>offset)
								break;
						}

						if (extraHeader.ID!=0x9901u)
							continue;

						{
							IFile::success_t success;
							file->read(success,&data,localOffset,sizeof(data));
							if (!success)
								break;
							localOffset += success.getSizeToProcess();
							if (localOffset>offset)
								break;
						}
						if (data.Vendor[0]=='A' && data.Vendor[1]=='E')
						{
							#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
							// encode values into Sig
							// AE-Version | Strength | ActualMode
							zipHeader.Sig =
								((data.Version & 0xff) << 24) |
								(data.EncryptionStrength << 16) |
								(data.CompressionMode);
							break;
							#else
							filename.clear(); // no support, can't decrypt
							#endif
						}
					}
				}
				else
					offset += zipHeader.ExtraFieldLength;

				// if bit 3 was set, use CentralDirectory for setup
				if (zipHeader.GeneralBitFlag&ZIP_INFO_IN_DATA_DESCRIPTOR)
				{
					SZIPFileCentralDirEnd dirEnd;
					dirEnd.Sig = 0u;

					// First place where the end record could be stored
					offset = file->getSize()-sizeof(SZIPFileCentralDirEnd)+1ull;
					while (dirEnd.Sig!=SZIPFileCentralDirEnd::ExpectedSig)
					{
						IFile::success_t success;
						file->read(success,&dirEnd,--offset,sizeof(dirEnd));
						if (!success)
							return nullptr;
					}
					items.reserve(dirEnd.TotalEntries);
					itemsMetadata.reserve(dirEnd.TotalEntries);
					offset = dirEnd.Offset;
					#if 0
					while (scanCentralDirectoryHeader(offset)) {}
					#endif
					assert(false); // if you ever hit this, msg @devsh
					break;
				}
			
				addItem(filename,offset,zipHeader);
				// move forward length of data
				offset += zipHeader.DataDescriptor.CompressedSize;
			}
		}
	}

	assert(items.size()==itemsMetadata.size());
	if (items.empty())
		return nullptr;

	return core::make_smart_refctd_ptr<CArchive>(std::move(file),core::smart_refctd_ptr(m_logger.get()),std::move(items),std::move(itemsMetadata));
}

#if 0
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
#endif

CFileArchive::file_buffer_t CArchiveLoaderZip::CArchive::getFileBuffer(const IFileArchive::SListEntry* item)
{
	const auto& header = m_itemsMetadata[item->ID];
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
	int16_t actualCompressionMethod = header.CompressionMethod;

	CFileArchive::file_buffer_t retval = { nullptr,item->size,nullptr };
	//
	void* decrypted = nullptr;
	size_t decryptedSize = header.DataDescriptor.CompressedSize;
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
	std::byte* mmapPtr = reinterpret_cast<std::byte*>(filePtr)+item->offset;

	// decrypt
	if ((header.GeneralBitFlag&ZIP_FILE_ENCRYPTED) && (header.CompressionMethod==99))
	{
		const uint8_t* salt = reinterpret_cast<uint8_t*>(mmapPtr);
	#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
		const uint16_t saltSize = ((header.Sig>>14u)&0x3fcu)+4u;
		{
			const size_t reduction = saltSize+12u;
			if (header.DataDescriptor.CompressedSize<=reduction)
				return retval;
			decryptedSize -= reduction;
		}
		mmapPtr += saltSize;
		uint16_t& pwVerification = *reinterpret_cast<uint16_t*>(mmapPtr);
		mmapPtr += 2u;
		uint16_t pwVerificationFile;

		fcrypt_ctx zctx; // the encryption context
		int rc = fcrypt_init(
			(header.Sig>>16u)&0xffu,
			(const unsigned char*)m_password.c_str(), // the password
			m_password.size(), // number of bytes in password
			salt, // the salt
			(unsigned char*)&pwVerificationFile, // on return contains password verifier
			&zctx
		); // encryption context
		if (pwVerification!=pwVerificationFile)
		{
			m_logger.log("Wrong password for ZIP Archive.",ILogger::ELL_ERROR);
			return retval;
		}

		if (actualCompressionMethod)
			decrypted = CPlainHeapAllocator(nullptr).alloc(decryptedSize);
		else
			decrypted = VirtualMemoryAllocator(nullptr).alloc(decryptedSize);
		if (!decrypted)
	#endif
			return retval;
	#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
		constexpr uint32_t kChunkSize = 0x8000u;
		for (size_t offset=0u; offset<decryptedSize; offset+=kChunkSize)
		{
			const uint32_t size = core::min<size_t>(decryptedSize-offset,kChunkSize);
			fcrypt_decrypt(
				reinterpret_cast<unsigned char*>(decrypted)+offset, // pointer to the data to decrypt
				size,   // how many bytes to decrypt
				&zctx // decryption context
			);
			mmapPtr += size;
		}
		char resMAC[10];
		rc = fcrypt_end(
			(unsigned char*)resMAC, // on return contains the authentication code
			&zctx // encryption context
		);
		if (rc != 10)
		{
			m_logger.log("Error on encryption closing",ILogger::ELL_ERROR);
			return retval;
		}
		const char* fileMAC = reinterpret_cast<char*>(mmapPtr);
		mmapPtr += 10;
		if (strncmp(fileMAC,resMAC,10))
		{
			m_logger.log("Error on encryption check", ILogger::ELL_ERROR);
			return retval;
		}

		actualCompressionMethod = (header.Sig & 0xffff);
	#endif
	}
	//
	if (actualCompressionMethod)
	{
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
				(header.GeneralBitFlag&0x1u) ? LZMA_FINISH_END:LZMA_FINISH_ANY, &status,
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
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
	nbl::system::CArchiveLoaderZip::SZIPFileDataDescriptor DataDescriptor;
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

	static constexpr uint32_t ExpectedSignature = 0x02014b50u;

	size_t calcSize() const
	{
		return sizeof(SZIPFileCentralDirFileHeader) + FilenameLength + ExtraFieldLength + FileCommentLength;
	}
} PACK_STRUCT;

static_assert(sizeof(SZIPFileCentralDirFileHeader) == 46);

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

	static constexpr uint16_t ExpectedSignature = 0x8b1fu;
} PACK_STRUCT;
#include "nbl/nblunpack.h"

static_assert(sizeof(SGZIPMemberHeader) == 10);

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

core::smart_refctd_ptr<IFileArchive> CArchiveLoaderZip::createArchiveFromGZIP(core::smart_refctd_ptr<system::IFile>&& file, const std::string_view& password) const
{
	std::shared_ptr<core::vector<IFileArchive::SFileList::SEntry>> items = std::make_shared<core::vector<IFileArchive::SFileList::SEntry>>();
	core::vector<SZIPFileHeader> itemsMetadata;

	items->reserve(1u);
	itemsMetadata.reserve(1u);
	auto addItem = [&items, &itemsMetadata](const std::string& _path, const size_t offset, const SZIPFileHeader& meta) -> void
	{
		// we need to have a filename or we skip
		if (_path.empty())
			return;

		auto& item = items->emplace_back();
		item.pathRelativeToArchive = _path;
		item.size = meta.DataDescriptor.UncompressedSize;
		item.offset = offset;
		item.ID = itemsMetadata.size();
		item.allocatorType = meta.CompressionMethod ? IFileArchive::EAT_VIRTUAL_ALLOC : IFileArchive::EAT_NULL;
		itemsMetadata.push_back(meta);
	};

	std::string filename;
	size_t gzipFileOffset = 0ull;
	auto readStringFromFile = [&file, &gzipFileOffset](auto charCallback) -> bool
		{
			char c = 0x45; // make sure we start with non-zero char
			while (c)
			{
				IFile::success_t success;
				file->read(success, &c, gzipFileOffset, sizeof(c));
				if (!success)
					return false;
				gzipFileOffset += success.getBytesToProcess();
				charCallback(c);
			}
			// if string is not null terminated, something went wrong reading the file
			return !c;
		};

	SGZIPMemberHeader gzipHeader;
	{
		IFile::success_t success;
		file->read(success, &gzipHeader, gzipFileOffset, sizeof(gzipHeader));
		if (!success)
			return nullptr;
		gzipFileOffset += sizeof(gzipHeader);
	}

	//! The gzip file format seems to think that there can be multiple files in a gzip file
	//! TODO: But OLD Irrlicht Impl doesn't honor it!?
	if (gzipHeader.sig != SGZIPMemberHeader::ExpectedSignature)
		return nullptr;

	// now get the file info
	if (gzipHeader.flags & EGZF_EXTRA_FIELDS)
	{
		// read lenth of extra data
		uint16_t dataLen;
		IFile::success_t success;
		file->read(success, &dataLen, gzipFileOffset, sizeof(dataLen));
		if (!success)
			return nullptr;
		gzipFileOffset += success.getBytesToProcess();
		// skip the extra data
		gzipFileOffset += dataLen;
	}
	//
	if (gzipHeader.flags & EGZF_FILE_NAME)
	{
		filename.clear();
		if (!readStringFromFile([&](const char c) {filename.push_back(c); }))
			return nullptr;
	}
	//
	if (gzipHeader.flags & EGZF_COMMENT)
	{
		if (!readStringFromFile([](const char c) {}))
			return nullptr;
	}
	// skip crc16
	if (gzipHeader.flags & EGZF_CRC16)
		gzipFileOffset += 2;


	SZIPFileHeader header{};
	header.FilenameLength = filename.length();
	header.CompressionMethod = gzipHeader.compressionMethod;
	header.DataDescriptor.CompressedSize = file->getSize() - (gzipFileOffset + sizeof(uint64_t));

	const size_t itemOffset = gzipFileOffset;

	gzipFileOffset += header.DataDescriptor.CompressedSize;
	// read CRC
	{
		IFile::success_t success;
		file->read(success, &header.DataDescriptor.CRC32, gzipFileOffset, sizeof(header.DataDescriptor.CRC32));
		if (!success)
			return nullptr;
		gzipFileOffset += success.getBytesToProcess();
	}
	// read uncompressed size
	{
		IFile::success_t success;
		file->read(success, &header.DataDescriptor.UncompressedSize, gzipFileOffset, sizeof(header.DataDescriptor.UncompressedSize));
		if (!success)
			return nullptr;
		gzipFileOffset += success.getBytesToProcess();
	}

	//
	addItem(filename, itemOffset, header);

	assert(items->size() == itemsMetadata.size());
	if (items->empty())
		return nullptr;

	return core::make_smart_refctd_ptr<CArchive>(std::move(file), core::smart_refctd_ptr(m_logger.get()), items, std::move(itemsMetadata));
}
core::smart_refctd_ptr<IFileArchive> CArchiveLoaderZip::createArchiveFromZIP(core::smart_refctd_ptr<system::IFile>&& file, const std::string_view& password) const
{
	const size_t fileSize = file->getSize();
	if (fileSize < sizeof(SZIPFileCentralDirEnd))
		return nullptr;

	SZIPFileCentralDirEnd dirEnd;
	{
		dirEnd.Sig = 0u;
		constexpr size_t kMaxZipCommentSize = 0xffffu;
		size_t endOfCentralDirectoryOffset = fileSize - sizeof(SZIPFileCentralDirEnd);
		const size_t minEndOffset = (fileSize > sizeof(SZIPFileCentralDirEnd) + kMaxZipCommentSize) ? (fileSize - sizeof(SZIPFileCentralDirEnd) - kMaxZipCommentSize) : 0u;
		bool found = false;
		while (true)
		{
			IFile::success_t success;
			file->read(success, &dirEnd, endOfCentralDirectoryOffset, sizeof(dirEnd));
			if (success && dirEnd.Sig == SZIPFileCentralDirEnd::ExpectedSig)
			{
				found = true;
				break;
			}
			if (endOfCentralDirectoryOffset == minEndOffset)
				break;
			--endOfCentralDirectoryOffset;
		}
		if (!found)
			return nullptr;
	}

	// multiple disks are not supported
	if (dirEnd.NumberDisk != 0 || dirEnd.NumberStart != 0)
	{
		assert(false);
		return nullptr;
	}
	if (dirEnd.Offset > fileSize || dirEnd.Size > fileSize - dirEnd.Offset)
		return nullptr;

	std::shared_ptr<core::vector<IFileArchive::SFileList::SEntry>> items = std::make_shared<core::vector<IFileArchive::SFileList::SEntry>>();
	core::vector<SZIPFileHeader> itemsMetadata;

	items->reserve(dirEnd.TotalEntries);
	itemsMetadata.reserve(dirEnd.TotalEntries);
	auto addItem = [&items, &itemsMetadata](const std::string& _path, const size_t offset, const SZIPFileHeader& meta) -> void
	{
		// we need to have a filename or we skip
		if (_path.empty())
			return;

		auto& item = items->emplace_back();
		item.pathRelativeToArchive = _path;
		item.size = meta.DataDescriptor.UncompressedSize;
		item.offset = offset;
		item.ID = itemsMetadata.size();
		item.allocatorType = meta.CompressionMethod ? IFileArchive::EAT_VIRTUAL_ALLOC : IFileArchive::EAT_NULL;
		itemsMetadata.push_back(meta);
	};

	size_t centralDirectoryOffset = dirEnd.Offset;
	for (int i = 0; i < dirEnd.TotalEntries; ++i)
	{
		SZIPFileCentralDirFileHeader centralDirectoryHeader;
		{
			if (centralDirectoryOffset > fileSize || fileSize - centralDirectoryOffset < sizeof(SZIPFileCentralDirFileHeader))
				return nullptr;
			IFile::success_t success;
			file->read(success, &centralDirectoryHeader, centralDirectoryOffset, sizeof(SZIPFileCentralDirFileHeader));
			if (!success)
				return nullptr;
		}

		if (centralDirectoryHeader.Sig != SZIPFileCentralDirFileHeader::ExpectedSignature)
		{
			// .zip file is corrupted
			assert(false);
			return nullptr;
		}

		const size_t centralHeaderSize = centralDirectoryHeader.calcSize();
		if (centralHeaderSize < sizeof(SZIPFileCentralDirFileHeader))
			return nullptr;
		if (centralDirectoryOffset + centralHeaderSize > fileSize)
			return nullptr;
		centralDirectoryOffset += centralHeaderSize;

		SZIPFileHeader localFileHeader;
		{
			const size_t localHeaderOffset = centralDirectoryHeader.RelativeOffsetOfLocalHeader;
			if (localHeaderOffset > fileSize || fileSize - localHeaderOffset < sizeof(SZIPFileHeader))
				return nullptr;
			IFile::success_t success;
			file->read(success, &localFileHeader, localHeaderOffset, sizeof(SZIPFileHeader));
			if (!success)
				return nullptr;
		}
		if (localFileHeader.Sig != SZIPFileHeader::ExpectedSignature)
			return nullptr;
		const size_t localHeaderSize = localFileHeader.calcSize();
		if (localHeaderSize < sizeof(SZIPFileHeader))
			return nullptr;
		if (centralDirectoryHeader.RelativeOffsetOfLocalHeader + localHeaderSize > fileSize)
			return nullptr;

		std::string filename;
		filename.resize(localFileHeader.FilenameLength);
		{
			IFile::success_t success;
			const size_t filenameOffset = centralDirectoryHeader.RelativeOffsetOfLocalHeader + sizeof(SZIPFileHeader);
			file->read(success, filename.data(), filenameOffset, localFileHeader.FilenameLength);
			// TODO: assertion
			if (!success)
				return nullptr;
		}
		
		// AES encryption
		if ((localFileHeader.GeneralBitFlag & ZIP_FILE_ENCRYPTED) && (localFileHeader.CompressionMethod == 99))
		{
			SZipFileExtraHeader extraHeader;
			SZipFileAESExtraData data;

			size_t localOffset = centralDirectoryHeader.RelativeOffsetOfLocalHeader + sizeof(SZIPFileHeader) + localFileHeader.FilenameLength;
			size_t offset = localOffset + localFileHeader.ExtraFieldLength;
			while (localOffset + sizeof(extraHeader) <= offset)
			{
				{
					IFile::success_t success;
					file->read(success, &extraHeader, localOffset, sizeof(extraHeader));
					if (!success)
						break;
					localOffset += sizeof(extraHeader);
				}

				if (extraHeader.Size < 0)
					break;
				const size_t extraSize = static_cast<uint16_t>(extraHeader.Size);
				if (extraSize == 0)
					break;
				if (localOffset + extraSize > offset)
					break;

				if (extraHeader.ID != 0x9901u)
				{
					localOffset += extraSize;
					continue;
				}

				if (extraSize < sizeof(SZipFileAESExtraData))
					break;
				{
					IFile::success_t success;
					file->read(success, &data, localOffset, sizeof(data));
					if (!success)
						break;
				}
				if (data.Vendor[0] == 'A' && data.Vendor[1] == 'E')
				{
#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
					// encode values into Sig
					// AE-Version | Strength | ActualMode
					localFileHeader.Sig =
						((data.Version & 0xff) << 24) |
						(data.EncryptionStrength << 16) |
						(data.CompressionMode);
					break;
#else
					filename.clear(); // no support, can't decrypt
#endif
				}
				localOffset += extraSize;
			}
		}

		// copying the data descriptor from the central directory header because it is always valid (data descriptor from local file header may be invalid when bit 3 in general purpose bit flag is set)
		localFileHeader.DataDescriptor = centralDirectoryHeader.DataDescriptor;

		const size_t fileDataOffset = centralDirectoryHeader.RelativeOffsetOfLocalHeader + localFileHeader.calcSize();
		addItem(filename, fileDataOffset, localFileHeader);
	}

	assert(items->size() == itemsMetadata.size());
	if (items->empty())
		return nullptr;

	return core::make_smart_refctd_ptr<CArchive>(std::move(file), core::smart_refctd_ptr(m_logger.get()), items, std::move(itemsMetadata));
}

core::smart_refctd_ptr<IFileArchive> CArchiveLoaderZip::createArchive_impl(core::smart_refctd_ptr<system::IFile>&& file, const std::string_view& password) const
{
	if (!file)
		return nullptr;

	uint16_t sig;
	IFile::success_t success;
	file->read(success, &sig, 0, sizeof(sig));
	if (!success)
		return nullptr;

	const bool isGZIP = sig == SGZIPMemberHeader::ExpectedSignature;
	if (isGZIP)
	{
		return createArchiveFromGZIP(std::move(file), password);
	}
	else
	{
		return createArchiveFromZIP(std::move(file), password);
	}
}

CFileArchive::file_buffer_t CArchiveLoaderZip::CArchive::getFileBuffer(const IFileArchive::SFileList::found_t& item)
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

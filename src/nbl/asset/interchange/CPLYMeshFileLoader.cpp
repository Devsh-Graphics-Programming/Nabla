// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifdef _NBL_COMPILE_WITH_PLY_LOADER_


#include "CPLYMeshFileLoader.h"
#include "nbl/asset/metadata/CPLYMetadata.h"

#include <numeric>
#include <charconv>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>
#include <fast_float/fast_float.h>

#include "nbl/asset/IAssetManager.h"

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"

//#include "nbl/asset/utils/IMeshManipulator.h"


namespace nbl::asset
{

CPLYMeshFileLoader::CPLYMeshFileLoader() = default;

const char** CPLYMeshFileLoader::getAssociatedFileExtensions() const
{
	static const char* ext[] = { "ply", nullptr };
	return ext;
}

bool CPLYMeshFileLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
{
    char buf[40];

	system::IFile::success_t success;
	_file->read(success,buf,0,sizeof(buf));
	if (!success)
		return false;

    char* header = buf;
    if (strncmp(header,"ply",3u)!=0)
        return false;
    
    header += 4;
    char* lf = strstr(header,"\n");
    if (!lf)
        return false;
	
    constexpr std::array<std::string_view,3> headers = {
        "format ascii 1.0",
        "format binary_little_endian 1.0",
        "format binary_big_endian 1.0"
    };
	return std::find(headers.begin(),headers.end(),std::string_view(header,lf))!=headers.end();
}

template<typename T>
T byteswap(const T& v)
{
	T retval;
	auto it = reinterpret_cast<const char*>(&v);
	std::reverse_copy(it,it+sizeof(T),reinterpret_cast<char*>(&retval));
	return retval;
}

template<typename Fn>
void plyRunParallelWorkers(const size_t workerCount, Fn&& fn)
{
	if (workerCount <= 1ull)
	{
		fn(0ull);
		return;
	}
	std::vector<std::thread> workers;
	workers.reserve(workerCount - 1ull);
	for (size_t workerIx = 1ull; workerIx < workerCount; ++workerIx)
	{
		workers.emplace_back([&fn, workerIx]()
		{
			fn(workerIx);
		});
	}
	fn(0ull);
	for (auto& worker : workers)
		worker.join();
}

class CPLYMappedFileMemoryResource final : public core::refctd_memory_resource
{
	public:
		explicit CPLYMappedFileMemoryResource(core::smart_refctd_ptr<system::IFile>&& file) : m_file(std::move(file))
		{
		}

		inline void* allocate(std::size_t bytes, std::size_t alignment) override
		{
			(void)bytes;
			(void)alignment;
			assert(false);
			return nullptr;
		}

		inline void deallocate(void* p, std::size_t bytes, std::size_t alignment) override
		{
			(void)p;
			(void)bytes;
			(void)alignment;
		}

	private:
		core::smart_refctd_ptr<system::IFile> m_file;
};

IGeometry<ICPUBuffer>::SDataView plyCreateMappedF32x3View(system::IFile* file, void* ptr, const size_t byteCount)
{
	if (!file || !ptr || byteCount == 0ull)
		return {};

	auto keepAliveResource = core::make_smart_refctd_ptr<CPLYMappedFileMemoryResource>(core::smart_refctd_ptr<system::IFile>(file));
	auto buffer = ICPUBuffer::create({
		{ byteCount },
		ptr,
		core::smart_refctd_ptr<core::refctd_memory_resource>(keepAliveResource),
		alignof(float)
	}, core::adopt_memory);
	if (!buffer)
		return {};

	IGeometry<ICPUBuffer>::SDataView view = {
		.composed = {
			.stride = sizeof(float) * 3ull,
			.format = EF_R32G32B32_SFLOAT,
			.rangeFormat = IGeometryBase::getMatchingAABBFormat(EF_R32G32B32_SFLOAT)
		},
		.src = {
			.offset = 0ull,
			.size = byteCount,
			.buffer = std::move(buffer)
		}
	};
	return view;
}

IGeometry<ICPUBuffer>::SDataView plyCreateAdoptedU32IndexView(core::vector<uint32_t>&& indices)
{
	if (indices.empty())
		return {};

	auto backer = core::make_smart_refctd_ptr<core::adoption_memory_resource<core::vector<uint32_t>>>(std::move(indices));
	auto& storage = backer->getBacker();
	auto* const ptr = storage.data();
	const size_t byteCount = storage.size() * sizeof(uint32_t);
	auto buffer = ICPUBuffer::create({ { byteCount }, ptr, core::smart_refctd_ptr<core::refctd_memory_resource>(std::move(backer)), alignof(uint32_t) }, core::adopt_memory);
	if (!buffer)
		return {};

	IGeometry<ICPUBuffer>::SDataView view = {
		.composed = {
			.stride = sizeof(uint32_t),
			.format = EF_R32_UINT,
			.rangeFormat = IGeometryBase::getMatchingAABBFormat(EF_R32_UINT)
		},
		.src = {
			.offset = 0u,
			.size = byteCount,
			.buffer = std::move(buffer)
		}
	};
	return view;
}

IGeometry<ICPUBuffer>::SDataView plyCreateAdoptedU16IndexView(core::vector<uint16_t>&& indices)
{
	if (indices.empty())
		return {};

	auto backer = core::make_smart_refctd_ptr<core::adoption_memory_resource<core::vector<uint16_t>>>(std::move(indices));
	auto& storage = backer->getBacker();
	auto* const ptr = storage.data();
	const size_t byteCount = storage.size() * sizeof(uint16_t);
	auto buffer = ICPUBuffer::create({ { byteCount }, ptr, core::smart_refctd_ptr<core::refctd_memory_resource>(std::move(backer)), alignof(uint16_t) }, core::adopt_memory);
	if (!buffer)
		return {};

	IGeometry<ICPUBuffer>::SDataView view = {
		.composed = {
			.stride = sizeof(uint16_t),
			.format = EF_R16_UINT,
			.rangeFormat = IGeometryBase::getMatchingAABBFormat(EF_R16_UINT)
		},
		.src = {
			.offset = 0u,
			.size = byteCount,
			.buffer = std::move(buffer)
		}
	};
	return view;
}

void plyRecomputeContentHashesParallel(ICPUPolygonGeometry* geometry)
{
	if (!geometry)
		return;

	core::vector<core::smart_refctd_ptr<ICPUBuffer>> buffers;
	auto appendViewBuffer = [&buffers](const IGeometry<ICPUBuffer>::SDataView& view) -> void
	{
		if (!view || !view.src.buffer)
			return;
		for (const auto& existing : buffers)
		{
			if (existing.get() == view.src.buffer.get())
				return;
		}
		buffers.push_back(core::smart_refctd_ptr<ICPUBuffer>(view.src.buffer));
	};

	appendViewBuffer(geometry->getPositionView());
	appendViewBuffer(geometry->getIndexView());
	appendViewBuffer(geometry->getNormalView());
	for (const auto& view : *geometry->getAuxAttributeViews())
		appendViewBuffer(view);
	for (const auto& view : *geometry->getJointWeightViews())
	{
		appendViewBuffer(view.indices);
		appendViewBuffer(view.weights);
	}
	if (auto jointOBB = geometry->getJointOBBView(); jointOBB)
		appendViewBuffer(*jointOBB);

	for (auto& buffer : buffers)
		buffer->setContentHash(buffer->computeContentHash());
}

struct SContext
{
	static constexpr uint64_t ReadWindowPaddingBytes = 1ull;
	
	//
	struct SProperty
	{
		static E_FORMAT getType(const char* typeString)
		{
			if (strcmp(typeString, "char")==0 || strcmp(typeString, "int8")==0)
				return EF_R8_SINT;
			else if (strcmp(typeString, "uchar")==0 || strcmp(typeString, "uint8")==0)
				return EF_R8_UINT;
			else if (strcmp(typeString, "short")==0 || strcmp(typeString, "int16")==0)
				return EF_R16_SINT;
			else if (strcmp(typeString, "ushort")==0 || strcmp(typeString, "uint16")==0)
				return EF_R16_UINT;
			else if (strcmp(typeString, "long")==0 || strcmp(typeString, "int")==0 || strcmp(typeString, "int32")==0)
				return EF_R32_SINT;
			else if (strcmp(typeString, "ulong")==0 || strcmp(typeString, "uint")==0 || strcmp(typeString, "uint32")==0)
				return EF_R32_UINT;
			else if (strcmp(typeString, "float")==0 || strcmp(typeString, "float32")==0)
				return EF_R32_SFLOAT;
			else if (strcmp(typeString, "double")==0 || strcmp(typeString, "float64")==0)
				return EF_R64_SFLOAT;
			else
				return EF_UNKNOWN;
		}

		bool isList() const {return type==EF_UNKNOWN && asset::isIntegerFormat(list.countType) && asset::isIntegerFormat(list.itemType);}

		void skip(SContext& _ctx) const
		{
			if (isList())
			{
				int32_t count = _ctx.getInt(list.countType);

				for (decltype(count) i=0; i<count; ++i)
					_ctx.getInt(list.itemType);
			}
			else if (_ctx.IsBinaryFile)
				_ctx.moveForward(getTexelOrBlockBytesize(type));
			else
				_ctx.getNextWord();
		}

		std::string Name;
		E_FORMAT type;
		struct SListTypes
		{
			E_FORMAT countType;
			E_FORMAT itemType;
		} list;
	};
	struct SElement
	{
		void skipElement(SContext& _ctx) const
		{
			if (_ctx.IsBinaryFile)
			{
				if (KnownSize)
					_ctx.moveForward(KnownSize);
				else
				for (auto i=0u; i<Properties.size(); ++i)
					Properties[i].skip(_ctx);
			}
			else
				_ctx.getNextLine();
		}

		// name of the element. We only want "vertex" and "face" elements
		// but we have to parse the others anyway.
		std::string Name;
		// Properties of this element
		core::vector<SProperty> Properties;
		// The number of elements in the file
		size_t Count;
		// known size in bytes, 0 if unknown
		uint32_t KnownSize;
	};

	static constexpr size_t DefaultIoReadWindowBytes = 50ull << 10;

	void init(size_t _ioReadWindowSize = DefaultIoReadWindowBytes)
	{
		ioReadWindowSize = std::max<size_t>(_ioReadWindowSize, DefaultIoReadWindowBytes);
		Buffer.resize(ioReadWindowSize + ReadWindowPaddingBytes, '\0');
		EndPointer = StartPointer = Buffer.data();
		LineEndPointer = EndPointer-1;
		UsingMappedBinaryWindow = false;

		fillBuffer();
	}

	// gets more data from the file
	void fillBuffer()
	{
		if (EndOfFile)
			return;
		else if (fileOffset>=inner.mainFile->getSize())
		{
			EndOfFile = true;
			return;
		}
		
		const auto length = std::distance(StartPointer,EndPointer);
		auto newStart = Buffer.data();
		// copy the remaining data to the start of the buffer
		if (length && StartPointer!=newStart)
			memmove(newStart,StartPointer,length);
		// reset start position
		StartPointer = newStart;
		EndPointer = newStart+length;

		// read data from the file
		const size_t usableBufferSize = Buffer.size() > 0ull ? Buffer.size() - ReadWindowPaddingBytes : 0ull;
		if (usableBufferSize <= length)
		{
			EndOfFile = true;
			return;
		}
		const size_t requestSize = usableBufferSize - length;
		system::IFile::success_t success;
		inner.mainFile->read(success,EndPointer,fileOffset,requestSize);
		const size_t bytesRead = success.getBytesProcessed();
		++readCallCount;
		readBytesTotal += bytesRead;
		if (bytesRead < readMinBytes)
			readMinBytes = bytesRead;
		fileOffset += bytesRead;
		EndPointer += bytesRead;

		// if we didn't completely fill the buffer
		if (bytesRead!=requestSize)
		{
			// cauterize the string
			*EndPointer = 0;
			EndOfFile = true;
		}
	}
	// Split the string data into a line in place by terminating it instead of copying.
	const char* getNextLine()
	{
		// move the start pointer along
		StartPointer = LineEndPointer+1;

		// crlf split across buffer move
		if (*StartPointer=='\n')
			*(StartPointer++) = '\0';

		// begin at the start of the next line
		const std::array<const char,3> Terminators = { '\0','\r','\n'};
		auto terminator = std::find_first_of(StartPointer,EndPointer,Terminators.begin(),Terminators.end());
		if (terminator!=EndPointer)
			*(terminator++) = '\0';

		// we have reached the end of the buffer
		if (terminator==EndPointer)
		{
			// get data from the file
			if (EndOfFile)
			{
				StartPointer = EndPointer-1;
				*StartPointer = '\0';
				return StartPointer;
			}
			else
			{
				fillBuffer();
				// reset line end pointer
				LineEndPointer = StartPointer-1;
				if (StartPointer!=EndPointer)
					return getNextLine();
				else
					return StartPointer;
			}
		}
		else
		{
			LineEndPointer = terminator-1;
			WordLength = -1;
			// return pointer to the start of the line
			return StartPointer;
		}
	}
	// null terminate the next word on the previous line and move the next word pointer along
	// since we already have a full line in the buffer, we never need to retrieve more data
	const char* getNextWord()
	{
		// move the start pointer along
		StartPointer += WordLength + 1;
		if (!*StartPointer)
			getNextLine();

		if (StartPointer==LineEndPointer)
		{
			WordLength = -1; //
			return LineEndPointer;
		}
		// process the next word
		{
			assert(LineEndPointer<=EndPointer);
			const std::array<const char,3> WhiteSpace = {'\0',' ','\t'};
			auto wordEnd = std::find_first_of(StartPointer,LineEndPointer,WhiteSpace.begin(),WhiteSpace.end());
			// null terminate the next word
			if (wordEnd!=LineEndPointer)
				*(wordEnd++) = '\0';
			// find next word
			auto notWhiteSpace = [WhiteSpace](const char c)->bool
			{
				return std::find(WhiteSpace.begin(),WhiteSpace.end(),c)==WhiteSpace.end();
			};
			auto nextWord = std::find_if(wordEnd,LineEndPointer,notWhiteSpace);
			WordLength = std::distance(StartPointer,nextWord)-1;
		}
		// return pointer to the start of current word
		return StartPointer;
	}
	size_t getAbsoluteOffset(const char* ptr) const
	{
		if (!ptr || ptr > EndPointer)
			return fileOffset;
		const size_t trailingBytes = static_cast<size_t>(EndPointer - ptr);
		return fileOffset >= trailingBytes ? (fileOffset - trailingBytes) : 0ull;
	}
	void useMappedBinaryWindow(const char* data, const size_t sizeBytes)
	{
		if (!data)
			return;
		StartPointer = const_cast<char*>(data);
		EndPointer = StartPointer + sizeBytes;
		LineEndPointer = StartPointer - 1;
		WordLength = -1;
		EndOfFile = true;
		UsingMappedBinaryWindow = true;
		fileOffset = inner.mainFile ? inner.mainFile->getSize() : fileOffset;
	}
	// skips x bytes in the file, getting more data if required
	void moveForward(const size_t bytes)
	{
		assert(IsBinaryFile);
		size_t remaining = bytes;
		if (remaining == 0ull)
			return;

		const size_t availableInitially = EndPointer > StartPointer ? static_cast<size_t>(EndPointer - StartPointer) : 0ull;
		if (remaining > availableInitially)
		{
			remaining -= availableInitially;
			StartPointer = EndPointer;
			if (remaining > ioReadWindowSize)
			{
				const size_t fileSize = inner.mainFile->getSize();
				const size_t fileRemaining = fileSize > fileOffset ? (fileSize - fileOffset) : 0ull;
				const size_t directSkip = std::min(remaining, fileRemaining);
				fileOffset += directSkip;
				remaining -= directSkip;
			}
		}

		while (remaining)
		{
			if (StartPointer >= EndPointer)
			{
				fillBuffer();
				if (StartPointer >= EndPointer)
					return;
			}

			const size_t available = static_cast<size_t>(EndPointer - StartPointer);
			const size_t step = std::min(available, remaining);
			StartPointer += step;
			remaining -= step;
		}
	}

	// read the next int from the file and move the start pointer along
	using widest_int_t = uint32_t;
	widest_int_t getInt(const E_FORMAT f)
	{
		assert(!isFloatingPointFormat(f));
		if (IsBinaryFile)
		{
			if (StartPointer+sizeof(widest_int_t)>EndPointer)
				fillBuffer();

			switch (getTexelOrBlockBytesize(f))
			{
				case 1:
					if (StartPointer+sizeof(int8_t)>EndPointer)
						break;
					return *(StartPointer++);
				case 2:
				{
					if (StartPointer+sizeof(int16_t)>EndPointer)
						break;
					auto retval = *(reinterpret_cast<int16_t*&>(StartPointer)++);
					if (IsWrongEndian)
						retval = byteswap(retval);
					return retval;
				}
				case 4:
				{
					if (StartPointer+sizeof(int32_t)>EndPointer)
						break;
					auto retval = *(reinterpret_cast<int32_t*&>(StartPointer)++);
					if (IsWrongEndian)
						retval = byteswap(retval);
					return retval;
				}
				default:
					assert(false);
					break;
			}
			return 0;
		}
		const char* word = getNextWord();
		if (!word)
			return 0u;
		const size_t tokenLen = WordLength >= 0 ? static_cast<size_t>(WordLength + 1) : std::strlen(word);
		const char* const wordEnd = word + tokenLen;
		if (word == wordEnd)
			return 0u;

		if (isSignedFormat(f))
		{
			int64_t value = 0;
			const auto parseResult = std::from_chars(word, wordEnd, value, 10);
			if (parseResult.ec == std::errc() && parseResult.ptr == wordEnd)
				return static_cast<widest_int_t>(value);

			char* fallbackEnd = nullptr;
			const auto fallback = std::strtoll(word, &fallbackEnd, 10);
			if (fallbackEnd && fallbackEnd != word)
				return static_cast<widest_int_t>(fallback);
			return 0u;
		}
		else
		{
			uint64_t value = 0u;
			const auto parseResult = std::from_chars(word, wordEnd, value, 10);
			if (parseResult.ec == std::errc() && parseResult.ptr == wordEnd)
				return static_cast<widest_int_t>(value);

			char* fallbackEnd = nullptr;
			const auto fallback = std::strtoull(word, &fallbackEnd, 10);
			if (fallbackEnd && fallbackEnd != word)
				return static_cast<widest_int_t>(fallback);
			return 0u;
		}
	}
	// read the next float from the file and move the start pointer along
	hlsl::float64_t getFloat(const E_FORMAT f)
	{
		assert(isFloatingPointFormat(f));
		if (IsBinaryFile)
		{
			if (StartPointer+sizeof(hlsl::float64_t)>EndPointer)
				fillBuffer();

			switch (getTexelOrBlockBytesize(f))
			{
				case 4:
				{
					if (StartPointer+sizeof(hlsl::float32_t)>EndPointer)
						break;
					auto retval = *(reinterpret_cast<hlsl::float32_t*&>(StartPointer)++);
					if (IsWrongEndian)
						retval = byteswap(retval);
					return retval;
				}
				case 8:
				{
					if (StartPointer+sizeof(hlsl::float64_t)>EndPointer)
						break;
					auto retval = *(reinterpret_cast<hlsl::float64_t*&>(StartPointer)++);
					if (IsWrongEndian)
						retval = byteswap(retval);
					return retval;
				}
				default:
					assert(false);
					break;
			}
			return 0;
		}
		const char* word = getNextWord();
		if (!word)
			return 0.0;
		const size_t tokenLen = WordLength >= 0 ? static_cast<size_t>(WordLength + 1) : std::strlen(word);
		const char* const wordEnd = word + tokenLen;
		if (word == wordEnd)
			return 0.0;

		hlsl::float64_t value = 0.0;
		const auto parseResult = fast_float::from_chars(word, wordEnd, value);
		if (parseResult.ec == std::errc() && parseResult.ptr == wordEnd)
			return value;

		char* fallbackEnd = nullptr;
		const auto fallback = std::strtod(word, &fallbackEnd);
		if (fallbackEnd && fallbackEnd != word)
			return fallback;
		return 0.0;
	}
	// read the next thing from the file and move the start pointer along
	void getData(void* dst, const E_FORMAT f)
	{
		const auto size = getTexelOrBlockBytesize(f);
		if (StartPointer+size>EndPointer)
		{
			fillBuffer();
			if (StartPointer+size>EndPointer)
				return;
		}
		if (IsWrongEndian)
			std::reverse_copy(StartPointer,StartPointer+size,reinterpret_cast<char*>(dst));
		else
			memcpy(dst,StartPointer,size);
		StartPointer += size;
	}
	struct SVertAttrIt
	{
		uint8_t* ptr;
		uint32_t stride;
		E_FORMAT dstFmt;
	};
	enum class EFastVertexReadResult : uint8_t
	{
		NotApplicable,
		Success,
		Error
	};
	EFastVertexReadResult readVertexElementFast(const SElement& el)
	{
		if (!IsBinaryFile || IsWrongEndian || el.Name != "vertex")
			return EFastVertexReadResult::NotApplicable;

		enum class ELayoutKind : uint8_t
		{
			XYZ,
			XYZ_N,
			XYZ_N_UV
		};

		auto allF32 = [&el]()->bool
		{
			for (const auto& prop : el.Properties)
			{
				if (prop.type != EF_R32_SFLOAT)
					return false;
			}
			return true;
		};
		if (!allF32())
			return EFastVertexReadResult::NotApplicable;

		auto matchNames = [&el](std::initializer_list<const char*> names)->bool
		{
			if (el.Properties.size() != names.size())
				return false;
			size_t i = 0ull;
			for (const auto* name : names)
			{
				if (el.Properties[i].Name != name)
					return false;
				++i;
			}
			return true;
		};

		ELayoutKind layout = ELayoutKind::XYZ;
		if (matchNames({ "x", "y", "z" }))
		{
			layout = ELayoutKind::XYZ;
		}
		else if (matchNames({ "x", "y", "z", "nx", "ny", "nz" }))
		{
			layout = ELayoutKind::XYZ_N;
		}
		else if (matchNames({ "x", "y", "z", "nx", "ny", "nz", "u", "v" }) || matchNames({ "x", "y", "z", "nx", "ny", "nz", "s", "t" }))
		{
			layout = ELayoutKind::XYZ_N_UV;
		}
		else
		{
			return EFastVertexReadResult::NotApplicable;
		}

		const size_t floatBytes = sizeof(hlsl::float32_t);
		auto validateTuple = [&](const size_t beginIx, const size_t componentCount, uint32_t& outStride, uint8_t*& outBase)->bool
		{
			if (beginIx + componentCount > vertAttrIts.size())
				return false;
			auto& first = vertAttrIts[beginIx];
			if (!first.ptr || first.dstFmt != EF_R32_SFLOAT)
				return false;
			outStride = first.stride;
			outBase = first.ptr;
			for (size_t c = 1ull; c < componentCount; ++c)
			{
				auto& it = vertAttrIts[beginIx + c];
				if (!it.ptr || it.dstFmt != EF_R32_SFLOAT)
					return false;
				if (it.stride != outStride)
					return false;
				if (it.ptr != outBase + c * floatBytes)
					return false;
			}
			return true;
		};

		uint32_t posStride = 0u;
		uint32_t normalStride = 0u;
		uint32_t uvStride = 0u;
		uint8_t* posBase = nullptr;
		uint8_t* normalBase = nullptr;
		uint8_t* uvBase = nullptr;
		switch (layout)
		{
			case ELayoutKind::XYZ:
				if (vertAttrIts.size() != 3u || !validateTuple(0u, 3u, posStride, posBase))
					return EFastVertexReadResult::NotApplicable;
				break;
			case ELayoutKind::XYZ_N:
				if (vertAttrIts.size() != 6u)
					return EFastVertexReadResult::NotApplicable;
				if (!validateTuple(0u, 3u, posStride, posBase) || !validateTuple(3u, 3u, normalStride, normalBase))
					return EFastVertexReadResult::NotApplicable;
				break;
			case ELayoutKind::XYZ_N_UV:
				if (vertAttrIts.size() != 8u)
					return EFastVertexReadResult::NotApplicable;
				if (!validateTuple(0u, 3u, posStride, posBase) || !validateTuple(3u, 3u, normalStride, normalBase) || !validateTuple(6u, 2u, uvStride, uvBase))
					return EFastVertexReadResult::NotApplicable;
				break;
		}

		const size_t srcBytesPerVertex = [layout]()->size_t
		{
			switch (layout)
			{
				case ELayoutKind::XYZ:
					return sizeof(hlsl::float32_t) * 3ull;
				case ELayoutKind::XYZ_N:
					return sizeof(hlsl::float32_t) * 6ull;
				case ELayoutKind::XYZ_N_UV:
					return sizeof(hlsl::float32_t) * 8ull;
				default:
					return 0ull;
			}
		}();
		if (srcBytesPerVertex == 0ull || el.Count > (std::numeric_limits<size_t>::max() / srcBytesPerVertex))
			return EFastVertexReadResult::Error;

		size_t remainingVertices = el.Count;
		while (remainingVertices > 0ull)
		{
			if (StartPointer + srcBytesPerVertex > EndPointer)
				fillBuffer();
			const size_t available = EndPointer > StartPointer ? static_cast<size_t>(EndPointer - StartPointer) : 0ull;
			if (available < srcBytesPerVertex)
				return EFastVertexReadResult::Error;

			const size_t batchVertices = std::min(remainingVertices, available / srcBytesPerVertex);
			const uint8_t* src = reinterpret_cast<const uint8_t*>(StartPointer);
			switch (layout)
			{
				case ELayoutKind::XYZ:
				{
					if (posStride == 3ull * floatBytes)
					{
						const size_t batchBytes = batchVertices * 3ull * floatBytes;
						std::memcpy(posBase, src, batchBytes);
						src += batchBytes;
						posBase += batchBytes;
					}
					else
					{
						for (size_t v = 0ull; v < batchVertices; ++v)
						{
							std::memcpy(posBase, src, 3ull * floatBytes);
							src += 3ull * floatBytes;
							posBase += posStride;
						}
					}
				}
				break;
				case ELayoutKind::XYZ_N:
				{
					for (size_t v = 0ull; v < batchVertices; ++v)
					{
						std::memcpy(posBase, src, 3ull * floatBytes);
						src += 3ull * floatBytes;
						posBase += posStride;
						std::memcpy(normalBase, src, 3ull * floatBytes);
						src += 3ull * floatBytes;
						normalBase += normalStride;
					}
				}
				break;
				case ELayoutKind::XYZ_N_UV:
				{
					for (size_t v = 0ull; v < batchVertices; ++v)
					{
						std::memcpy(posBase, src, 3ull * floatBytes);
						src += 3ull * floatBytes;
						posBase += posStride;
						std::memcpy(normalBase, src, 3ull * floatBytes);
						src += 3ull * floatBytes;
						normalBase += normalStride;
						std::memcpy(uvBase, src, 2ull * floatBytes);
						src += 2ull * floatBytes;
						uvBase += uvStride;
					}
				}
				break;
			}

			const size_t consumed = batchVertices * srcBytesPerVertex;
			StartPointer += consumed;
			remainingVertices -= batchVertices;
		}

		const size_t posAdvance = el.Count * posStride;
		vertAttrIts[0].ptr += posAdvance;
		vertAttrIts[1].ptr += posAdvance;
		vertAttrIts[2].ptr += posAdvance;
		if (layout == ELayoutKind::XYZ_N || layout == ELayoutKind::XYZ_N_UV)
		{
			const size_t normalAdvance = el.Count * normalStride;
			vertAttrIts[3].ptr += normalAdvance;
			vertAttrIts[4].ptr += normalAdvance;
			vertAttrIts[5].ptr += normalAdvance;
		}
		if (layout == ELayoutKind::XYZ_N_UV)
		{
			const size_t uvAdvance = el.Count * uvStride;
			vertAttrIts[6].ptr += uvAdvance;
			vertAttrIts[7].ptr += uvAdvance;
		}
		return EFastVertexReadResult::Success;
	}
	void readVertex(const IAssetLoader::SAssetLoadParams& _params, const SElement& el)
	{
		assert(el.Name=="vertex");
		assert(el.Properties.size()==vertAttrIts.size());
		if (!IsBinaryFile)
			getNextLine();

		for (size_t j=0; j<el.Count; ++j)
		for (auto i=0u; i<vertAttrIts.size(); i++)
		{
			const auto& prop = el.Properties[i];
			auto& it = vertAttrIts[i];
			if (!it.ptr)
			{
				prop.skip(*this);
				continue;
			}
			if (!IsBinaryFile)
			{
				if (isIntegerFormat(prop.type))
				{
					uint64_t tmp = getInt(prop.type);
					encodePixels(it.dstFmt,it.ptr,&tmp);
				}
				else
				{
					hlsl::float64_t tmp = getFloat(prop.type);
					encodePixels(it.dstFmt,it.ptr,&tmp);
				}
			}
			else if (it.dstFmt!=prop.type)
			{
				assert(isIntegerFormat(it.dstFmt)==isIntegerFormat(prop.type));
				if (isIntegerFormat(it.dstFmt))
				{
					uint64_t tmp = getInt(prop.type);
					encodePixels(it.dstFmt,it.ptr,&tmp);
				}
				else
				{
					hlsl::float64_t tmp = getFloat(prop.type);
					encodePixels(it.dstFmt,it.ptr,&tmp);
				}
			}
			else
				getData(it.ptr,prop.type);
			//
			it.ptr += it.stride;
		}
	}
	bool readFace(const SElement& Element, core::vector<uint32_t>& _outIndices, uint32_t& _maxIndex, const uint32_t vertexCount)
	{
		if (!IsBinaryFile)
			getNextLine();
		const bool hasVertexCount = vertexCount != 0u;

		for (const auto& prop : Element.Properties)
		{
			if (prop.isList() && (prop.Name=="vertex_indices" || prop.Name == "vertex_index"))
			{
				const uint32_t count = getInt(prop.list.countType);
				const auto srcIndexFmt = prop.list.itemType;
				if (count < 3u)
				{
					for (uint32_t j = 0u; j < count; ++j)
						getInt(srcIndexFmt);
					continue;
				}
				if (count > 3u)
					_outIndices.reserve(_outIndices.size() + static_cast<size_t>(count - 2u) * 3ull);
				auto emitFan = [&_outIndices, &_maxIndex, hasVertexCount, vertexCount](auto&& readIndex, const uint32_t faceVertexCount)->bool
				{
					uint32_t i0 = readIndex();
					uint32_t i1 = readIndex();
					uint32_t i2 = readIndex();
					if (hasVertexCount)
					{
						if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
							return false;
					}
					else
					{
						_maxIndex = std::max(_maxIndex, std::max(i0, std::max(i1, i2)));
					}
					_outIndices.push_back(i0);
					_outIndices.push_back(i1);
					_outIndices.push_back(i2);
					uint32_t prev = i2;
					for (uint32_t j = 3u; j < faceVertexCount; ++j)
					{
						const uint32_t idx = readIndex();
						if (hasVertexCount)
						{
							if (idx >= vertexCount)
								return false;
						}
						else
						{
							_maxIndex = std::max(_maxIndex, idx);
						}
						_outIndices.push_back(i0);
						_outIndices.push_back(prev);
						_outIndices.push_back(idx);
						prev = idx;
					}
					return true;
				};

				if (IsBinaryFile && !IsWrongEndian && srcIndexFmt == EF_R32_UINT)
				{
					const size_t bytesNeeded = static_cast<size_t>(count) * sizeof(uint32_t);
					if (StartPointer + bytesNeeded > EndPointer)
						fillBuffer();
					if (StartPointer + bytesNeeded <= EndPointer)
					{
						const uint8_t* ptr = reinterpret_cast<const uint8_t*>(StartPointer);
						auto readIndex = [&ptr]() -> uint32_t
						{
							uint32_t v = 0u;
							std::memcpy(&v, ptr, sizeof(v));
							ptr += sizeof(v);
							return v;
						};
						if (!emitFan(readIndex, count))
							return false;
						StartPointer = reinterpret_cast<char*>(const_cast<uint8_t*>(ptr));
						continue;
					}
				}
				else if (IsBinaryFile && !IsWrongEndian && srcIndexFmt == EF_R16_UINT)
				{
					const size_t bytesNeeded = static_cast<size_t>(count) * sizeof(uint16_t);
					if (StartPointer + bytesNeeded > EndPointer)
						fillBuffer();
					if (StartPointer + bytesNeeded <= EndPointer)
					{
						const uint8_t* ptr = reinterpret_cast<const uint8_t*>(StartPointer);
						auto readIndex = [&ptr]() -> uint32_t
						{
							uint16_t v = 0u;
							std::memcpy(&v, ptr, sizeof(v));
							ptr += sizeof(v);
							return static_cast<uint32_t>(v);
						};
						if (!emitFan(readIndex, count))
							return false;
						StartPointer = reinterpret_cast<char*>(const_cast<uint8_t*>(ptr));
						continue;
					}
				}

				auto readIndex = [&]() -> uint32_t
				{
					return static_cast<uint32_t>(getInt(srcIndexFmt));
				};
				if (!emitFan(readIndex, count))
					return false;
			}
			else if (prop.Name == "intensity")
			{
				// todo: face intensity
				prop.skip(*this);
			}
			else
				prop.skip(*this);
		}
		return true;
	}

	enum class EFastFaceReadResult : uint8_t
	{
		NotApplicable,
		Success,
		Error
	};

	EFastFaceReadResult readFaceElementFast(const SElement& element, core::vector<uint32_t>& _outIndices, uint32_t& _maxIndex, uint64_t& _faceCount, const uint32_t vertexCount)
	{
		if (!IsBinaryFile || IsWrongEndian)
			return EFastFaceReadResult::NotApplicable;
		if (element.Properties.size() != 1u)
			return EFastFaceReadResult::NotApplicable;

		const auto& prop = element.Properties[0];
		if (!prop.isList() || (prop.Name != "vertex_indices" && prop.Name != "vertex_index"))
			return EFastFaceReadResult::NotApplicable;
		if (prop.list.countType != EF_R8_UINT)
			return EFastFaceReadResult::NotApplicable;

		const E_FORMAT srcIndexFmt = prop.list.itemType;
		const bool isSrcU32 = srcIndexFmt == EF_R32_UINT;
		const bool isSrcS32 = srcIndexFmt == EF_R32_SINT;
		const bool isSrcU16 = srcIndexFmt == EF_R16_UINT;
		const bool isSrcS16 = srcIndexFmt == EF_R16_SINT;
		if (!isSrcU32 && !isSrcS32 && !isSrcU16 && !isSrcS16)
			return EFastFaceReadResult::NotApplicable;

		const bool is32Bit = isSrcU32 || isSrcS32;
		const size_t indexSize = is32Bit ? sizeof(uint32_t) : sizeof(uint16_t);
		const bool hasVertexCount = vertexCount != 0u;
		const bool trackMaxIndex = !hasVertexCount;
		const size_t minTriangleRecordSize = sizeof(uint8_t) + indexSize * 3u;
		if (element.Count > (std::numeric_limits<size_t>::max() / minTriangleRecordSize))
			return EFastFaceReadResult::Error;
		const size_t minBytesNeeded = element.Count * minTriangleRecordSize;
		if (StartPointer + minBytesNeeded <= EndPointer)
		{
			if (element.Count > (std::numeric_limits<size_t>::max() / 3u))
				return EFastFaceReadResult::Error;
			const size_t triIndices = element.Count * 3u;
			if (_outIndices.size() > (std::numeric_limits<size_t>::max() - triIndices))
				return EFastFaceReadResult::Error;
			const size_t oldSize = _outIndices.size();
			const uint32_t oldMaxIndex = _maxIndex;
			_outIndices.resize(oldSize + triIndices);
			uint32_t* out = _outIndices.data() + oldSize;
			const uint8_t* ptr = reinterpret_cast<const uint8_t*>(StartPointer);
			bool fallbackToGeneric = false;
			if (is32Bit)
			{
				const size_t hw = std::thread::hardware_concurrency();
				const size_t maxWorkersByWork = std::max<size_t>(1ull, minBytesNeeded / (640ull << 10));
				size_t workerCount = hw ? std::min(hw, maxWorkersByWork) : 1ull;
				if (workerCount > 1ull)
				{
					const size_t recordBytes = sizeof(uint8_t) + 3ull * sizeof(uint32_t);
					const bool needMax = trackMaxIndex;
					const bool validateAgainstVertexCount = hasVertexCount;
					std::vector<uint8_t> workerNonTriangle(workerCount, 0u);
					std::vector<uint8_t> workerInvalid(workerCount, 0u);
					std::vector<uint32_t> workerMax(needMax ? workerCount : 0ull, 0u);
					auto parseChunk = [&](const size_t workerIx, const size_t beginFace, const size_t endFace) -> void
					{
						const uint8_t* in = ptr + beginFace * recordBytes;
						uint32_t* outLocal = out + beginFace * 3ull;
						uint32_t localMax = 0u;
						uint32_t localSignBits = 0u;
						for (size_t faceIx = beginFace; faceIx < endFace; ++faceIx)
						{
							if (*in != 3u)
							{
								workerNonTriangle[workerIx] = 1u;
								break;
							}
							++in;
							std::memcpy(outLocal, in, 3ull * sizeof(uint32_t));
							const uint32_t i0 = outLocal[0];
							const uint32_t i1 = outLocal[1];
							const uint32_t i2 = outLocal[2];
							if (isSrcS32)
								localSignBits |= (i0 | i1 | i2);
							if (i0 > localMax) localMax = i0;
							if (i1 > localMax) localMax = i1;
							if (i2 > localMax) localMax = i2;
							in += 3ull * sizeof(uint32_t);
							outLocal += 3ull;
						}
						if (isSrcS32 && (localSignBits & 0x80000000u))
							workerInvalid[workerIx] = 1u;
						if (validateAgainstVertexCount && localMax >= vertexCount)
							workerInvalid[workerIx] = 1u;
						if (needMax)
							workerMax[workerIx] = localMax;
					};
					plyRunParallelWorkers(workerCount, [&](const size_t workerIx)
					{
						const size_t begin = (element.Count * workerIx) / workerCount;
						const size_t end = (element.Count * (workerIx + 1ull)) / workerCount;
						parseChunk(workerIx, begin, end);
					});

					const bool anyNonTriangle = std::any_of(workerNonTriangle.begin(), workerNonTriangle.end(), [](const uint8_t v) { return v != 0u; });
					if (anyNonTriangle)
					{
						_outIndices.resize(oldSize);
						_maxIndex = oldMaxIndex;
						return EFastFaceReadResult::NotApplicable;
					}
					const bool anyInvalid = std::any_of(workerInvalid.begin(), workerInvalid.end(), [](const uint8_t v) { return v != 0u; });
					if (anyInvalid)
					{
						_outIndices.resize(oldSize);
						_maxIndex = oldMaxIndex;
						return EFastFaceReadResult::Error;
					}
					if (trackMaxIndex)
					{
						for (const uint32_t local : workerMax)
							if (local > _maxIndex)
								_maxIndex = local;
					}

					StartPointer = reinterpret_cast<char*>(const_cast<uint8_t*>(ptr + element.Count * recordBytes));
					_faceCount += element.Count;
					return EFastFaceReadResult::Success;
				}
			}

			if (is32Bit)
			{
				if (isSrcU32)
				{
					if (trackMaxIndex)
					{
						for (size_t j = 0u; j < element.Count; ++j)
						{
							const uint8_t c = *ptr++;
							if (c != 3u)
							{
								fallbackToGeneric = true;
								break;
							}
							std::memcpy(out, ptr, 3ull * sizeof(uint32_t));
							ptr += 3ull * sizeof(uint32_t);
							if (out[0] > _maxIndex) _maxIndex = out[0];
							if (out[1] > _maxIndex) _maxIndex = out[1];
							if (out[2] > _maxIndex) _maxIndex = out[2];
							out += 3;
						}
					}
					else
					{
						uint32_t localMax = 0u;
						for (size_t j = 0u; j < element.Count; ++j)
						{
							const uint8_t c = *ptr++;
							if (c != 3u)
							{
								fallbackToGeneric = true;
								break;
							}
							std::memcpy(out, ptr, 3ull * sizeof(uint32_t));
							ptr += 3ull * sizeof(uint32_t);
							if (out[0] > localMax) localMax = out[0];
							if (out[1] > localMax) localMax = out[1];
							if (out[2] > localMax) localMax = out[2];
							out += 3;
						}
						if (!fallbackToGeneric && localMax >= vertexCount)
							return EFastFaceReadResult::Error;
					}
				}
				else if (trackMaxIndex)
				{
					for (size_t j = 0u; j < element.Count; ++j)
					{
						const uint8_t c = *ptr++;
						if (c != 3u)
						{
							fallbackToGeneric = true;
							break;
						}
						std::memcpy(out, ptr, 3ull * sizeof(uint32_t));
						ptr += 3ull * sizeof(uint32_t);
						if ((out[0] | out[1] | out[2]) & 0x80000000u)
							return EFastFaceReadResult::Error;
						if (out[0] > _maxIndex) _maxIndex = out[0];
						if (out[1] > _maxIndex) _maxIndex = out[1];
						if (out[2] > _maxIndex) _maxIndex = out[2];
						out += 3;
					}
				}
				else
				{
					uint32_t localMax = 0u;
					uint32_t localSignBits = 0u;
					for (size_t j = 0u; j < element.Count; ++j)
					{
						const uint8_t c = *ptr++;
						if (c != 3u)
						{
							fallbackToGeneric = true;
							break;
						}
						std::memcpy(out, ptr, 3ull * sizeof(uint32_t));
						ptr += 3ull * sizeof(uint32_t);
						localSignBits |= (out[0] | out[1] | out[2]);
						if (out[0] > localMax) localMax = out[0];
						if (out[1] > localMax) localMax = out[1];
						if (out[2] > localMax) localMax = out[2];
						out += 3;
					}
					if (!fallbackToGeneric)
					{
						if (localSignBits & 0x80000000u)
							return EFastFaceReadResult::Error;
						if (localMax >= vertexCount)
							return EFastFaceReadResult::Error;
					}
				}
			}
			else
			{
				if (isSrcU16)
				{
					if (trackMaxIndex)
					{
						for (size_t j = 0u; j < element.Count; ++j)
						{
							const uint8_t c = *ptr++;
							if (c != 3u)
							{
								fallbackToGeneric = true;
								break;
							}
							uint16_t tri[3] = {};
							std::memcpy(tri, ptr, sizeof(tri));
							ptr += sizeof(tri);
							out[0] = tri[0];
							out[1] = tri[1];
							out[2] = tri[2];
							if (out[0] > _maxIndex) _maxIndex = out[0];
							if (out[1] > _maxIndex) _maxIndex = out[1];
							if (out[2] > _maxIndex) _maxIndex = out[2];
							out += 3;
						}
					}
					else
					{
						for (size_t j = 0u; j < element.Count; ++j)
						{
							const uint8_t c = *ptr++;
							if (c != 3u)
							{
								fallbackToGeneric = true;
								break;
							}
							uint16_t tri[3] = {};
							std::memcpy(tri, ptr, sizeof(tri));
							ptr += sizeof(tri);
							out[0] = tri[0];
							out[1] = tri[1];
							out[2] = tri[2];
							if (out[0] >= vertexCount || out[1] >= vertexCount || out[2] >= vertexCount)
								return EFastFaceReadResult::Error;
							out += 3;
						}
					}
				}
				else if (trackMaxIndex)
				{
					for (size_t j = 0u; j < element.Count; ++j)
					{
						const uint8_t c = *ptr++;
						if (c != 3u)
						{
							fallbackToGeneric = true;
							break;
						}
						int16_t tri[3] = {};
						std::memcpy(tri, ptr, sizeof(tri));
						ptr += sizeof(tri);
						if ((static_cast<uint16_t>(tri[0]) | static_cast<uint16_t>(tri[1]) | static_cast<uint16_t>(tri[2])) & 0x8000u)
							return EFastFaceReadResult::Error;
						out[0] = static_cast<uint32_t>(tri[0]);
						out[1] = static_cast<uint32_t>(tri[1]);
						out[2] = static_cast<uint32_t>(tri[2]);
						if (out[0] > _maxIndex) _maxIndex = out[0];
						if (out[1] > _maxIndex) _maxIndex = out[1];
						if (out[2] > _maxIndex) _maxIndex = out[2];
						out += 3;
					}
				}
				else
				{
					for (size_t j = 0u; j < element.Count; ++j)
					{
						const uint8_t c = *ptr++;
						if (c != 3u)
						{
							fallbackToGeneric = true;
							break;
						}
						int16_t tri[3] = {};
						std::memcpy(tri, ptr, sizeof(tri));
						ptr += sizeof(tri);
						if ((static_cast<uint16_t>(tri[0]) | static_cast<uint16_t>(tri[1]) | static_cast<uint16_t>(tri[2])) & 0x8000u)
							return EFastFaceReadResult::Error;
						out[0] = static_cast<uint32_t>(tri[0]);
						out[1] = static_cast<uint32_t>(tri[1]);
						out[2] = static_cast<uint32_t>(tri[2]);
						if (out[0] >= vertexCount || out[1] >= vertexCount || out[2] >= vertexCount)
							return EFastFaceReadResult::Error;
						out += 3;
					}
				}
			}

			if (!fallbackToGeneric)
			{
				StartPointer = reinterpret_cast<char*>(const_cast<uint8_t*>(ptr));
				_faceCount += element.Count;
				return EFastFaceReadResult::Success;
			}

			_outIndices.resize(oldSize);
			_maxIndex = oldMaxIndex;
		}

		if (element.Count > (std::numeric_limits<size_t>::max() / 3u))
			return EFastFaceReadResult::Error;
		const size_t reserveCount = element.Count * 3u;
		if (_outIndices.size() > (std::numeric_limits<size_t>::max() - reserveCount))
			return EFastFaceReadResult::Error;
		_outIndices.reserve(_outIndices.size() + reserveCount);
		auto ensureBytes = [this](const size_t bytes)->bool
		{
			if (StartPointer + bytes > EndPointer)
				fillBuffer();
			return StartPointer + bytes <= EndPointer;
		};
		auto readCount = [&ensureBytes, this](int32_t& outCount)->bool
		{
			if (!ensureBytes(sizeof(uint8_t)))
				return false;
			outCount = static_cast<uint8_t>(*StartPointer++);
			return true;
		};
		auto readIndex = [&ensureBytes, this, is32Bit, isSrcU32, isSrcU16](uint32_t& out)->bool
		{
			if (is32Bit)
			{
				if (!ensureBytes(sizeof(uint32_t)))
					return false;
				if (isSrcU32)
				{
					std::memcpy(&out, StartPointer, sizeof(uint32_t));
				}
				else
				{
					int32_t v = 0;
					std::memcpy(&v, StartPointer, sizeof(v));
					if (v < 0)
						return false;
					out = static_cast<uint32_t>(v);
				}
				StartPointer += sizeof(uint32_t);
				return true;
			}

			if (!ensureBytes(sizeof(uint16_t)))
				return false;
			if (isSrcU16)
			{
				uint16_t v = 0u;
				std::memcpy(&v, StartPointer, sizeof(uint16_t));
				out = v;
			}
			else
			{
				int16_t v = 0;
				std::memcpy(&v, StartPointer, sizeof(int16_t));
				if (v < 0)
					return false;
				out = static_cast<uint32_t>(v);
			}
			StartPointer += sizeof(uint16_t);
			return true;
		};

		for (size_t j = 0u; j < element.Count; ++j)
		{
			int32_t countSigned = 0;
			if (!readCount(countSigned))
				return EFastFaceReadResult::Error;
			const uint32_t count = static_cast<uint32_t>(countSigned);
			if (count < 3u)
			{
				uint32_t dummy = 0u;
				for (uint32_t k = 0u; k < count; ++k)
				{
					if (!readIndex(dummy))
						return EFastFaceReadResult::Error;
				}
				++_faceCount;
				continue;
			}

			uint32_t i0 = 0u;
			uint32_t i1 = 0u;
			uint32_t i2 = 0u;
			if (!readIndex(i0) || !readIndex(i1) || !readIndex(i2))
				return EFastFaceReadResult::Error;

			if (trackMaxIndex)
			{
				_maxIndex = std::max(_maxIndex, std::max(i0, std::max(i1, i2)));
			}
			else if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
			{
				return EFastFaceReadResult::Error;
			}
			_outIndices.push_back(i0);
			_outIndices.push_back(i1);
			_outIndices.push_back(i2);

			uint32_t prev = i2;
			for (uint32_t k = 3u; k < count; ++k)
			{
				uint32_t idx = 0u;
				if (!readIndex(idx))
					return EFastFaceReadResult::Error;
				if (trackMaxIndex)
				{
					_maxIndex = std::max(_maxIndex, idx);
				}
				else if (idx >= vertexCount)
				{
					return EFastFaceReadResult::Error;
				}
				_outIndices.push_back(i0);
				_outIndices.push_back(prev);
				_outIndices.push_back(idx);
				prev = idx;
			}

			++_faceCount;
		}

		return EFastFaceReadResult::Success;
	}

	IAssetLoader::SAssetLoadContext inner;
	uint32_t topHierarchyLevel;
	IAssetLoader::IAssetLoaderOverride* loaderOverride;
	// input buffer must be at least twice as long as the longest line in the file
	core::vector<char> Buffer;
	size_t ioReadWindowSize = DefaultIoReadWindowBytes;
	core::vector<SElement> ElementList = {};
	char* StartPointer = nullptr, *EndPointer = nullptr, *LineEndPointer = nullptr;
	int32_t LineLength = 0;
	int32_t WordLength = -1; // this variable is a misnomer, its really the offset to next word minus one
	bool IsBinaryFile = false, IsWrongEndian = false, EndOfFile = false;
	bool UsingMappedBinaryWindow = false;
	size_t fileOffset = {};
	uint64_t readCallCount = 0ull;
	uint64_t readBytesTotal = 0ull;
	uint64_t readMinBytes = std::numeric_limits<uint64_t>::max();
	//
	core::vector<SVertAttrIt> vertAttrIts;
};

//! creates/loads an animated mesh from the file.
SAssetBundle CPLYMeshFileLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	using namespace nbl::core;
	if (!_file)
		return {};

	using clock_t = std::chrono::high_resolution_clock;
	const auto totalStart = clock_t::now();
	double headerMs = 0.0;
	double vertexMs = 0.0;
	double vertexFastMs = 0.0;
	double vertexGenericMs = 0.0;
	double faceMs = 0.0;
	double skipMs = 0.0;
	double layoutNegotiateMs = 0.0;
	double viewCreateMs = 0.0;
	double hashRangeMs = 0.0;
	double indexBuildMs = 0.0;
	double aabbMs = 0.0;
	uint64_t faceCount = 0u;
	uint64_t fastFaceElementCount = 0u;
	uint64_t fastVertexElementCount = 0u;
	uint32_t maxIndexRead = 0u;
	const uint64_t fileSize = _file->getSize();
	const auto ioPlan = resolveFileIOPolicy(_params.ioPolicy, fileSize, true);
	if (!ioPlan.valid)
	{
		_params.logger.log("PLY loader: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str(), ioPlan.reason);
		return {};
	}

	SContext ctx = {
		asset::IAssetLoader::SAssetLoadContext{
			_params,
			_file
		},
		_hierarchyLevel,
		_override
	};
	uint64_t desiredReadWindow = ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile ? (fileSize + SContext::ReadWindowPaddingBytes) : ioPlan.chunkSizeBytes;
	if (ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile)
	{
		const bool mappedInput = static_cast<const system::IFile*>(_file)->getMappedPointer() != nullptr;
		if (mappedInput && fileSize > (SContext::DefaultIoReadWindowBytes * 2ull))
			desiredReadWindow = SContext::DefaultIoReadWindowBytes;
	}
	const uint64_t safeReadWindow = std::min<uint64_t>(desiredReadWindow, static_cast<uint64_t>(std::numeric_limits<size_t>::max() - SContext::ReadWindowPaddingBytes));
	ctx.init(static_cast<size_t>(safeReadWindow));

	// start with empty mesh
    auto geometry = make_smart_refctd_ptr<ICPUPolygonGeometry>();
	uint32_t vertCount=0;

	// Currently only supports ASCII or binary meshes
	if (strcmp(ctx.getNextLine(),"ply"))
	{
		_params.logger.log("Not a valid PLY file %s", system::ILogger::ELL_ERROR,ctx.inner.mainFile->getFileName().string().c_str());
		return {};
	}

	// cut the next line out
	ctx.getNextLine();
	// grab the word from this line
	const char* word = ctx.getNextWord();
	// ignore comments
	for (; strcmp(word,"comment")==0; ctx.getNextLine())
		word = ctx.getNextWord();

	bool readingHeader = true;
	bool continueReading = true;
	ctx.IsBinaryFile = false;
	ctx.IsWrongEndian= false;
	const auto headerStart = clock_t::now();

	do
	{
		if (strcmp(word,"property") == 0)
		{
			word = ctx.getNextWord();

			if (ctx.ElementList.empty())
			{
				_params.logger.log("PLY property token found before element %s", system::ILogger::ELL_WARNING, word);
			}
			else
			{
				// get element
				auto& el = ctx.ElementList.back();
				
				// fill property struct
				auto& prop = el.Properties.emplace_back();
				prop.type = prop.getType(word);
				if (prop.type==EF_UNKNOWN)
				{
					el.KnownSize = false;

					word = ctx.getNextWord();

					prop.list.countType = prop.getType(word);
					if (ctx.IsBinaryFile && !isIntegerFormat(prop.list.countType))
					{
						_params.logger.log("Cannot read binary PLY file containing data types of unknown or non integer length %s", system::ILogger::ELL_WARNING, word);
						continueReading = false;
					}
					else
					{
						word = ctx.getNextWord();
						prop.list.itemType = prop.getType(word);
						if (ctx.IsBinaryFile && !isIntegerFormat(prop.list.itemType))
						{
							_params.logger.log("Cannot read binary PLY file containing data types of unknown or non integer length %s", system::ILogger::ELL_ERROR, word);
							continueReading = false;
						}
					}
				}
				else if (ctx.IsBinaryFile && prop.type==EF_UNKNOWN)
				{
					_params.logger.log("Cannot read binary PLY file containing data types of unknown length %s", system::ILogger::ELL_ERROR, word);
					continueReading = false;
				}
				else
					el.KnownSize += getTexelOrBlockBytesize(prop.type);

				prop.Name = ctx.getNextWord();
			}
		}
		else if (strcmp(word,"element")==0)
		{
			auto& el = ctx.ElementList.emplace_back();
			el.Name = ctx.getNextWord();
			const char* const countWord = ctx.getNextWord();
			uint64_t parsedCount = 0ull;
			if (countWord)
			{
				const char* const countWordEnd = countWord + std::strlen(countWord);
				const auto parseResult = std::from_chars(countWord, countWordEnd, parsedCount, 10);
				if (!(parseResult.ec == std::errc() && parseResult.ptr == countWordEnd))
				{
					char* fallbackEnd = nullptr;
					parsedCount = std::strtoull(countWord, &fallbackEnd, 10);
				}
			}
			el.Count = static_cast<size_t>(parsedCount);
			el.KnownSize = 0;
			if (el.Name=="vertex")
				vertCount = el.Count;
		}
		else if (strcmp(word,"comment")==0)
		{
			// ignore line
		}
		// must be `format {binary_little_endian|binary_big_endian|ascii} 1.0`
		else if (strcmp(word,"format") == 0)
		{
			word = ctx.getNextWord();

			if (strcmp(word, "binary_little_endian") == 0)
			{
				ctx.IsBinaryFile = true;
			}
			else if (strcmp(word, "binary_big_endian") == 0)
			{
				ctx.IsBinaryFile = true;
				ctx.IsWrongEndian = true;
			}
			else if (strcmp(word, "ascii")==0)
			{
			}
			else
			{
				// abort if this isn't an ascii or a binary mesh
				_params.logger.log("Unsupported PLY mesh format %s", system::ILogger::ELL_ERROR, word);
				continueReading = false;
			}

			if (continueReading)
			{
				word = ctx.getNextWord();
				if (strcmp(word, "1.0"))
				{
					_params.logger.log("Unsupported PLY mesh version %s",system::ILogger::ELL_WARNING,word);
				}
			}
		}
		else if (strcmp(word,"end_header")==0)
		{
			readingHeader = false;
			if (ctx.IsBinaryFile)
			{
				char* const binaryStartInBuffer = ctx.LineEndPointer + 1;
				const auto* const mappedBase = reinterpret_cast<const char*>(static_cast<const system::IFile*>(_file)->getMappedPointer());
				if (mappedBase)
				{
					const size_t binaryOffset = ctx.getAbsoluteOffset(binaryStartInBuffer);
					const size_t remainingBytes = static_cast<size_t>(binaryOffset < fileSize ? (fileSize - binaryOffset) : 0ull);
					ctx.useMappedBinaryWindow(mappedBase + binaryOffset, remainingBytes);
				}
				else
				{
					ctx.StartPointer = binaryStartInBuffer;
				}
			}
		}
		else
		{
			_params.logger.log("Unknown item in PLY file %s", system::ILogger::ELL_WARNING, word);
		}

		if (readingHeader && continueReading)
		{
			ctx.getNextLine();
			word = ctx.getNextWord();
		}
	}
	while (readingHeader && continueReading);
	headerMs = std::chrono::duration<double, std::milli>(clock_t::now() - headerStart).count();

	//
	if (!continueReading)
		return {};

	// now to read the actual data from the file
	using index_t = uint32_t;
	core::vector<index_t> indices = {};

	// loop through each of the elements
	bool verticesProcessed = false;

	for (uint32_t i=0; i<ctx.ElementList.size(); ++i)
	{
		auto& el = ctx.ElementList[i];
		if (el.Name=="vertex") // TODO: are multiple of these possible in a file? do we create a geometry collection then? Probably not -> https://paulbourke.net/dataformats/ply/
		{
			if (verticesProcessed)
			{
				_params.logger.log("Multiple `vertex` elements not supported!", system::ILogger::ELL_ERROR);
				return {};
			}
			const bool mappedXYZAliasCandidate =
				ctx.IsBinaryFile &&
				(!ctx.IsWrongEndian) &&
				ctx.UsingMappedBinaryWindow &&
				el.Properties.size() == 3u &&
				el.Properties[0].type == EF_R32_SFLOAT &&
				el.Properties[1].type == EF_R32_SFLOAT &&
				el.Properties[2].type == EF_R32_SFLOAT &&
				el.Properties[0].Name == "x" &&
				el.Properties[1].Name == "y" &&
				el.Properties[2].Name == "z";
			if (mappedXYZAliasCandidate)
			{
				if (el.Count > (std::numeric_limits<size_t>::max() / (sizeof(float) * 3ull)))
					return {};
				const size_t mappedBytes = el.Count * sizeof(float) * 3ull;
				if (ctx.StartPointer + mappedBytes > ctx.EndPointer)
					return {};
				const auto vertexStart = clock_t::now();
				auto mappedPosView = plyCreateMappedF32x3View(_file, ctx.StartPointer, mappedBytes);
				if (!mappedPosView)
					return {};
				geometry->setPositionView(std::move(mappedPosView));
				ctx.StartPointer += mappedBytes;
				++fastVertexElementCount;
				const double elapsedMs = std::chrono::duration<double, std::milli>(clock_t::now() - vertexStart).count();
				vertexFastMs += elapsedMs;
				vertexMs += elapsedMs;
				verticesProcessed = true;
				continue;
			}
			ICPUPolygonGeometry::SDataViewBase posView = {}, normalView = {}, uvView = {};
			core::vector<ICPUPolygonGeometry::SDataView> extraViews;
			const auto layoutStart = clock_t::now();
			for (auto& vertexProperty : el.Properties)
			{
				const auto& propertyName = vertexProperty.Name;
				// only positions and normals need to be structured/canonicalized in any way
				auto negotiateFormat = [&vertexProperty](ICPUPolygonGeometry::SDataViewBase& view, const uint8_t component)->void
				{
					assert(getFormatChannelCount(vertexProperty.type)!=0);
					if (getTexelOrBlockBytesize(vertexProperty.type)>getTexelOrBlockBytesize(view.format))
						view.format = vertexProperty.type;
					view.stride = hlsl::max<uint32_t>(view.stride,component);
				};
				if (propertyName=="x")
					negotiateFormat(posView,0);
				else if (propertyName=="y")
					negotiateFormat(posView,1);
				else if (propertyName=="z")
					negotiateFormat(posView,2);
				else if (propertyName=="nx")
					negotiateFormat(normalView,0);
				else if (propertyName=="ny")
					negotiateFormat(normalView,1);
				else if (propertyName=="nz")
					negotiateFormat(normalView,2);
				else if (propertyName=="u" || propertyName=="s")
					negotiateFormat(uvView,0);
				else if (propertyName=="v" || propertyName=="t")
					negotiateFormat(uvView,1);
				else
				{
// TODO: record the `propertyName`
					const auto extraViewStart = clock_t::now();
					extraViews.push_back(createView(vertexProperty.type,el.Count));
					viewCreateMs += std::chrono::duration<double, std::milli>(clock_t::now() - extraViewStart).count();
				}
			}
			layoutNegotiateMs += std::chrono::duration<double, std::milli>(clock_t::now() - layoutStart).count();
			auto setFinalFormat = [&ctx](ICPUPolygonGeometry::SDataViewBase& view)->void
			{
				const auto componentFormat = view.format;
				const auto componentCount = view.stride+1;
				// turn single channel format to multiple
				view.format = [=]()->E_FORMAT
				{
					switch (view.format)
					{
						case EF_R8_SINT:
							switch (componentCount)
							{
								case 1:
									return EF_R8_SINT;
								case 2:
									return EF_R8G8_SINT;
								case 3:
									return EF_R8G8B8_SINT;
								case 4:
									return EF_R8G8B8A8_SINT;
								default:
									break;
							}
							break;
						case EF_R8_UINT:
							switch (componentCount)
							{
								case 1:
									return EF_R8_UINT;
								case 2:
									return EF_R8G8_UINT;
								case 3:
									return EF_R8G8B8_UINT;
								case 4:
									return EF_R8G8B8A8_UINT;
								default:
									break;
							}
							break;
						case EF_R16_SINT:
							switch (componentCount)
							{
								case 1:
									return EF_R16_SINT;
								case 2:
									return EF_R16G16_SINT;
								case 3:
									return EF_R16G16B16_SINT;
								case 4:
									return EF_R16G16B16A16_SINT;
								default:
									break;
							}
							break;
						case EF_R16_UINT:
							switch (componentCount)
							{
								case 1:
									return EF_R16_UINT;
								case 2:
									return EF_R16G16_UINT;
								case 3:
									return EF_R16G16B16_UINT;
								case 4:
									return EF_R16G16B16A16_UINT;
								default:
									break;
							}
							break;
						case EF_R32_SINT:
							switch (componentCount)
							{
								case 1:
									return EF_R32_SINT;
								case 2:
									return EF_R32G32_SINT;
								case 3:
									return EF_R32G32B32_SINT;
								case 4:
									return EF_R32G32B32A32_SINT;
								default:
									break;
							}
							break;
						case EF_R32_UINT:
							switch (componentCount)
							{
								case 1:
									return EF_R32_UINT;
								case 2:
									return EF_R32G32_UINT;
								case 3:
									return EF_R32G32B32_UINT;
								case 4:
									return EF_R32G32B32A32_UINT;
								default:
									break;
							}
							break;
						case EF_R32_SFLOAT:
							switch (componentCount)
							{
								case 1:
									return EF_R32_SFLOAT;
								case 2:
									return EF_R32G32_SFLOAT;
								case 3:
									return EF_R32G32B32_SFLOAT;
								case 4:
									return EF_R32G32B32A32_SFLOAT;
								default:
									break;
							}
							break;
						case EF_R64_SFLOAT:
							switch (componentCount)
							{
								case 1:
									return EF_R64_SFLOAT;
								case 2:
									return EF_R64G64_SFLOAT;
								case 3:
									return EF_R64G64B64_SFLOAT;
								case 4:
									return EF_R64G64B64A64_SFLOAT;
								default:
									break;
							}
							break;
						default:
							break;
					}
					return EF_UNKNOWN;
				}();
				view.stride = getTexelOrBlockBytesize(view.format);
				//
				for (auto c=0u; c<componentCount; c++)
				{
					size_t offset = getTexelOrBlockBytesize(componentFormat)*c;
					ctx.vertAttrIts.push_back({
						.ptr = reinterpret_cast<uint8_t*>(offset),
						.stride = view.stride,
						.dstFmt = componentFormat
					});
				}
			};
			if (posView.format!=EF_UNKNOWN)
			{
				const auto viewCreateStart = clock_t::now();
				auto beginIx = ctx.vertAttrIts.size();
				setFinalFormat(posView);
				auto view = createView(posView.format,el.Count);
				for (const auto size=ctx.vertAttrIts.size(); beginIx!=size; beginIx++)
					ctx.vertAttrIts[beginIx].ptr += ptrdiff_t(view.src.buffer->getPointer())+view.src.offset;
				geometry->setPositionView(std::move(view));
				viewCreateMs += std::chrono::duration<double, std::milli>(clock_t::now() - viewCreateStart).count();
			}
			if (normalView.format!=EF_UNKNOWN)
			{
				const auto viewCreateStart = clock_t::now();
				auto beginIx = ctx.vertAttrIts.size();
				setFinalFormat(normalView);
				auto view = createView(normalView.format,el.Count);
				for (const auto size=ctx.vertAttrIts.size(); beginIx!=size; beginIx++)
					ctx.vertAttrIts[beginIx].ptr += ptrdiff_t(view.src.buffer->getPointer())+view.src.offset;
				geometry->setNormalView(std::move(view));
				viewCreateMs += std::chrono::duration<double, std::milli>(clock_t::now() - viewCreateStart).count();
			}
			if (uvView.format!=EF_UNKNOWN)
			{
				const auto viewCreateStart = clock_t::now();
				auto beginIx = ctx.vertAttrIts.size();
				setFinalFormat(uvView);
				auto view = createView(uvView.format,el.Count);
				for (const auto size=ctx.vertAttrIts.size(); beginIx!=size; beginIx++)
					ctx.vertAttrIts[beginIx].ptr += ptrdiff_t(view.src.buffer->getPointer())+view.src.offset;
				geometry->getAuxAttributeViews()->push_back(std::move(view));
				viewCreateMs += std::chrono::duration<double, std::milli>(clock_t::now() - viewCreateStart).count();
			}
			//
			for (auto& view : extraViews)
				ctx.vertAttrIts.push_back({
					.ptr = reinterpret_cast<uint8_t*>(view.src.buffer->getPointer())+view.src.offset,
					.stride = getTexelOrBlockBytesize(view.composed.format),
					.dstFmt = view.composed.format
				});
			for (auto& view : extraViews)
				geometry->getAuxAttributeViews()->push_back(std::move(view));
			// loop through vertex properties
			const auto vertexStart = clock_t::now();
			const auto fastVertexResult = ctx.readVertexElementFast(el);
			if (fastVertexResult == SContext::EFastVertexReadResult::Success)
			{
				++fastVertexElementCount;
				const double elapsedMs = std::chrono::duration<double, std::milli>(clock_t::now() - vertexStart).count();
				vertexFastMs += elapsedMs;
				vertexMs += elapsedMs;
			}
			else if (fastVertexResult == SContext::EFastVertexReadResult::NotApplicable)
			{
				ctx.readVertex(_params,el);
				const double elapsedMs = std::chrono::duration<double, std::milli>(clock_t::now() - vertexStart).count();
				vertexGenericMs += elapsedMs;
				vertexMs += elapsedMs;
			}
			else
			{
				_params.logger.log("PLY vertex fast path failed on malformed data for %s", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str());
				return {};
			}
			verticesProcessed = true;
		}
		else if (el.Name=="face")
		{
			const auto faceStart = clock_t::now();
			const uint32_t vertexCount32 = vertCount <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ? static_cast<uint32_t>(vertCount) : 0u;
			const auto fastFaceResult = ctx.readFaceElementFast(el,indices,maxIndexRead,faceCount,vertexCount32);
			if (fastFaceResult == SContext::EFastFaceReadResult::Success)
			{
				++fastFaceElementCount;
			}
			else if (fastFaceResult == SContext::EFastFaceReadResult::NotApplicable)
			{
				indices.reserve(indices.size() + el.Count * 3u);
				for (size_t j=0; j<el.Count; ++j)
				{
					if (!ctx.readFace(el, indices, maxIndexRead, vertexCount32))
						return {};
					++faceCount;
				}
			}
			else
			{
				_params.logger.log("PLY face fast path failed on malformed data for %s", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str());
				return {};
			}
			faceMs += std::chrono::duration<double, std::milli>(clock_t::now() - faceStart).count();
		}
		else
		{
			// skip these elements
			const auto skipStart = clock_t::now();
			if (ctx.IsBinaryFile && el.KnownSize)
			{
				const uint64_t bytesToSkip64 = static_cast<uint64_t>(el.KnownSize) * static_cast<uint64_t>(el.Count);
				if (bytesToSkip64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
					return {};
				ctx.moveForward(static_cast<size_t>(bytesToSkip64));
			}
			else
			{
				for (size_t j=0; j<el.Count; ++j)
					el.skipElement(ctx);
			}
			skipMs += std::chrono::duration<double, std::milli>(clock_t::now() - skipStart).count();
		}
	}

	const auto aabbStart = clock_t::now();
	CPolygonGeometryManipulator::recomputeAABB(geometry.get());
	aabbMs = std::chrono::duration<double, std::milli>(clock_t::now() - aabbStart).count();

	const uint64_t indexCount = static_cast<uint64_t>(indices.size());
	const auto indexStart = clock_t::now();
	if (indices.empty())
	{
		// no index buffer means point cloud
		geometry->setIndexing(IPolygonGeometryBase::PointList());
	}
	else
	{
		if (vertCount != 0u && maxIndexRead >= vertCount)
		{
			_params.logger.log("PLY indices out of range for %s", system::ILogger::ELL_ERROR, _file->getFileName().string().c_str());
			return {};
		}

		geometry->setIndexing(IPolygonGeometryBase::TriangleList());
		const bool canUseU16 = (vertCount != 0u) ? (vertCount <= std::numeric_limits<uint16_t>::max()) : (maxIndexRead <= std::numeric_limits<uint16_t>::max());
		if (canUseU16)
		{
			core::vector<uint16_t> indices16(indices.size());
			for (size_t i = 0u; i < indices.size(); ++i)
				indices16[i] = static_cast<uint16_t>(indices[i]);
			auto view = plyCreateAdoptedU16IndexView(std::move(indices16));
			if (!view)
				return {};
			geometry->setIndexView(std::move(view));
		}
		else
		{
			auto view = plyCreateAdoptedU32IndexView(std::move(indices));
			if (!view)
				return {};
			geometry->setIndexView(std::move(view));
		}
	}
	indexBuildMs = std::chrono::duration<double, std::milli>(clock_t::now() - indexStart).count();

	if (_params.loaderFlags & IAssetLoader::ELPF_COMPUTE_CONTENT_HASHES)
	{
		const auto hashStart = clock_t::now();
		plyRecomputeContentHashesParallel(geometry.get());
		hashRangeMs = std::chrono::duration<double, std::milli>(clock_t::now() - hashStart).count();
	}

	const auto totalMs = std::chrono::duration<double, std::milli>(clock_t::now() - totalStart).count();
	const double stageRemainderMs = std::max(0.0, totalMs - (headerMs + vertexMs + faceMs + skipMs + layoutNegotiateMs + viewCreateMs + hashRangeMs + indexBuildMs + aabbMs));
	const uint64_t ioMinRead = ctx.readCallCount ? ctx.readMinBytes : 0ull;
	const uint64_t ioAvgRead = ctx.readCallCount ? (ctx.readBytesTotal / ctx.readCallCount) : 0ull;
	if (
		fileSize > (1ull << 20) &&
		(
			ioAvgRead < 1024ull ||
			(ioMinRead < 64ull && ctx.readCallCount > 1024ull)
		)
	)
	{
		_params.logger.log(
			"PLY loader tiny-io guard: file=%s reads=%llu min=%llu avg=%llu",
			system::ILogger::ELL_WARNING,
			_file->getFileName().string().c_str(),
			static_cast<unsigned long long>(ctx.readCallCount),
			static_cast<unsigned long long>(ioMinRead),
			static_cast<unsigned long long>(ioAvgRead));
	}
	_params.logger.log(
		"PLY loader perf: file=%s total=%.3f ms header=%.3f vertex=%.3f vertex_fast_ms=%.3f vertex_generic_ms=%.3f face=%.3f skip=%.3f layout_negotiate=%.3f view_create=%.3f hash_range=%.3f index=%.3f aabb=%.3f remainder=%.3f binary=%d verts=%llu faces=%llu idx=%llu vertex_fast=%llu face_fast=%llu io_reads=%llu io_min_read=%llu io_avg_read=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
		system::ILogger::ELL_PERFORMANCE,
		_file->getFileName().string().c_str(),
		totalMs,
		headerMs,
		vertexMs,
		vertexFastMs,
		vertexGenericMs,
		faceMs,
		skipMs,
		layoutNegotiateMs,
		viewCreateMs,
		hashRangeMs,
		indexBuildMs,
		aabbMs,
		stageRemainderMs,
		ctx.IsBinaryFile ? 1 : 0,
		static_cast<unsigned long long>(vertCount),
		static_cast<unsigned long long>(faceCount),
		static_cast<unsigned long long>(indexCount),
		static_cast<unsigned long long>(fastVertexElementCount),
		static_cast<unsigned long long>(fastFaceElementCount),
		static_cast<unsigned long long>(ctx.readCallCount),
		static_cast<unsigned long long>(ioMinRead),
		static_cast<unsigned long long>(ioAvgRead),
		toString(_params.ioPolicy.strategy),
		toString(ioPlan.strategy),
		static_cast<unsigned long long>(ioPlan.chunkSizeBytes),
		ioPlan.reason);

	auto meta = core::make_smart_refctd_ptr<CPLYMetadata>();
	return SAssetBundle(std::move(meta),{std::move(geometry)});
}


} // end namespace nbl::asset
#endif // _NBL_COMPILE_WITH_PLY_LOADER_

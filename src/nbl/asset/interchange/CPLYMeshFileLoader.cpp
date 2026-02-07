// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifdef _NBL_COMPILE_WITH_PLY_LOADER_


#include "CPLYMeshFileLoader.h"
#include "nbl/asset/metadata/CPLYMetadata.h"

#include <numeric>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cstring>

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
inline T byteswap(const T& v)
{
	T retval;
	auto it = reinterpret_cast<const char*>(&v);
	std::reverse_copy(it,it+sizeof(T),reinterpret_cast<char*>(&retval));
	return retval;
}

struct SContext
{
	
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

		inline bool isList() const {return type==EF_UNKNOWN && asset::isIntegerFormat(list.countType) && asset::isIntegerFormat(list.itemType);}

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

	inline void init(size_t _ioReadWindowSize = 50ull << 10)
	{
		ioReadWindowSize = std::max<size_t>(_ioReadWindowSize, 50ull << 10);
		Buffer.resize(ioReadWindowSize + 1ull, '\0');
		EndPointer = StartPointer = Buffer.data();
		LineEndPointer = EndPointer-1;

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
		const size_t usableBufferSize = Buffer.size() > 0ull ? Buffer.size() - 1ull : 0ull;
		if (usableBufferSize <= length)
		{
			EndOfFile = true;
			return;
		}
		const size_t requestSize = usableBufferSize - length;
		system::IFile::success_t success;
		inner.mainFile->read(success,EndPointer,fileOffset,requestSize);
		const size_t bytesRead = success.getBytesProcessed();
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
	// skips x bytes in the file, getting more data if required
	void moveForward(const size_t bytes)
	{
		assert(IsBinaryFile);
		if (StartPointer+bytes>=EndPointer)
			fillBuffer();

		if (StartPointer+bytes<EndPointer)
			StartPointer += bytes;
		else
			StartPointer = EndPointer;
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
		return std::atoi(getNextWord());
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
		return std::strtod(getNextWord(), nullptr);
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
	inline void readVertex(const IAssetLoader::SAssetLoadParams& _params, const SElement& el)
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
	bool readFace(const SElement& Element, core::vector<uint32_t>& _outIndices, uint32_t& _maxIndex)
	{
		if (!IsBinaryFile)
			getNextLine();

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
				auto emitFan = [&_outIndices, &_maxIndex](auto&& readIndex, const uint32_t faceVertexCount)->void
				{
					uint32_t i0 = readIndex();
					uint32_t i1 = readIndex();
					uint32_t i2 = readIndex();
					_maxIndex = std::max(_maxIndex, std::max(i0, std::max(i1, i2)));
					_outIndices.push_back(i0);
					_outIndices.push_back(i1);
					_outIndices.push_back(i2);
					uint32_t prev = i2;
					for (uint32_t j = 3u; j < faceVertexCount; ++j)
					{
						const uint32_t idx = readIndex();
						_maxIndex = std::max(_maxIndex, idx);
						_outIndices.push_back(i0);
						_outIndices.push_back(prev);
						_outIndices.push_back(idx);
						prev = idx;
					}
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
						emitFan(readIndex, count);
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
						emitFan(readIndex, count);
						StartPointer = reinterpret_cast<char*>(const_cast<uint8_t*>(ptr));
						continue;
					}
				}

				auto readIndex = [&]() -> uint32_t
				{
					return static_cast<uint32_t>(getInt(srcIndexFmt));
				};
				emitFan(readIndex, count);
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

	bool readFaceElementFast(const SElement& element, core::vector<uint32_t>& _outIndices, uint32_t& _maxIndex, uint64_t& _faceCount)
	{
		if (!IsBinaryFile || IsWrongEndian)
			return false;
		if (element.Properties.size() != 1u)
			return false;

		const auto& prop = element.Properties[0];
		if (!prop.isList() || (prop.Name != "vertex_indices" && prop.Name != "vertex_index"))
			return false;
		if (prop.list.countType != EF_R8_UINT)
			return false;

		const E_FORMAT srcIndexFmt = prop.list.itemType;
		if (srcIndexFmt != EF_R32_UINT && srcIndexFmt != EF_R16_UINT)
			return false;

		const size_t indexSize = srcIndexFmt == EF_R32_UINT ? sizeof(uint32_t) : sizeof(uint16_t);
		const size_t minTriangleRecordSize = sizeof(uint8_t) + indexSize * 3u;
		const size_t minBytesNeeded = element.Count * minTriangleRecordSize;
		if (StartPointer + minBytesNeeded <= EndPointer)
		{
			char* scan = StartPointer;
			bool allTriangles = true;
			for (size_t j = 0u; j < element.Count; ++j)
			{
				const uint8_t c = static_cast<uint8_t>(*scan++);
				if (c != 3u)
				{
					allTriangles = false;
					break;
				}
				scan += indexSize * 3u;
			}

			if (allTriangles)
			{
				const size_t oldSize = _outIndices.size();
				_outIndices.resize(oldSize + element.Count * 3u);
				uint32_t* out = _outIndices.data() + oldSize;
				const uint8_t* ptr = reinterpret_cast<const uint8_t*>(StartPointer);

				if (srcIndexFmt == EF_R32_UINT)
				{
					for (size_t j = 0u; j < element.Count; ++j)
					{
						++ptr; // list count
						uint32_t i0 = 0u;
						uint32_t i1 = 0u;
						uint32_t i2 = 0u;
						std::memcpy(&i0, ptr, sizeof(i0));
						ptr += sizeof(i0);
						std::memcpy(&i1, ptr, sizeof(i1));
						ptr += sizeof(i1);
						std::memcpy(&i2, ptr, sizeof(i2));
						ptr += sizeof(i2);
						_maxIndex = std::max(_maxIndex, std::max(i0, std::max(i1, i2)));
						out[0] = i0;
						out[1] = i1;
						out[2] = i2;
						out += 3;
					}
				}
				else
				{
					for (size_t j = 0u; j < element.Count; ++j)
					{
						++ptr; // list count
						uint16_t t0 = 0u;
						uint16_t t1 = 0u;
						uint16_t t2 = 0u;
						std::memcpy(&t0, ptr, sizeof(t0));
						ptr += sizeof(t0);
						std::memcpy(&t1, ptr, sizeof(t1));
						ptr += sizeof(t1);
						std::memcpy(&t2, ptr, sizeof(t2));
						ptr += sizeof(t2);
						const uint32_t i0 = t0;
						const uint32_t i1 = t1;
						const uint32_t i2 = t2;
						_maxIndex = std::max(_maxIndex, std::max(i0, std::max(i1, i2)));
						out[0] = i0;
						out[1] = i1;
						out[2] = i2;
						out += 3;
					}
				}

				StartPointer = reinterpret_cast<char*>(const_cast<uint8_t*>(ptr));
				_faceCount += element.Count;
				return true;
			}
		}

		_outIndices.reserve(_outIndices.size() + element.Count * 3u);
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
		auto readIndex = [&ensureBytes, this, srcIndexFmt](uint32_t& out)->bool
		{
			if (srcIndexFmt == EF_R32_UINT)
			{
				if (!ensureBytes(sizeof(uint32_t)))
					return false;
				std::memcpy(&out, StartPointer, sizeof(uint32_t));
				StartPointer += sizeof(uint32_t);
				return true;
			}

			if (!ensureBytes(sizeof(uint16_t)))
				return false;
			uint16_t v = 0u;
			std::memcpy(&v, StartPointer, sizeof(uint16_t));
			StartPointer += sizeof(uint16_t);
			out = v;
			return true;
		};

		for (size_t j = 0u; j < element.Count; ++j)
		{
			int32_t countSigned = 0;
			if (!readCount(countSigned))
				return false;
			if (countSigned < 0)
				return false;
			const uint32_t count = static_cast<uint32_t>(countSigned);
			if (count < 3u)
			{
				uint32_t dummy = 0u;
				for (uint32_t k = 0u; k < count; ++k)
				{
					if (!readIndex(dummy))
						return false;
				}
				++_faceCount;
				continue;
			}

			uint32_t i0 = 0u;
			uint32_t i1 = 0u;
			uint32_t i2 = 0u;
			if (!readIndex(i0) || !readIndex(i1) || !readIndex(i2))
				return false;

			_maxIndex = std::max(_maxIndex, std::max(i0, std::max(i1, i2)));
			_outIndices.push_back(i0);
			_outIndices.push_back(i1);
			_outIndices.push_back(i2);

			uint32_t prev = i2;
			for (uint32_t k = 3u; k < count; ++k)
			{
				uint32_t idx = 0u;
				if (!readIndex(idx))
					return false;
				_maxIndex = std::max(_maxIndex, idx);
				_outIndices.push_back(i0);
				_outIndices.push_back(prev);
				_outIndices.push_back(idx);
				prev = idx;
			}

			++_faceCount;
		}

		return true;
	}

	IAssetLoader::SAssetLoadContext inner;
	uint32_t topHierarchyLevel;
	IAssetLoader::IAssetLoaderOverride* loaderOverride;
	// input buffer must be at least twice as long as the longest line in the file
	core::vector<char> Buffer;
	size_t ioReadWindowSize = 50ull << 10;
	core::vector<SElement> ElementList = {};
	char* StartPointer = nullptr, *EndPointer = nullptr, *LineEndPointer = nullptr;
	int32_t LineLength = 0;
	int32_t WordLength = -1; // this variable is a misnomer, its really the offset to next word minus one
	bool IsBinaryFile = false, IsWrongEndian = false, EndOfFile = false;
	size_t fileOffset = {};
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
	double faceMs = 0.0;
	double skipMs = 0.0;
	double hashRangeMs = 0.0;
	double indexBuildMs = 0.0;
	double aabbMs = 0.0;
	uint64_t faceCount = 0u;
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
	const uint64_t desiredReadWindow = ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile ? (fileSize + 1ull) : ioPlan.chunkSizeBytes;
	const uint64_t safeReadWindow = std::min<uint64_t>(desiredReadWindow, static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 1ull));
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
			el.Count = atoi(ctx.getNextWord());
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
				ctx.StartPointer = ctx.LineEndPointer+1;
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
			ICPUPolygonGeometry::SDataViewBase posView = {}, normalView = {}, uvView = {};
			core::vector<ICPUPolygonGeometry::SDataView> extraViews;
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
					extraViews.push_back(createView(vertexProperty.type,el.Count));
				}
			}
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
				auto beginIx = ctx.vertAttrIts.size();
				setFinalFormat(posView);
				auto view = createView(posView.format,el.Count);
				for (const auto size=ctx.vertAttrIts.size(); beginIx!=size; beginIx++)
					ctx.vertAttrIts[beginIx].ptr += ptrdiff_t(view.src.buffer->getPointer())+view.src.offset;
				geometry->setPositionView(std::move(view));
			}
			if (normalView.format!=EF_UNKNOWN)
			{
				auto beginIx = ctx.vertAttrIts.size();
				setFinalFormat(normalView);
				auto view = createView(normalView.format,el.Count);
				for (const auto size=ctx.vertAttrIts.size(); beginIx!=size; beginIx++)
					ctx.vertAttrIts[beginIx].ptr += ptrdiff_t(view.src.buffer->getPointer())+view.src.offset;
				geometry->setNormalView(std::move(view));
			}
			if (uvView.format!=EF_UNKNOWN)
			{
				auto beginIx = ctx.vertAttrIts.size();
				setFinalFormat(uvView);
				auto view = createView(uvView.format,el.Count);
				for (const auto size=ctx.vertAttrIts.size(); beginIx!=size; beginIx++)
					ctx.vertAttrIts[beginIx].ptr += ptrdiff_t(view.src.buffer->getPointer())+view.src.offset;
				geometry->getAuxAttributeViews()->push_back(std::move(view));
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
			ctx.readVertex(_params,el);
			vertexMs += std::chrono::duration<double, std::milli>(clock_t::now() - vertexStart).count();
			verticesProcessed = true;
		}
		else if (el.Name=="face")
		{
			const auto faceStart = clock_t::now();
			indices.reserve(indices.size() + el.Count * 3u);
			for (size_t j=0; j<el.Count; ++j)
			{
				if (!ctx.readFace(el,indices,maxIndexRead))
					return {};
				++faceCount;
			}
			faceMs += std::chrono::duration<double, std::milli>(clock_t::now() - faceStart).count();
		}
		else
		{
			// skip these elements
			const auto skipStart = clock_t::now();
			for (size_t j=0; j<el.Count; ++j)
				el.skipElement(ctx);
			skipMs += std::chrono::duration<double, std::milli>(clock_t::now() - skipStart).count();
		}
	}

	hashRangeMs = 0.0;

	const auto aabbStart = clock_t::now();
	CPolygonGeometryManipulator::recomputeAABB(geometry.get());
	aabbMs = std::chrono::duration<double, std::milli>(clock_t::now() - aabbStart).count();

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
		if (maxIndexRead <= std::numeric_limits<uint16_t>::max())
		{
			auto view = IGeometryLoader::createView(EF_R16_UINT, indices.size());
			if (!view)
				return {};
			auto* dst = reinterpret_cast<uint16_t*>(view.getPointer());
			for (size_t i = 0u; i < indices.size(); ++i)
				dst[i] = static_cast<uint16_t>(indices[i]);
			geometry->setIndexView(std::move(view));
		}
		else
		{
			auto view = IGeometryLoader::createView(EF_R32_UINT, indices.size());
			if (!view)
				return {};
			std::memcpy(view.getPointer(), indices.data(), indices.size() * sizeof(uint32_t));
			geometry->setIndexView(std::move(view));
		}
	}
	indexBuildMs = std::chrono::duration<double, std::milli>(clock_t::now() - indexStart).count();

	const auto totalMs = std::chrono::duration<double, std::milli>(clock_t::now() - totalStart).count();
	_params.logger.log(
		"PLY loader perf: file=%s total=%.3f ms header=%.3f vertex=%.3f face=%.3f skip=%.3f hash_range=%.3f index=%.3f aabb=%.3f binary=%d verts=%llu faces=%llu idx=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
		system::ILogger::ELL_PERFORMANCE,
		_file->getFileName().string().c_str(),
		totalMs,
		headerMs,
		vertexMs,
		faceMs,
		skipMs,
		hashRangeMs,
		indexBuildMs,
		aabbMs,
		ctx.IsBinaryFile ? 1 : 0,
		static_cast<unsigned long long>(vertCount),
		static_cast<unsigned long long>(faceCount),
		static_cast<unsigned long long>(indices.size()),
		toString(_params.ioPolicy.strategy),
		toString(ioPlan.strategy),
		static_cast<unsigned long long>(ioPlan.chunkSizeBytes),
		ioPlan.reason);

	auto meta = core::make_smart_refctd_ptr<CPLYMetadata>();
	return SAssetBundle(std::move(meta),{std::move(geometry)});
}


} // end namespace nbl::asset
#endif // _NBL_COMPILE_WITH_PLY_LOADER_

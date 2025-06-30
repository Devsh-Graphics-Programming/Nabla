// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifdef _NBL_COMPILE_WITH_PLY_LOADER_


#include "CPLYMeshFileLoader.h"

#include <numeric>

#include "nbl/asset/IAssetManager.h"

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"

//#include "nbl/asset/utils/IMeshManipulator.h"


namespace nbl::asset
{

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
			else if (strcmp(typeString, "long")==0 || strcmp(typeString, "int")==0 || strcmp(typeString, "int16")==0)
				return EF_R32_SINT;
			else if (strcmp(typeString, "ulong")==0 || strcmp(typeString, "uint16")==0)
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
					_ctx.getInt(list.countType);
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

	inline void init()
	{
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
		const size_t requestSize = Buffer.size()-length;
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
		return std::atoi(getNextWord());
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
			// conversion required? 
			if (it.dstFmt!=prop.type)
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
	bool readFace(const SElement& Element, core::vector<uint32_t>& _outIndices)
	{
		if (!IsBinaryFile)
			getNextLine();

		for (const auto& prop : Element.Properties)
		{
			if (prop.isList() && (prop.Name=="vertex_indices" || prop.Name == "vertex_index"))
			{
				const uint32_t count = getInt(prop.list.countType);
				//_NBL_DEBUG_BREAK_IF(count != 3)
				const auto srcIndexFmt = prop.list.itemType;

				_outIndices.push_back(getInt(srcIndexFmt));
				_outIndices.push_back(getInt(srcIndexFmt));
				_outIndices.push_back(getInt(srcIndexFmt));
				// TODO: handle varying vertex count faces via variable vertex count geometry collections (PLY loader should be a Geometry Collection loader)
				for (auto j=3u; j<count; ++j)
				{
					// this seems to be a triangle fan ?
					_outIndices.push_back(_outIndices.front());
					_outIndices.push_back(_outIndices.back());
					_outIndices.push_back(getInt(srcIndexFmt));
				}
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

	IAssetLoader::SAssetLoadContext inner;
	uint32_t topHierarchyLevel;
	IAssetLoader::IAssetLoaderOverride* loaderOverride;
	// input buffer must be at least twice as long as the longest line in the file
	std::array<char,50<<10> Buffer; // 50kb seems sane to store a line
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

	SContext ctx = {
		asset::IAssetLoader::SAssetLoadContext{
			_params,
			_file
		},
		_hierarchyLevel,
		_override
	};
	ctx.init();

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

	//
	if (!continueReading)
		return {};

	// now to read the actual data from the file
	using index_t = uint32_t;
	core::vector<index_t> indices = {};
	//
	auto createView = [](const E_FORMAT format, const size_t elCount)->ICPUPolygonGeometry::SDataView
	{
		const auto stride = asset::getTexelOrBlockBytesize(format);
		auto buffer = ICPUBuffer::create({stride*elCount});
		return {
			.composed = {
				.stride = stride,
				.format = format,
				.rangeFormat = IGeometryBase::getMatchingAABBFormat(format)
			},
			.src = {
				.offset = 0,
				.size = buffer->getSize(),
				.buffer = std::move(buffer)
			}
		};
	};

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
			ICPUPolygonGeometry::SDataViewBase posView = {}, normalView = {};
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
				else
				{
// TODO: record the `propertyName`
					geometry->getAuxAttributeViews()->push_back(createView(vertexProperty.type,el.Count));
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
			//
			for (auto& view : *geometry->getAuxAttributeViews())
				ctx.vertAttrIts.push_back({
					.ptr = reinterpret_cast<uint8_t*>(view.src.buffer->getPointer())+view.src.offset,
					.stride = getTexelOrBlockBytesize(view.composed.format),
					.dstFmt = view.composed.format
				});
			// loop through vertex properties
			ctx.readVertex(_params,el);
			verticesProcessed = true;
		}
		else if (el.Name=="face")
		{
			for (size_t j=0; j<el.Count; ++j)
				ctx.readFace(el,indices);
		}
		else
		{
			// skip these elements
			for (size_t j=0; j<el.Count; ++j)
				el.skipElement(ctx);
		}
	}

	if (indices.empty())
	{
		// no index buffer means point cloud
		geometry->setIndexing(IPolygonGeometryBase::PointList());
	}
	else
	{
		geometry->setIndexing(IPolygonGeometryBase::TriangleList());
		auto buffer = ICPUBuffer::create({{indices.size()*sizeof(index_t),IBuffer::EUF_INDEX_BUFFER_BIT},indices.data()});
		hlsl::shapes::AABB<4,index_t> aabb;
		aabb.minVx[0] = *std::min_element(indices.begin(),indices.end());
		aabb.maxVx[0] = *std::max_element(indices.begin(),indices.end());
		geometry->setIndexView({
			.composed = {
				.encodedDataRange = {.u32=aabb},
				.stride = sizeof(index_t),
				.format = EF_R32_UINT,
				.rangeFormat = IGeometryBase::EAABBFormat::U32
			},
			.src = {.offset=0,.size=buffer->getSize(),.buffer=std::move(buffer)}
		});
	}

	CPolygonGeometryManipulator::recomputeContentHashes(geometry.get());

	auto meta = core::make_smart_refctd_ptr<CPLYMetadata>();
	return SAssetBundle(std::move(meta),{std::move(geometry)});
}


} // end namespace nbl::asset
#endif // _NBL_COMPILE_WITH_PLY_LOADER_

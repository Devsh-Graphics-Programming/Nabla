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

#if 0
void CPLYMeshFileLoader::initialize()
{
	auto precomputeAndCachePipeline = [&](CPLYMeshFileLoader::E_TYPE type, bool indexBufferBindingAvailable)
	{

			auto mbPipelineLayout = defaultOverride.findDefaultAsset<ICPUPipelineLayout>("nbl/builtin/pipeline_layout/loader/PLY", fakeContext, 0u).first;

			const std::array<SVertexInputAttribParams, 4> vertexAttribParamsAllOptions =
			{
				SVertexInputAttribParams(0u, EF_R32G32B32_SFLOAT, 0),
				SVertexInputAttribParams(1u, EF_R32G32B32A32_SFLOAT, 0),
				SVertexInputAttribParams(2u, EF_R32G32_SFLOAT, 0),
				SVertexInputAttribParams(3u, EF_R32G32B32_SFLOAT, 0)
			};

			SVertexInputParams inputParams;
			
			std::vector<uint8_t> availableAttributes = { ET_POS };
			if (type != ET_POS)
				availableAttributes.push_back(static_cast<uint8_t>(type));

			for (auto& attrib : availableAttributes)
			{
				const auto currentBitmask = core::createBitmask({ attrib });
				inputParams.enabledBindingFlags |= currentBitmask;
				inputParams.enabledAttribFlags |= currentBitmask;
				inputParams.bindings[attrib] = { asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(vertexAttribParamsAllOptions[attrib].format)), EVIR_PER_VERTEX };
				inputParams.attributes[attrib] = vertexAttribParamsAllOptions[attrib];
			}
		
			SBlendParams blendParams;
			SPrimitiveAssemblyParams primitiveAssemblyParams;
			if (indexBufferBindingAvailable)
				primitiveAssemblyParams.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST;
			else
				primitiveAssemblyParams.primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_POINT_LIST;

	};
}
#endif

struct SContext
{
	
	//
	struct SProperty
	{
		enum EType : uint8_t
		{
			Int8,
			Int16,
			Int32,
			Float32,
			Float64,
			List,
			Unknown
		};
		static EType getType(const char* typeString)
		{
			if (strcmp(typeString, "char") == 0 ||
				strcmp(typeString, "uchar") == 0 ||
				strcmp(typeString, "int8") == 0 ||
				strcmp(typeString, "uint8") == 0)
			{
				return EType::Int8;
			}
			else if (strcmp(typeString, "uint") == 0 ||
				strcmp(typeString, "int16") == 0 ||
				strcmp(typeString, "uint16") == 0 ||
				strcmp(typeString, "short") == 0 ||
				strcmp(typeString, "ushort") == 0)
			{
				return EType::Int16;
			}
			else if (strcmp(typeString, "int") == 0 ||
				strcmp(typeString, "long") == 0 ||
				strcmp(typeString, "ulong") == 0 ||
				strcmp(typeString, "int32") == 0 ||
				strcmp(typeString, "uint32") == 0)
			{
				return EType::Int32;
			}
			else if (strcmp(typeString, "float") == 0 ||
				strcmp(typeString, "float32") == 0)
			{
				return EType::Float32;
			}
			else if (strcmp(typeString, "float64") == 0 ||
				strcmp(typeString, "double") == 0)
			{
				return EType::Float64;
			}
			else if (strcmp(typeString, "list") == 0)
			{
				return EType::List;
			}
			else
			{
				// unsupported type.
				// cannot be loaded in binary mode
				return EType::Unknown;
			}
		}
		static inline uint8_t getTypeSize(const EType type)
		{
			switch (type)
			{
				case EType::Int8:
					return 1;
				case EType::Int16:
					return 2;
				case EType::Int32:
				case EType::Float32:
					return 4;
				case EType::Float64:
					return 8;
				default:
					return 0;
			}
		}
		static inline uint8_t isTypeInt(const EType type)
		{
			switch (type)
			{
				case EType::Int8: [[fallthrough]];
				case EType::Int16: [[fallthrough]];
				case EType::Int32:
					return true;
				default:
					return false;
			}
		}
		static inline bool isTypeFloat(const EType type) {return type==EType::Float32 || type==EType::Float64;}

		inline uint8_t size() const {return getTypeSize(type);}
		inline bool isFloat() const {return isTypeFloat(type);}

		void skip(SContext& _ctx)
		{
			if (type==EType::List)
			{
				int32_t count = _ctx.getInt(list.countType);

				for (decltype(count) i=0; i<count; ++i)
					_ctx.getInt(list.countType);
			}
			else if (_ctx.IsBinaryFile)
				_ctx.moveForward(size());
			else
				_ctx.getNextWord();
		}

		std::string Name;
		EType type;
		struct SListTypes
		{
			EType countType;
			EType itemType;
		} list;
	};
	struct SElement
	{
		void skipElement(SContext& _ctx)
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
			LineEndPointer = terminator;
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
				*wordEnd = '\0';
			// find next word
			auto notWhiteSpace = [WhiteSpace](const char c)->bool
			{
				return std::find(WhiteSpace.begin(),WhiteSpace.end(),c)!=WhiteSpace.end();
			};
			auto nextWord = std::find_if(wordEnd,LineEndPointer,notWhiteSpace);
			WordLength = std::distance(StartPointer,wordEnd)-1;
		}
		// return pointer to the start of current word
		return _ctx.StartPointer;
	}
	// skips x bytes in the file, getting more data if required
	void moveForward(const size_t bytes)
	{
		assert(IsBinaryFile);
		if (StartPointer+bytes>=_ctx.EndPointer)
			fillBuffer();

		if (StartPointer+bytes<EndPointer)
			StartPointer += bytes;
		else
			StartPointer = EndPointer;
	}

	// read the next int from the file and move the start pointer along
	using widest_int_t = uint32_t;
	widest_int_t getInt(const SProperty::EType t)
	{
		assert(SProperty::isTypeInt(t));
		if (IsBinaryFile)
		{
			if (!EndOfFile && StartPointer+sizeof(widest_int_t)>EndPointer)
				fillBuffer();

			switch (t)
			{
				case SProperty::EType::Int8:
					if (StartPointer+sizeof(int8_t)>EndPointer)
						break;
					return *(_ctx.StartPointer++);
				case SProperty::EType::Int16:
				{
					if (StartPointer+sizeof(int16_t)>EndPointer)
						break;
					auto retval = *(reinterpret_cast<int16_t*&>(_ctx.StartPointer)++);
					if (_ctx.IsWrongEndian)
						retval = core::Byteswap::byteswap(retval);
					return retval;
				}
				case SProperty::EType::Int32:
				{
					if (StartPointer+sizeof(int32_t)>EndPointer)
						break;
					auto retval = *(reinterpret_cast<int32_t*&>(_ctx.StartPointer)++);
					if (_ctx.IsWrongEndian)
						retval = core::Byteswap::byteswap(retval);
					return retval;
				}
				default:
					return 0;
			}
		}
		else
			return std::atoi(_ctx.getNextWord());
	}
	bool readVertex(SContext& _ctx, SBufferBinding<asset::ICPUBuffer> outAttributes[4], const uint32_t& currentVertexIndex, const IAssetLoader::SAssetLoadParams& _params)
	{
		if (!_ctx.IsBinaryFile)
			getNextLine(_ctx);

		std::pair<bool, core::vectorSIMDf> attribs[4];
		attribs[ET_COL].second.W = 1.f;
		attribs[ET_NORM].second.Y = 1.f;

		constexpr auto ET_POS_BYTESIZE = asset::getTexelOrBlockBytesize<EF_R32G32B32_SFLOAT>();
		constexpr auto ET_NORM_BYTESIZE = asset::getTexelOrBlockBytesize<EF_R32G32B32_SFLOAT>();
		constexpr auto ET_UV_BYTESIZE = asset::getTexelOrBlockBytesize<EF_R32G32_SFLOAT>();
		constexpr auto ET_COL_BYTESIZE = asset::getTexelOrBlockBytesize<EF_R32G32B32A32_SFLOAT>();

		bool result = false;
		for (uint32_t i = 0; i < Element.Properties.size(); ++i)
		{
			E_PLY_PROPERTY_TYPE t = Element.Properties[i].Type;

			if (Element.Properties[i].Name == "x")
			{
				auto& value = attribs[ET_POS].second.X = getFloat(_ctx, t);
				attribs[ET_POS].first = true;

				if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
					performActionBasedOnOrientationSystem<float>(value, [](float& varToFlip) { varToFlip = -varToFlip; });

				const size_t propertyOffset = ET_POS_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_POS].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[0] = value;
			}
			else if (Element.Properties[i].Name == "y")
			{
				auto& value = attribs[ET_POS].second.Y = getFloat(_ctx, t);
				attribs[ET_POS].first = true;

				const size_t propertyOffset = ET_POS_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_POS].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[1] = value;
			}
			else if (Element.Properties[i].Name == "z")
			{
				auto& value = attribs[ET_POS].second.Z = getFloat(_ctx, t);
				attribs[ET_POS].first = true;

				const size_t propertyOffset = ET_POS_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_POS].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[2] = value;
			}
			else if (Element.Properties[i].Name == "nx")
			{
				auto& value = attribs[ET_NORM].second.X = getFloat(_ctx, t);
				attribs[ET_NORM].first = result = true;

				if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
					performActionBasedOnOrientationSystem<float>(attribs[ET_NORM].second.X, [](float& varToFlip) { varToFlip = -varToFlip; });

				const size_t propertyOffset = ET_NORM_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_NORM].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[0] = value;
			}
			else if (Element.Properties[i].Name == "ny")
			{
				auto& value = attribs[ET_NORM].second.Y = getFloat(_ctx, t);
				attribs[ET_NORM].first = result = true;

				const size_t propertyOffset = ET_NORM_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_NORM].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[1] = value;
			}
			else if (Element.Properties[i].Name == "nz")
			{
				auto& value = attribs[ET_NORM].second.Z = getFloat(_ctx, t);
				attribs[ET_NORM].first = result = true;

				const size_t propertyOffset = ET_NORM_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_NORM].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[2] = value;
			}
			// there isn't a single convention for the UV, some softwares like Blender or Assimp use "st" instead of "uv"
			else if (Element.Properties[i].Name == "u" || Element.Properties[i].Name == "s")
			{
				auto& value = attribs[ET_UV].second.X = getFloat(_ctx, t);
				attribs[ET_UV].first = true;

				const size_t propertyOffset = ET_UV_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_UV].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[0] = value;
			}
			else if (Element.Properties[i].Name == "v" || Element.Properties[i].Name == "t")
			{
				auto& value = attribs[ET_UV].second.Y = getFloat(_ctx, t);
				attribs[ET_UV].first = true;

				const size_t propertyOffset = ET_UV_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_UV].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[1] = value;
			}
			else if (Element.Properties[i].Name == "red")
			{
				float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(_ctx.getInt( t)) / 255.f;
				attribs[ET_COL].second.X = value;
				attribs[ET_COL].first = true;

				const size_t propertyOffset = ET_COL_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_COL].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[0] = value;
			}
			else if (Element.Properties[i].Name == "green")
			{
				float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(_ctx.getInt( t)) / 255.f;
				attribs[ET_COL].second.Y = value;
				attribs[ET_COL].first = true;

				const size_t propertyOffset = ET_COL_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_COL].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[1] = value;
			}
			else if (Element.Properties[i].Name == "blue")
			{
				float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(_ctx.getInt( t)) / 255.f;
				attribs[ET_COL].second.Z = value;
				attribs[ET_COL].first = true;

				const size_t propertyOffset = ET_COL_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_COL].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[2] = value;
			}
			else if (Element.Properties[i].Name == "alpha")
			{
				float value = Element.Properties[i].isFloat() ? getFloat(_ctx, t) : float(_ctx.getInt( t)) / 255.f;
				attribs[ET_COL].second.W = value;
				attribs[ET_COL].first = true;

				const size_t propertyOffset = ET_COL_BYTESIZE * currentVertexIndex;
				uint8_t* data = reinterpret_cast<uint8_t*>(outAttributes[ET_COL].buffer->getPointer()) + propertyOffset;

				reinterpret_cast<float*>(data)[3] = value;
			}
			else
				skipProperty(_ctx, Element.Properties[i]);
		}

		return result;
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

			if (!ctx.ElementList.empty())
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
				using EType = SContext::SProperty::EType;
				if (prop.type==EType::List)
				{
					el.KnownSize = false;

					word = ctx.getNextWord();

					prop.list.countType = prop.getType(word);
					if (ctx.IsBinaryFile && prop.list.countType==EType::Unknown)
					{
						_params.logger.log("Cannot read binary PLY file containing data types of unknown length %s", system::ILogger::ELL_WARNING, word);
						continueReading = false;
					}
					else
					{
						word = ctx.getNextWord();
						prop.list.itemType = prop.getType(word);
						if (ctx.IsBinaryFile && prop.list.itemType==EType::Unknown)
						{
							_params.logger.log("Cannot read binary PLY file containing data types of unknown length %s", system::ILogger::ELL_ERROR, word);
							continueReading = false;
						}
					}
				}
				else if (ctx.IsBinaryFile && prop.type==EType::Unknown)
				{
					_params.logger.log("Cannot read binary PLY file containing data types of unknown length %s", system::ILogger::ELL_ERROR, word);
					continueReading = false;
				}
				else
					el.KnownSize += prop.size();

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
			else if (strcmp(word, "ascii"))
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
	IGeometry<ICPUBuffer>::SDataView positions = {}, normals = {}, uvs = {}, colors = {};

	// loop through each of the elements
	for (uint32_t i=0; i<ctx.ElementList.size(); ++i)
	{
		auto& el = ctx.ElementList[i];
		if (el.Name=="vertex")
		{
			for (auto& vertexProperty : el.Properties)
			{
				const auto& propertyName = vertexProperty.Name;

				if (propertyName == "x" || propertyName == "y" || propertyName == "z")
				{
					if (!positions.src)
					{
						auto buffer = ICPUBuffer::create({asset::getTexelOrBlockBytesize(EF_R32G32B32_SFLOAT)*el.Count});
						positions.src = {.offset=0,.size=buffer->getSize(),.buffer=std::move(buffer)};
					}
				}
				else if(propertyName == "nx" || propertyName == "ny" || propertyName == "nz")
				{
					if (!normals.src)
					{
						auto buffer = ICPUBuffer::create({asset::getTexelOrBlockBytesize(EF_R32G32B32_SFLOAT)*el.Count});
						normals.src = {.offset=0,.size=buffer->getSize(),.buffer=std::move(buffer)};
					}
				}
				else if (propertyName == "u" || propertyName == "s" || propertyName == "v" || propertyName == "t")
				{
					if (!uvs.src)
					{
						auto buffer = ICPUBuffer::create({asset::getTexelOrBlockBytesize(EF_R32G32B32_SFLOAT)*el.Count});
						uvs.src = {.offset=0,.size=buffer->getSize(),.buffer=std::move(buffer)};
					}
				}
				else if (propertyName == "red" || propertyName == "green" || propertyName == "blue" || propertyName == "alpha")
				{
					if (!colors.src)
					{
						auto buffer = ICPUBuffer::create({asset::getTexelOrBlockBytesize(EF_R32G32B32_SFLOAT)*el.Count});
						uvs.src = {.offset=0,.size=buffer->getSize(),.buffer=std::move(buffer)};
					}
				}			
			}

			// loop through vertex properties
			for (size_t j=0; j<el.Count; ++j)
				hasNormals &= readVertex(ctx,el,attributes,j,_params);
		}
		else if (el.Name=="face")
		{
			for (size_t j=0; j<el.Count; ++j)
				readFace(ctx,el,indices);
		}
		else
		{
			// skip these elements
			for (size_t j=0; j<el.Count; ++j)
				skipElement(ctx,el);
		}
	}

    if (indices.size())
    {
		asset::SBufferBinding<ICPUBuffer> indexBinding = { 0, asset::ICPUBuffer::create({ indices.size() * sizeof(uint32_t) }) };
		memcpy(indexBinding.buffer->getPointer(), indices.data(), indexBinding.buffer->getSize());
				
		mb->setIndexCount(indices.size());
		mb->setIndexBufferBinding(std::move(indexBinding));
		mb->setIndexType(asset::EIT_32BIT);

		if (!genVertBuffersForMBuffer(mb.get(), attributes, ctx))
			return {};
    }
    else
    {
		mb->setIndexCount(attributes[ET_POS].buffer->getSize());
		mb->setIndexType(EIT_UNKNOWN);

		if (!genVertBuffersForMBuffer(mb.get(), attributes, ctx))
			return {};
    }

	if (positions)
		geometry->setPositionView(std::move(positions));
	if (normals)
		geometry->setNormalView(std::move(normals));
	auto* auxViews = geometry->getAuxAttributeViews();
	if (uvs)
		auxViews->push_back(std::move(uvs));
	if (colors)
		auxViews->push_back(std::move(colors));

	auto meta = core::make_smart_refctd_ptr<CPLYMetadata>();
	return SAssetBundle(std::move(meta),{ std::move(geometry) });
}

static void performActionBasedOnOrientationSystem(const asset::IAssetLoader::SAssetLoadParams& _params, std::function<void()> performOnRightHanded, std::function<void()> performOnLeftHanded)
{
	if (_params.loaderFlags & IAssetLoader::ELPF_RIGHT_HANDED_MESHES)
		performOnRightHanded();
	else
		performOnLeftHanded();
}
  


bool CPLYMeshFileLoader::readFace(SContext& _ctx, const SPLYElement& Element, core::vector<uint32_t>& _outIndices)
{
	if (!_ctx.IsBinaryFile)
		getNextLine(_ctx);

	for (uint32_t i = 0; i < Element.Properties.size(); ++i)
	{
		if ((Element.Properties[i].Name == "vertex_indices" ||
			Element.Properties[i].Name == "vertex_index") && Element.Properties[i].Type == EPLYPT_LIST)
		{
			int32_t count = _ctx.getInt( Element.Properties[i].Data.List.CountType);
			//_NBL_DEBUG_BREAK_IF(count != 3)

			uint32_t a = _ctx.getInt( Element.Properties[i].Data.List.ItemType),
				b = _ctx.getInt( Element.Properties[i].Data.List.ItemType),
				c = _ctx.getInt( Element.Properties[i].Data.List.ItemType);
			int32_t j = 3;

			_outIndices.push_back(a);
			_outIndices.push_back(b);
			_outIndices.push_back(c);

			for (; j < count; ++j)
			{
				b = c;
				c = _ctx.getInt( Element.Properties[i].Data.List.ItemType);
				_outIndices.push_back(a);
				_outIndices.push_back(c);
				_outIndices.push_back(b);
			}
		}
		else if (Element.Properties[i].Name == "intensity")
		{
			// todo: face intensity
			skipProperty(_ctx, Element.Properties[i]);
		}
		else
			skipProperty(_ctx, Element.Properties[i]);
	}
	return true;
}








bool CPLYMeshFileLoader::genVertBuffersForMBuffer(
	asset::ICPUMeshBuffer* _mbuf,
	const asset::SBufferBinding<asset::ICPUBuffer> attributes[4],
	SContext& context
) const
{
	core::vector<uint8_t> availableAttributes;
	for (auto i = 0; i < 4; ++i)
		if (attributes[i].buffer)
			availableAttributes.push_back(i);

	{
		size_t check = attributes[0].buffer->getSize();
		for (size_t i = 1u; i < 4u; ++i)
		{
			if (attributes[i].buffer && attributes[i].buffer->getSize() != check)
				return false;
			else if (attributes[i].buffer)
				check = attributes[i].buffer->getSize();
		}
	}

	auto getPipeline = [&]() -> core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>
	{
		constexpr std::array<uint8_t, 3> avaiableOptionsForShaders { ET_COL, ET_UV, ET_NORM };

		auto fetchPipelineFromCache = [&](CPLYMeshFileLoader::E_TYPE attribute)
		{
			const IAssetLoader::SAssetLoadContext fakeContext(IAssetLoader::SAssetLoadParams{}, nullptr);
			const std::string hash = getPipelineCacheKey(attribute, _mbuf->getIndexBufferBinding().buffer.get());

			const asset::IAsset::E_TYPE types[]{ asset::IAsset::ET_RENDERPASS_INDEPENDENT_PIPELINE, (asset::IAsset::E_TYPE)0u };
			auto pipelineBundle = context.loaderOverride->findCachedAsset(hash, types, fakeContext, context.topHierarchyLevel + ICPURenderpassIndependentPipeline::DESC_SET_HIERARCHYLEVELS_BELOW);
			{
				bool status = !pipelineBundle.getContents().empty();
				assert(status);
			}

			auto mbPipeline = core::smart_refctd_ptr_static_cast<asset::ICPURenderpassIndependentPipeline>(pipelineBundle.getContents().begin()[0]);

			return mbPipeline;
		};

		core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline> mbPipeline;
		{
			for (auto& anOption : avaiableOptionsForShaders)
			{
				auto found = std::find(availableAttributes.begin(), availableAttributes.end(), anOption);
				if (found != availableAttributes.end())
					mbPipeline = fetchPipelineFromCache(static_cast<E_TYPE>(anOption));
			}

			if(!mbPipeline)
				mbPipeline = fetchPipelineFromCache(ET_POS);
		}

		return mbPipeline;
	};

	auto mbPipeline = getPipeline();

	for (auto index = 0; index < 4; ++index)
	{
		auto attribute = attributes[index];
		if (attribute.buffer)
			_mbuf->setVertexBufferBinding(std::move(attribute), index);
	}
	
	_mbuf->setPipeline(std::move(mbPipeline));

    return true;
}






// read the next float from the file and move the start pointer along
float CPLYMeshFileLoader::getFloat(SContext& _ctx, E_PLY_PROPERTY_TYPE t)
{
	float retVal = 0.0f;

	if (_ctx.IsBinaryFile)
	{
		if (_ctx.EndPointer - _ctx.StartPointer < 8)
			fillBuffer(_ctx);

		if (_ctx.EndPointer - _ctx.StartPointer > 0)
		{
			switch (t)
			{
			case EPLYPT_INT8:
				retVal = *_ctx.StartPointer;
				_ctx.StartPointer++;
				break;
			case EPLYPT_INT16:
				if (_ctx.IsWrongEndian)
					retVal = core::Byteswap::byteswap(*(reinterpret_cast<int16_t*>(_ctx.StartPointer)));
				else
					retVal = *(reinterpret_cast<int16_t*>(_ctx.StartPointer));
				_ctx.StartPointer += 2;
				break;
			case EPLYPT_INT32:
				if (_ctx.IsWrongEndian)
					retVal = float(core::Byteswap::byteswap(*(reinterpret_cast<int32_t*>(_ctx.StartPointer))));
				else
					retVal = float(*(reinterpret_cast<int32_t*>(_ctx.StartPointer)));
				_ctx.StartPointer += 4;
				break;
			case EPLYPT_FLOAT32:
				if (_ctx.IsWrongEndian)
					retVal = core::Byteswap::byteswap(*(reinterpret_cast<float*>(_ctx.StartPointer)));
				else
					retVal = *(reinterpret_cast<float*>(_ctx.StartPointer));
				_ctx.StartPointer += 4;
				break;
			case EPLYPT_FLOAT64:
				char tmp[8];
				memcpy(tmp, _ctx.StartPointer, 8);
				if (_ctx.IsWrongEndian)
					for (size_t i = 0u; i < 4u; ++i)
						std::swap(tmp[i], tmp[7u - i]);
				retVal = float(*(reinterpret_cast<double*>(tmp)));
				_ctx.StartPointer += 8;
				break;
			case EPLYPT_LIST:
			case EPLYPT_UNKNOWN:
			default:
				retVal = 0.0f;
				_ctx.StartPointer++; // ouch!
			}
		}
		else
			retVal = 0.0f;
	}
	else
	{
		char* word = _ctx.getNextWord();
		switch (t)
		{
		case EPLYPT_INT8:
		case EPLYPT_INT16:
		case EPLYPT_INT32:
			retVal = float(atoi(word));
			break;
		case EPLYPT_FLOAT32:
		case EPLYPT_FLOAT64:
			retVal = float(atof(word));
			break;
		case EPLYPT_LIST:
		case EPLYPT_UNKNOWN:
		default:
			retVal = 0.0f;
		}
	}

	return retVal;
}


} // end namespace scene
} // end namespace nbl

#endif // _NBL_COMPILE_WITH_PLY_LOADER_

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
						retval = core::Byteswap::byteswap(retval);
					return retval;
				}
				case 4:
				{
					if (StartPointer+sizeof(int32_t)>EndPointer)
						break;
					auto retval = *(reinterpret_cast<int32_t*&>(StartPointer)++);
					if (IsWrongEndian)
						retval = core::Byteswap::byteswap(retval);
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
	template<typename T>
	void getData(T* dst, const E_FORMAT f)
	{
#if 0
		float retVal = 0.0f;

		assert(!isFloatingPointFormat(f));
		if (IsBinaryFile)
		{
			if (StartPointer+sizeof(hlsl::float64_t)>EndPointer)
				fillBuffer();

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
			return atof(getNextWord());
#endif
	}
	bool readVertex(const uint32_t& currentVertexIndex, const IAssetLoader::SAssetLoadParams& _params)
	{
#if 0
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
#endif
		return false;
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
				for (auto j=0u; j<count; ++j)
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
	ICPUPolygonGeometry::SDataView posView = {}, normalView = {}, uvView = {}, colorView = {};
	core::vector<uint8_t> positions = {}, normals = {}, uvs = {}, colors = {};

	// loop through each of the elements
	for (uint32_t i=0; i<ctx.ElementList.size(); ++i)
	{
		auto& el = ctx.ElementList[i];
		if (el.Name=="vertex") // TODO: are multiple of these possible in a file? do we create a geometry collection then?
		{
#if 0
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
#endif
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

	auto initView = [](ICPUPolygonGeometry::SDataView& view, core::vector<uint8_t>& data)->bool
	{
		if (data.empty())
			return false;
		view.composed.stride = getTexelOrBlockBytesize(view.composed.format);
		using aabb_format_e = IGeometryBase::EAABBFormat;
		view.composed.rangeFormat = IGeometryBase::getMatchingAABBFormat(view.composed.format);
		view.src.offset = 0;
		view.src.size = data.size();
		// TODO: use a polymoprhic memory resource so the vector can be adopted
		view.src.buffer = ICPUBuffer::create({{view.src.size},data.data()});
		return static_cast<bool>(view.src);
	};
	if (initView(posView,positions))
		geometry->setPositionView(std::move(posView));
	if (initView(normalView,normals))
		geometry->setNormalView(std::move(normalView));
	auto* auxViews = geometry->getAuxAttributeViews();
	if (initView(uvView,uvs))
		auxViews->push_back(std::move(uvView));
	if (initView(colorView,colors))
		auxViews->push_back(std::move(colorView));

	auto meta = core::make_smart_refctd_ptr<CPLYMetadata>();
	return SAssetBundle(std::move(meta),{ std::move(geometry) });
}

// TODO: move to IGeometryLoader
static void performActionBasedOnOrientationSystem(const asset::IAssetLoader::SAssetLoadParams& _params, std::function<void()> performOnRightHanded, std::function<void()> performOnLeftHanded)
{
	if (_params.loaderFlags & IAssetLoader::ELPF_RIGHT_HANDED_MESHES)
		performOnRightHanded();
	else
		performOnLeftHanded();
}


} // end namespace nbl::asset
#endif // _NBL_COMPILE_WITH_PLY_LOADER_

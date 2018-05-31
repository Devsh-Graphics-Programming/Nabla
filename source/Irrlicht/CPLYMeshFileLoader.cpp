// Copyright (C) 2009-2012 Gaz Davidson
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"
#ifdef _IRR_COMPILE_WITH_PLY_LOADER_

#include <numeric>

#include "CPLYMeshFileLoader.h"
#include "IMeshManipulator.h"
#include "SMesh.h"
#include "SAnimatedMesh.h"
#include "IReadFile.h"
#include "os.h"

namespace irr
{
namespace scene
{

// input buffer must be at least twice as long as the longest line in the file
#define PLY_INPUT_BUFFER_SIZE 51200 // file is loaded in 50k chunks


// constructor
CPLYMeshFileLoader::CPLYMeshFileLoader(scene::ISceneManager* smgr)
: SceneManager(smgr), File(0), Buffer(0)
{
}


CPLYMeshFileLoader::~CPLYMeshFileLoader()
{
	// delete the buffer in case we didn't earlier
	// (we do, but this could be disabled to increase the speed of loading hundreds of meshes)
	if (Buffer)
	{
		delete [] Buffer;
		Buffer = 0;
	}

	// Destroy the element list if it exists
	for (uint32_t i=0; i<ElementList.size(); ++i)
		delete ElementList[i];
	ElementList.clear();
}


//! returns true if the file maybe is able to be loaded by this class
bool CPLYMeshFileLoader::isALoadableFileExtension(const io::path& filename) const
{
	return core::hasFileExtension(filename, "ply");
}


//! creates/loads an animated mesh from the file.
ICPUMesh* CPLYMeshFileLoader::createMesh(io::IReadFile* file)
{
	if (!file)
		return 0;

	File = file;
	File->grab();

	// attempt to allocate the buffer and fill with data
	if (!allocateBuffer())
	{
		File->drop();
		File = 0;
		return 0;
	}

	// start with empty mesh
    SCPUMesh* mesh = nullptr;
	uint32_t vertCount=0;

	// Currently only supports ASCII meshes
	if (strcmp(getNextLine(), "ply"))
	{
		os::Printer::log("Not a valid PLY file", file->getFileName().c_str(), ELL_ERROR);
	}
	else
	{
		// cut the next line out
		getNextLine();
		// grab the word from this line
		char *word = getNextWord();

		// ignore comments
		while (strcmp(word, "comment") == 0)
		{
			getNextLine();
			word = getNextWord();
		}

		bool readingHeader = true;
		bool continueReading = true;
		IsBinaryFile = false;
		IsWrongEndian= false;

		do
		{
			if (strcmp(word, "format") == 0)
			{
				word = getNextWord();

				if (strcmp(word, "binary_little_endian") == 0)
                {
					IsBinaryFile = true;
				}
				else if (strcmp(word, "binary_big_endian") == 0)
				{
					IsBinaryFile = true;
					IsWrongEndian = true;
				}
				else if (strcmp(word, "ascii"))
				{
					// abort if this isn't an ascii or a binary mesh
					os::Printer::log("Unsupported PLY mesh format", word, ELL_ERROR);
					continueReading = false;
				}

				if (continueReading)
				{
					word = getNextWord();
					if (strcmp(word, "1.0"))
					{
						os::Printer::log("Unsupported PLY mesh version", word, ELL_WARNING);
					}
				}
			}
			else if (strcmp(word, "property") == 0)
			{
				word = getNextWord();

				if (!ElementList.size())
				{
					os::Printer::log("PLY property found before element", word, ELL_WARNING);
				}
				else
				{
					// get element
					SPLYElement* el = ElementList[ElementList.size()-1];

					// fill property struct
					SPLYProperty prop;
					prop.Type = getPropertyType(word);
					el->KnownSize += prop.size();

					if (prop.Type == EPLYPT_LIST)
					{
						el->IsFixedWidth = false;

						word = getNextWord();

						prop.Data.List.CountType = getPropertyType(word);
						if (IsBinaryFile && prop.Data.List.CountType == EPLYPT_UNKNOWN)
						{
							os::Printer::log("Cannot read binary PLY file containing data types of unknown length", word, ELL_ERROR);
							continueReading = false;
						}
						else
						{
							word = getNextWord();
							prop.Data.List.ItemType = getPropertyType(word);
							if (IsBinaryFile && prop.Data.List.ItemType == EPLYPT_UNKNOWN)
							{
								os::Printer::log("Cannot read binary PLY file containing data types of unknown length", word, ELL_ERROR);
								continueReading = false;
							}
						}
					}
					else if (IsBinaryFile && prop.Type == EPLYPT_UNKNOWN)
					{
						os::Printer::log("Cannot read binary PLY file containing data types of unknown length", word, ELL_ERROR);
						continueReading = false;
					}

					prop.Name = getNextWord();

					// add property to element
					el->Properties.push_back(prop);
				}
			}
			else if (strcmp(word, "element") == 0)
			{
				SPLYElement* el = new SPLYElement;
				el->Name = getNextWord();
				el->Count = atoi(getNextWord());
				el->IsFixedWidth = true;
				el->KnownSize = 0;
				ElementList.push_back(el);

				if (el->Name == "vertex")
					vertCount = el->Count;

			}
			else if (strcmp(word, "end_header") == 0)
			{
				readingHeader = false;
				if (IsBinaryFile)
				{
					StartPointer = LineEndPointer + 1;
				}
			}
			else if (strcmp(word, "comment") == 0)
			{
				// ignore line
			}
			else
			{
				os::Printer::log("Unknown item in PLY file", word, ELL_WARNING);
			}

			if (readingHeader && continueReading)
			{
				getNextLine();
				word = getNextWord();
			}
		}
		while (readingHeader && continueReading);

		// now to read the actual data from the file
		if (continueReading)
		{
			// create a mesh buffer
            ICPUMeshBuffer *mb = new ICPUMeshBuffer();
            auto desc = new ICPUMeshDataFormatDesc();
            mb->setMeshDataAndFormat(desc);
            desc->drop();

            std::vector<core::vectorSIMDf> attribs[4];
            std::vector<uint32_t> indices;

			bool hasNormals=true;
			// loop through each of the elements
			for (uint32_t i=0; i<ElementList.size(); ++i)
			{
				// do we want this element type?
				if (ElementList[i]->Name == "vertex")
				{
					// loop through vertex properties
					for (uint32_t j=0; j < ElementList[i]->Count; ++j)
						hasNormals &= readVertex(*ElementList[i], attribs);
				}
				else if (ElementList[i]->Name == "face")
				{
					// read faces
					for (uint32_t j=0; j < ElementList[i]->Count; ++j)
						readFace(*ElementList[i], indices);
				}
				else
				{
					// skip these elements
					for (uint32_t j=0; j < ElementList[i]->Count; ++j)
						skipElement(*ElementList[i]);
				}
			}

            if (!genVertBuffersForMBuffer(mb, attribs))
            {
                mb->drop();
                delete [] Buffer;
                Buffer = nullptr;
                File->drop();
                File = nullptr;
                return nullptr;
            }
            if (indices.size())
            {
                core::ICPUBuffer* idxBuf = new core::ICPUBuffer(4 * indices.size());
                memcpy(idxBuf->getPointer(), indices.data(), idxBuf->getSize());
                desc->mapIndexBuffer(idxBuf);
                idxBuf->drop();
                mb->setIndexCount(indices.size());
                mb->setIndexType(video::EIT_32BIT);
                mb->setPrimitiveType(EPT_TRIANGLES);
            }
            else
            {
                mb->setPrimitiveType(EPT_POINTS);
                mb->setIndexCount(attribs[E_POS].size());
                //mb->getMaterial().setFlag(video::EMF_POINTCLOUD, true);
            }

            mesh = new SCPUMesh();

			mb->recalculateBoundingBox();
			//if (!hasNormals)
			//	SceneManager->getMeshManipulator()->recalculateNormals(mb);
			mesh->addMeshBuffer(mb);
			mesh->recalculateBoundingBox();
			mb->drop();
		}
	}


	// free the buffer
	delete [] Buffer;
	Buffer = nullptr;
	File->drop();
    File = nullptr;

	// if we managed to create a mesh, return it
	return mesh;
}


bool CPLYMeshFileLoader::readVertex(const SPLYElement &Element, std::vector<core::vectorSIMDf> _outAttribs[4])
{
	if (!IsBinaryFile)
		getNextLine();

    std::pair<bool, core::vectorSIMDf> attribs[4];
    attribs[E_COL].second.W = 1.f;
    attribs[E_NORM].second.Y = 1.f;

	bool result=false;
	for (uint32_t i=0; i < Element.Properties.size(); ++i)
	{
		E_PLY_PROPERTY_TYPE t = Element.Properties[i].Type;

        if (Element.Properties[i].Name == "x")
        {
            attribs[E_POS].second.X = getFloat(t);
            attribs[E_POS].first = true;
        }
        else if (Element.Properties[i].Name == "y")
        {
            attribs[E_POS].second.Y = getFloat(t);
            attribs[E_POS].first = true;
        }
        else if (Element.Properties[i].Name == "z")
        {
            attribs[E_POS].second.Z = getFloat(t);
            attribs[E_POS].first = true;
        }
		else if (Element.Properties[i].Name == "nx")
		{
			attribs[E_NORM].second.X = getFloat(t);
			attribs[E_NORM].first = result=true;
		}
		else if (Element.Properties[i].Name == "ny")
		{
			attribs[E_NORM].second.Y = getFloat(t);
            attribs[E_NORM].first = result=true;
		}
		else if (Element.Properties[i].Name == "nz")
		{
			attribs[E_NORM].second.Z = getFloat(t);
            attribs[E_NORM].first = result=true;
		}
        // there isn't a single convention for the UV, some softwares like Blender or Assimp use "st" instead of "uv"
        else if (Element.Properties[i].Name == "u" || Element.Properties[i].Name == "s")
        {
            attribs[E_UV].second.X = getFloat(t);
            attribs[E_UV].first = true;
        }
        else if (Element.Properties[i].Name == "v" || Element.Properties[i].Name == "t")
        {
            attribs[E_UV].second.Y = getFloat(t);
            attribs[E_UV].first = true;
        }
		else if (Element.Properties[i].Name == "red")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(t) : float(getInt(t))/255.f;
			attribs[E_COL].second.X = value;
            attribs[E_COL].first = true;
		}
		else if (Element.Properties[i].Name == "green")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(t) : float(getInt(t))/255.f;
			attribs[E_COL].second.Y = value;
            attribs[E_COL].first = true;
		}
		else if (Element.Properties[i].Name == "blue")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(t) : float(getInt(t))/255.f;
			attribs[E_COL].second.Z = value;
            attribs[E_COL].first = true;
		}
		else if (Element.Properties[i].Name == "alpha")
		{
			float value = Element.Properties[i].isFloat() ? getFloat(t) : float(getInt(t))/255.f;
			attribs[E_COL].second.W = value;
            attribs[E_COL].first = true;
		}
		else
			skipProperty(Element.Properties[i]);
	}

    for(size_t i = 0u; i < 4u; ++i)
        if (attribs[i].first)
            _outAttribs[i].push_back(attribs[i].second);

	return result;
}


bool CPLYMeshFileLoader::readFace(const SPLYElement &Element, std::vector<uint32_t>& _outIndices)
{
	if (!IsBinaryFile)
		getNextLine();

	for (uint32_t i=0; i < Element.Properties.size(); ++i)
	{
		if ( (Element.Properties[i].Name == "vertex_indices" ||
			Element.Properties[i].Name == "vertex_index") && Element.Properties[i].Type == EPLYPT_LIST)
		{
			int32_t count = getInt(Element.Properties[i].Data.List.CountType);
            //_IRR_DEBUG_BREAK_IF(count != 3)

			uint32_t a = getInt(Element.Properties[i].Data.List.ItemType),
				b = getInt(Element.Properties[i].Data.List.ItemType),
				c = getInt(Element.Properties[i].Data.List.ItemType);
			int32_t j = 3;

			_outIndices.push_back(a);
			_outIndices.push_back(b);
			_outIndices.push_back(c);

			for (; j < count; ++j)
			{
				b = c;
				c = getInt(Element.Properties[i].Data.List.ItemType);
				_outIndices.push_back(a);
				_outIndices.push_back(c);
				_outIndices.push_back(b);
			}
		}
		else if (Element.Properties[i].Name == "intensity")
		{
			// todo: face intensity
			skipProperty(Element.Properties[i]);
		}
		else
			skipProperty(Element.Properties[i]);
	}
	return true;
}


// skips an element and all properties. return false on EOF
void CPLYMeshFileLoader::skipElement(const SPLYElement &Element)
{
	if (IsBinaryFile)
		if (Element.IsFixedWidth)
			moveForward(Element.KnownSize);
		else
			for (uint32_t i=0; i < Element.Properties.size(); ++i)
				skipProperty(Element.Properties[i]);
	else
		getNextLine();
}


void CPLYMeshFileLoader::skipProperty(const SPLYProperty &Property)
{
	if (Property.Type == EPLYPT_LIST)
	{
		int32_t count = getInt(Property.Data.List.CountType);

		for (int32_t i=0; i < count; ++i)
			getInt(Property.Data.List.CountType);
	}
	else
	{
		if (IsBinaryFile)
			moveForward(Property.size());
		else
			getNextWord();
	}
}


bool CPLYMeshFileLoader::allocateBuffer()
{
	// Destroy the element list if it exists
	for (uint32_t i=0; i<ElementList.size(); ++i)
		delete ElementList[i];
	ElementList.clear();

	if (!Buffer)
		Buffer = new char[PLY_INPUT_BUFFER_SIZE];

	// not enough memory?
	if (!Buffer)
		return false;

	// blank memory
	memset(Buffer, 0, PLY_INPUT_BUFFER_SIZE);

	StartPointer = Buffer;
	EndPointer = Buffer;
	LineEndPointer = Buffer-1;
	WordLength = -1;
	EndOfFile = false;

	// get data from the file
	fillBuffer();

	return true;
}


// gets more data from the file. returns false on EOF
void CPLYMeshFileLoader::fillBuffer()
{
	if (EndOfFile)
		return;

	uint32_t length = (uint32_t)(EndPointer - StartPointer);
	if (length && StartPointer != Buffer)
	{
		// copy the remaining data to the start of the buffer
		memcpy(Buffer, StartPointer, length);
	}
	// reset start position
	StartPointer = Buffer;
	EndPointer = StartPointer + length;

	if (File->getPos() == File->getSize())
	{
		EndOfFile = true;
	}
	else
	{
		// read data from the file
		uint32_t count = File->read(EndPointer, PLY_INPUT_BUFFER_SIZE - length);

		// increment the end pointer by the number of bytes read
		EndPointer = EndPointer + count;

		// if we didn't completely fill the buffer
		if (count != PLY_INPUT_BUFFER_SIZE - length)
		{
			// blank the rest of the memory
			memset(EndPointer, 0, Buffer + PLY_INPUT_BUFFER_SIZE - EndPointer);

			// end of file
			EndOfFile = true;
		}
	}
}


// skips x bytes in the file, getting more data if required
void CPLYMeshFileLoader::moveForward(uint32_t bytes)
{
	if (StartPointer + bytes >= EndPointer)
		fillBuffer();
	if (StartPointer + bytes < EndPointer)
		StartPointer += bytes;
	else
		StartPointer = EndPointer;
}

bool CPLYMeshFileLoader::genVertBuffersForMBuffer(ICPUMeshBuffer* _mbuf, const std::vector<core::vectorSIMDf> _attribs[4]) const
{
    {
    size_t check = _attribs[0].size();
    for (size_t i = 1u; i < 4u; ++i)
    {
        if (_attribs[i].size() != 0u && _attribs[i].size() != check)
            return false;
        else if (_attribs[i].size() != 0u)
            check = _attribs[i].size();
    }
    }
    auto putAttr = [&_attribs](ICPUMeshBuffer* _buf, size_t _attr, E_VERTEX_ATTRIBUTE_ID _vaid)
    {
        size_t i = 0u;
        for (const core::vectorSIMDf& v : _attribs[_attr])
            _buf->setAttribute(v, _vaid, i++);
    };

    size_t sizes[4];
    sizes[E_POS] = !_attribs[E_POS].empty() * 3 * sizeof(float);
    sizes[E_COL] = !_attribs[E_COL].empty() * 4 * sizeof(float);
    sizes[E_UV] = !_attribs[E_UV].empty() * 2 * sizeof(float);
    sizes[E_NORM] = !_attribs[E_NORM].empty() * 3 * sizeof(float);

    size_t offsets[4]{ 0u };
    for (size_t i = 1u; i < 4u; ++i)
        offsets[i] = offsets[i-1] + sizes[i-1];

    const size_t stride = std::accumulate(sizes, sizes+4, 0u);

    core::ICPUBuffer* buf = new core::ICPUBuffer(_attribs[E_POS].size() * stride);

    auto desc = _mbuf->getMeshDataAndFormat();
    if (sizes[E_POS])
        desc->mapVertexAttrBuffer(buf, EVAI_ATTR0, ECPA_THREE, ECT_FLOAT, stride, offsets[E_POS]);
    if (sizes[E_COL])
        desc->mapVertexAttrBuffer(buf, EVAI_ATTR1, ECPA_FOUR, ECT_FLOAT, stride, offsets[E_COL]);
    if (sizes[E_UV])
        desc->mapVertexAttrBuffer(buf, EVAI_ATTR2, ECPA_TWO, ECT_FLOAT, stride, offsets[E_UV]);
    if (sizes[E_NORM])
        desc->mapVertexAttrBuffer(buf, EVAI_ATTR3, ECPA_THREE, ECT_FLOAT, stride, offsets[E_NORM]);
    buf->drop();

    E_VERTEX_ATTRIBUTE_ID vaids[4];
    vaids[E_POS] = EVAI_ATTR0;
    vaids[E_COL] = EVAI_ATTR1;
    vaids[E_UV] = EVAI_ATTR2;
    vaids[E_NORM] = EVAI_ATTR3;

    for (size_t i = 0u; i < 4u; ++i)
    {
        if (sizes[i])
            putAttr(_mbuf, i, vaids[i]);
    }

    float d[100];
    memcpy(d, buf->getPointer(), 400);

    return true;
}


E_PLY_PROPERTY_TYPE CPLYMeshFileLoader::getPropertyType(const char* typeString) const
{
	if (strcmp(typeString, "char") == 0 ||
		strcmp(typeString, "uchar") == 0 ||
		strcmp(typeString, "int8") == 0 ||
		strcmp(typeString, "uint8") == 0)
	{
		return EPLYPT_INT8;
	}
	else if (strcmp(typeString, "uint") == 0 ||
		strcmp(typeString, "int16") == 0 ||
		strcmp(typeString, "uint16") == 0 ||
		strcmp(typeString, "short") == 0 ||
		strcmp(typeString, "ushort") == 0)
	{
		return EPLYPT_INT16;
	}
	else if (strcmp(typeString, "int") == 0 ||
		strcmp(typeString, "long") == 0 ||
		strcmp(typeString, "ulong") == 0 ||
		strcmp(typeString, "int32") == 0 ||
		strcmp(typeString, "uint32") == 0)
	{
		return EPLYPT_INT32;
	}
	else if (strcmp(typeString, "float") == 0 ||
		strcmp(typeString, "float32") == 0)
	{
		return EPLYPT_FLOAT32;
	}
	else if (strcmp(typeString, "float64") == 0 ||
		strcmp(typeString, "double") == 0)
	{
		return EPLYPT_FLOAT64;
	}
	else if ( strcmp(typeString, "list") == 0 )
	{
		return EPLYPT_LIST;
	}
	else
	{
		// unsupported type.
		// cannot be loaded in binary mode
		return EPLYPT_UNKNOWN;
	}
}


// Split the string data into a line in place by terminating it instead of copying.
char* CPLYMeshFileLoader::getNextLine()
{
	// move the start pointer along
	StartPointer = LineEndPointer + 1;

	// crlf split across buffer move
	if (*StartPointer == '\n')
	{
		*StartPointer = '\0';
		++StartPointer;
	}

	// begin at the start of the next line
	char* pos = StartPointer;
	while (pos < EndPointer && *pos && *pos != '\r' && *pos != '\n')
		++pos;

	if ( pos < EndPointer && ( *(pos+1) == '\r' || *(pos+1) == '\n') )
	{
		*pos = '\0';
		++pos;
	}

	// we have reached the end of the buffer
	if (pos >= EndPointer)
	{
		// get data from the file
		if (!EndOfFile)
		{
			fillBuffer();
			// reset line end pointer
			LineEndPointer = StartPointer - 1;

			if (StartPointer != EndPointer)
				return getNextLine();
			else
				return Buffer;
		}
		else
		{
			// EOF
			StartPointer = EndPointer-1;
			*StartPointer = '\0';
			return StartPointer;
		}
	}
	else
	{
		// null terminate the string in place
		*pos = '\0';
		LineEndPointer = pos;
		WordLength = -1;
		// return pointer to the start of the line
		return StartPointer;
	}
}


// null terminate the next word on the previous line and move the next word pointer along
// since we already have a full line in the buffer, we never need to retrieve more data
char* CPLYMeshFileLoader::getNextWord()
{
	// move the start pointer along
	StartPointer += WordLength + 1;
    if (!*StartPointer)
        getNextLine();

	if (StartPointer == LineEndPointer)
	{
		WordLength = -1; //
		return LineEndPointer;
	}
	// begin at the start of the next word
	char* pos = StartPointer;
	while (*pos && pos < LineEndPointer && pos < EndPointer && *pos != ' ' && *pos != '\t')
		++pos;

	while(*pos && pos < LineEndPointer && pos < EndPointer && (*pos == ' ' || *pos == '\t') )
	{
		// null terminate the string in place
		*pos = '\0';
		++pos;
	}
	--pos;
	WordLength = (int32_t)(pos-StartPointer);
	// return pointer to the start of the word
	return StartPointer;
}


// read the next float from the file and move the start pointer along
float CPLYMeshFileLoader::getFloat(E_PLY_PROPERTY_TYPE t)
{
	float retVal = 0.0f;

	if (IsBinaryFile)
	{
		if (EndPointer - StartPointer < 8)
			fillBuffer();

		if (EndPointer - StartPointer > 0)
		{
			switch (t)
			{
			case EPLYPT_INT8:
				retVal = *StartPointer;
				StartPointer++;
				break;
			case EPLYPT_INT16:
				if (IsWrongEndian)
					retVal = os::Byteswap::byteswap(*(reinterpret_cast<int16_t*>(StartPointer)));
				else
					retVal = *(reinterpret_cast<int16_t*>(StartPointer));
				StartPointer += 2;
				break;
			case EPLYPT_INT32:
				if (IsWrongEndian)
					retVal = float(os::Byteswap::byteswap(*(reinterpret_cast<int32_t*>(StartPointer))));
				else
					retVal = float(*(reinterpret_cast<int32_t*>(StartPointer)));
				StartPointer += 4;
				break;
			case EPLYPT_FLOAT32:
				if (IsWrongEndian)
					retVal = os::Byteswap::byteswap(*(reinterpret_cast<float*>(StartPointer)));
				else
					retVal = *(reinterpret_cast<float*>(StartPointer));
				StartPointer += 4;
				break;
			case EPLYPT_FLOAT64:
                char tmp[8];
                memcpy(tmp, StartPointer, 8);
                if (IsWrongEndian)
                    for (size_t i = 0u; i < 4u; ++i)
                        std::swap(tmp[i], tmp[7u-i]);
				retVal = float(*(reinterpret_cast<double*>(tmp)));
				StartPointer += 8;
				break;
			case EPLYPT_LIST:
			case EPLYPT_UNKNOWN:
			default:
				retVal = 0.0f;
				StartPointer++; // ouch!
			}
		}
		else
			retVal = 0.0f;
	}
	else
	{
		char* word = getNextWord();
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


// read the next int from the file and move the start pointer along
uint32_t CPLYMeshFileLoader::getInt(E_PLY_PROPERTY_TYPE t)
{
	uint32_t retVal = 0;

	if (IsBinaryFile)
	{
		if (!EndOfFile && EndPointer - StartPointer < 8)
			fillBuffer();

		if (EndPointer - StartPointer)
		{
			switch (t)
			{
			case EPLYPT_INT8:
				retVal = *StartPointer;
				StartPointer++;
				break;
			case EPLYPT_INT16:
				if (IsWrongEndian)
					retVal = os::Byteswap::byteswap(*(reinterpret_cast<uint16_t*>(StartPointer)));
				else
					retVal = *(reinterpret_cast<uint16_t*>(StartPointer));
				StartPointer += 2;
				break;
			case EPLYPT_INT32:
				if (IsWrongEndian)
					retVal = os::Byteswap::byteswap(*(reinterpret_cast<int32_t*>(StartPointer)));
				else
					retVal = *(reinterpret_cast<int32_t*>(StartPointer));
				StartPointer += 4;
				break;
			case EPLYPT_FLOAT32:
				if (IsWrongEndian)
					retVal = (uint32_t)os::Byteswap::byteswap(*(reinterpret_cast<float*>(StartPointer)));
				else
					retVal = (uint32_t)(*(reinterpret_cast<float*>(StartPointer)));
				StartPointer += 4;
				break;
			case EPLYPT_FLOAT64:
				// todo: byteswap 64-bit
				retVal = (uint32_t)(*(reinterpret_cast<double*>(StartPointer)));
				StartPointer += 8;
				break;
			case EPLYPT_LIST:
			case EPLYPT_UNKNOWN:
			default:
				retVal = 0;
				StartPointer++; // ouch!
			}
		}
		else
			retVal = 0;
	}
	else
	{
		char* word = getNextWord();
		switch (t)
		{
		case EPLYPT_INT8:
		case EPLYPT_INT16:
		case EPLYPT_INT32:
			retVal = atoi(word);
			break;
		case EPLYPT_FLOAT32:
		case EPLYPT_FLOAT64:
			retVal = uint32_t(atof(word));
			break;
		case EPLYPT_LIST:
		case EPLYPT_UNKNOWN:
		default:
			retVal = 0;
		}
	}
	return retVal;
}


} // end namespace scene
} // end namespace irr

#endif // _IRR_COMPILE_WITH_PLY_LOADER_


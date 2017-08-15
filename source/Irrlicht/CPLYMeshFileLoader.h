// Copyright (C) 2009-2012 Gaz Davidson
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_PLY_MESH_FILE_LOADER_H_INCLUDED__
#define __C_PLY_MESH_FILE_LOADER_H_INCLUDED__

#include "IMeshLoader.h"
#include "ISceneManager.h"

namespace irr
{
namespace scene
{

enum E_PLY_PROPERTY_TYPE
{
	EPLYPT_INT8  = 0,
	EPLYPT_INT16,
	EPLYPT_INT32,
	EPLYPT_FLOAT32,
	EPLYPT_FLOAT64,
	EPLYPT_LIST,
	EPLYPT_UNKNOWN
};

//! Meshloader capable of loading obj meshes.
class CPLYMeshFileLoader : public IMeshLoader
{
public:

	//! Constructor
	CPLYMeshFileLoader(scene::ISceneManager* smgr);

	//! Destructor
	virtual ~CPLYMeshFileLoader();

	//! returns true if the file maybe is able to be loaded by this class
	//! based on the file extension (e.g. ".ply")
	virtual bool isALoadableFileExtension(const io::path& filename) const;

	//! creates/loads an animated mesh from the file.
	virtual ICPUMesh* createMesh(io::IReadFile* file);

private:

	struct SPLYProperty
	{
		core::stringc Name;
		E_PLY_PROPERTY_TYPE Type;
		union
		{
			uint8_t  Int8;
			uint16_t Int16;
			uint32_t Int32;
			float Float32;
			double Double;
			struct SPLYListProperty
			{
				E_PLY_PROPERTY_TYPE CountType;
				E_PLY_PROPERTY_TYPE ItemType;
			} List;

		} Data;

		inline uint32_t size() const
		{
			switch(Type)
			{
			case EPLYPT_INT8:
				return 1;
			case EPLYPT_INT16:
				return 2;
			case EPLYPT_INT32:
			case EPLYPT_FLOAT32:
				return 4;
			case EPLYPT_FLOAT64:
				return 8;
			case EPLYPT_LIST:
			case EPLYPT_UNKNOWN:
			default:
				return 0;
			}
		}

		inline bool isFloat() const
		{
			switch(Type)
			{
			case EPLYPT_FLOAT32:
			case EPLYPT_FLOAT64:
				return true;
			case EPLYPT_INT8:
			case EPLYPT_INT16:
			case EPLYPT_INT32:
			case EPLYPT_LIST:
			case EPLYPT_UNKNOWN:
			default:
				return false;
			}
		}
	};

	struct SPLYElement
	{
		// name of the element. We only want "vertex" and "face" elements
		// but we have to parse the others anyway.
		core::stringc Name;
		// The number of elements in the file
		uint32_t Count;
		// Properties of this element
		core::array<SPLYProperty> Properties;
		// in binary files, true if this is a fixed size
		bool IsFixedWidth;
		// known size in bytes, 0 if unknown
		uint32_t KnownSize;
	};

	bool allocateBuffer();
	int8_t* getNextLine();
	int8_t* getNextWord();
	void fillBuffer();
	E_PLY_PROPERTY_TYPE getPropertyType(const int8_t* typeString) const;

	bool readVertex(const SPLYElement &Element, scene::CDynamicMeshBuffer* mb);
	bool readFace(const SPLYElement &Element, scene::CDynamicMeshBuffer* mb);
	void skipElement(const SPLYElement &Element);
	void skipProperty(const SPLYProperty &Property);
	float getFloat(E_PLY_PROPERTY_TYPE t);
	uint32_t getInt(E_PLY_PROPERTY_TYPE t);
	void moveForward(uint32_t bytes);

	core::array<SPLYElement*> ElementList;

	scene::ISceneManager* SceneManager;
	io::IReadFile *File;
	int8_t *Buffer;
	bool IsBinaryFile, IsWrongEndian, EndOfFile;
	int32_t LineLength, WordLength;
	int8_t *StartPointer, *EndPointer, *LineEndPointer;
};

} // end namespace scene
} // end namespace irr

#endif


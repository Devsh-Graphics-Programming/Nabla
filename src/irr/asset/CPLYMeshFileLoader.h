// Copyright (C) 2009-2012 Gaz Davidson
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_PLY_MESH_FILE_LOADER_H_INCLUDED__
#define __C_PLY_MESH_FILE_LOADER_H_INCLUDED__

#include "ISceneManager.h"
#include "irr/asset/IAssetLoader.h"

namespace irr
{
namespace asset
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
class CPLYMeshFileLoader : public asset::IAssetLoader
{
protected:
	//! Destructor
	virtual ~CPLYMeshFileLoader();

public:
	//! Constructor
	CPLYMeshFileLoader(scene::ISceneManager* smgr);

    virtual bool isALoadableFileFormat(io::IReadFile* _file) const override;

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{ "ply", nullptr };
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

	//! creates/loads an animated mesh from the file.
    virtual asset::IAsset* loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

private:

	struct SPLYProperty
	{
		core::stringc Name;
		E_PLY_PROPERTY_TYPE Type;
		#include "irr/irrpack.h"
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
			} List PACK_STRUCT;

		} Data PACK_STRUCT;
		#include "irr/irrunpack.h"

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
		core::vector<SPLYProperty> Properties;
		// in binary files, true if this is a fixed size
		bool IsFixedWidth;
		// known size in bytes, 0 if unknown
		uint32_t KnownSize;
	};

    struct SContext
    {
        core::vector<SPLYElement*> ElementList;

        io::IReadFile *File = nullptr;
        char *Buffer = nullptr;
        bool IsBinaryFile = false, IsWrongEndian = false, EndOfFile = false;
        int32_t LineLength = 0, WordLength = 0;
        char *StartPointer = nullptr, *EndPointer = nullptr, *LineEndPointer = nullptr;

        ~SContext()
        {
            if (File)
                File->drop();
            File = nullptr;
            if (Buffer)
            {
                delete[] Buffer;
                Buffer = nullptr;
            }
            for (auto& e : ElementList)
                if (e) delete e;
            ElementList.clear();
        }
    };

    enum { E_POS = 0, E_UV = 2, E_NORM = 3, E_COL = 1 };

	bool allocateBuffer(SContext& _ctx);
	char* getNextLine(SContext& _ctx);
	char* getNextWord(SContext& _ctx);
	void fillBuffer(SContext& _ctx);
	E_PLY_PROPERTY_TYPE getPropertyType(const char* typeString) const;

	bool readVertex(SContext& _ctx, const SPLYElement &Element, core::vector<core::vectorSIMDf> _attribs[4]);
	bool readFace(SContext& _ctx, const SPLYElement &Element, core::vector<uint32_t>& _outIndices);
	void skipElement(SContext& _ctx, const SPLYElement &Element);
	void skipProperty(SContext& _ctx, const SPLYProperty &Property);
	float getFloat(SContext& _ctx, E_PLY_PROPERTY_TYPE t);
	uint32_t getInt(SContext& _ctx, E_PLY_PROPERTY_TYPE t);
	void moveForward(SContext& _ctx, uint32_t bytes);

    bool genVertBuffersForMBuffer(asset::ICPUMeshBuffer* _mbuf, const core::vector<core::vectorSIMDf> _attribs[4]) const;

    scene::ISceneManager* SceneManager;
};

} // end namespace scene
} // end namespace irr

#endif


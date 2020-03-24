// Copyright (C) 2009-2012 Gaz Davidson
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_PLY_MESH_FILE_LOADER_H_INCLUDED__
#define __C_PLY_MESH_FILE_LOADER_H_INCLUDED__

#include "irr/asset/IAssetLoader.h"
#include "irr/asset/ICPUMeshBuffer.h"
#include "irr/asset/CPLYPipelineMetadata.h"

namespace irr
{
namespace asset
{

// input buffer must be at least twice as long as the longest line in the file
#define PLY_INPUT_BUFFER_SIZE 51200 // file is loaded in 50k chunks

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
class CPLYMeshFileLoader : public IAssetLoader
{
protected:
	//! Destructor
	virtual ~CPLYMeshFileLoader();

public:
	//! Constructor
	CPLYMeshFileLoader(IAssetManager* _am);

    virtual bool isALoadableFileFormat(io::IReadFile* _file) const override;

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{ "ply", nullptr };
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return IAsset::ET_MESH; }

	//! creates/loads an animated mesh from the file.
    virtual SAssetBundle loadAsset(io::IReadFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
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
        core::vector<std::unique_ptr<SPLYElement>> ElementList;
	
		io::IReadFile* File;
		char* Buffer = nullptr;
        bool IsBinaryFile = false, IsWrongEndian = false, EndOfFile = false;
        int32_t LineLength = 0, WordLength = 0;
		char* StartPointer = nullptr, *EndPointer = nullptr, *LineEndPointer = nullptr;

		std::function<void()> deallocate = [&]()
		{ 
			if (Buffer)
			{
				_IRR_DELETE_ARRAY(Buffer, PLY_INPUT_BUFFER_SIZE);
				Buffer = nullptr;
			}
			ElementList.clear();
		};

		core::SRAIIBasedExiter<decltype(deallocate)> exiter = core::makeRAIIExiter(deallocate);
    };

    enum { E_POS = 0, E_UV = 2, E_NORM = 3, E_COL = 1 };

	bool allocateBuffer(SContext& _ctx);
	char* getNextLine(SContext& _ctx);
	char* getNextWord(SContext& _ctx);
	void fillBuffer(SContext& _ctx);
	E_PLY_PROPERTY_TYPE getPropertyType(const char* typeString) const;

	bool readVertex(SContext& _ctx, const SPLYElement &Element, core::vector<core::vectorSIMDf> _attribs[4], const IAssetLoader::SAssetLoadParams& _params);
	bool readFace(SContext& _ctx, const SPLYElement &Element, core::vector<uint32_t>& _outIndices);

	void skipElement(SContext& _ctx, const SPLYElement &Element);
	void skipProperty(SContext& _ctx, const SPLYProperty &Property);
	float getFloat(SContext& _ctx, E_PLY_PROPERTY_TYPE t);
	uint32_t getInt(SContext& _ctx, E_PLY_PROPERTY_TYPE t);
	void moveForward(SContext& _ctx, uint32_t bytes);

    bool genVertBuffersForMBuffer(ICPUMeshBuffer* _mbuf, const core::vector<core::vectorSIMDf> _attribs[4]) const;

	template<typename aType>
	static inline void performActionBasedOnOrientationSystem(aType& varToHandle, void (*performOnCertainOrientation)(aType& varToHandle))
	{
		performOnCertainOrientation(varToHandle);
	}

	IAssetManager* m_assetMgr;
};

} // end namespace asset
} // end namespace irr

#endif

// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_X_MESH_FILE_LOADER_H_INCLUDED__
#define __C_X_MESH_FILE_LOADER_H_INCLUDED__

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_X_LOADER_

#include "irr/asset/IAssetLoader.h"
#include "irr/asset/CCPUSkinnedMesh.h"
#include <sstream>

namespace irr
{
class IrrlichtDevice;
namespace io
{
class IFileSystem;
class IReadFile;
} // end namespace io
namespace scene
{
class ISceneManager;
}
namespace asset
{

#include "irr/irrpack.h"
struct SkinnedVertexIntermediateData
{
    SkinnedVertexIntermediateData()
    {
        memset(this, 0, 20);
    }
    uint8_t boneIDs[4];
    float boneWeights[4];
} PACK_STRUCT;

struct SkinnedVertexFinalData
{
public:
    uint8_t boneIDs[4];
    //! rgb10a2
    uint32_t boneWeights;
} PACK_STRUCT;
#include "irr/irrunpack.h"


//! Meshloader capable of loading x meshes.
class CXMeshFileLoader : public asset::IAssetLoader
{
public:

	//! Constructor
	CXMeshFileLoader(IAssetManager* _manager);
    ~CXMeshFileLoader();

    virtual bool isALoadableFileFormat(io::IReadFile* _file) const override;

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{ "x", nullptr };
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

    //! creates/loads an animated mesh from the file.
    virtual asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

	struct SXTemplateMaterial
	{
		std::string Name; // template name from Xfile
		video::SCPUMaterial Material; // material
	};

    //! REMOVE EVENTUALLY
    struct SXVertex
    {
        core::vector3df Pos;
        core::vector3df Normal;
        core::vector2df TCoords;
    };

	struct SXMesh
	{
		SXMesh() : BoneCount(0),AttachedJointID(-1), HasVertexColors(false) {}
		// this mesh contains triangulated texture data.
		// because in an .x file, faces can be made of more than 3
		// vertices, the indices data structure is triangulated during the
		// loading process. The IndexCountPerFace vector is filled during
		// this triangulation process and stores how much indices belong to
		// every face. This data structure can be ignored, because all data
		// in this structure is triangulated.

		core::stringc Name;

		uint32_t BoneCount;

		core::vector<uint16_t> IndexCountPerFace; // default 3, but could be more

		core::vector<asset::ICPUSkinnedMeshBuffer*> Buffers;

		core::vector<SXVertex> Vertices;
		core::vector<uint32_t> Colors;
		core::vector<core::vector2df> TCoords2;
		core::vector<SkinnedVertexIntermediateData> VertexSkinWeights;

		core::vector<uint32_t> Indices;

		core::vector<uint32_t> FaceMaterialIndices; // index of material for each face

		core::vector<video::SCPUMaterial> Materials; // material vector

		int32_t AttachedJointID;

		bool HasVertexColors;
	};

private:
    struct SContext
    {
        SContext(const asset::IAssetLoader::SAssetLoadContext& _inner, uint32_t _topHierarchyLevel, asset::IAssetLoader::IAssetLoaderOverride* _ovrr) :
            Inner(_inner),
			topHierarchyLevel(_topHierarchyLevel),
            loaderOverride(_ovrr),
            AllJoints(0), AnimatedMesh(0),
            BinaryNumCount(0),
            CurFrame(0), MajorVersion(0), MinorVersion(0), BinaryFormat(false), FloatSize(0)
        {}

        ~SContext()
        {
            for (SXMesh* m : Meshes)
                delete m;
        }

        core::vector<asset::ICPUSkinnedMesh::SJoint*> *AllJoints;

        asset::CCPUSkinnedMesh* AnimatedMesh;

        std::istringstream fileContents;
        // counter for number arrays in binary format
        uint32_t BinaryNumCount;
        io::path FilePath;

        asset::ICPUSkinnedMesh::SJoint *CurFrame;

        core::vector<SXMesh*> Meshes;

        core::vector<SXTemplateMaterial> TemplateMaterials;

        uint32_t MajorVersion;
        uint32_t MinorVersion;
        bool BinaryFormat;
        int8_t FloatSize;

        asset::IAssetLoader::SAssetLoadContext Inner;
		uint32_t topHierarchyLevel;
        asset::IAssetLoader::IAssetLoaderOverride* loaderOverride;
    };

	bool load(SContext& _ctx, io::IReadFile* file, const asset::IAssetLoader::SAssetLoadParams& _params);

	bool readFileIntoMemory(SContext& _ctx, io::IReadFile* file);

	bool parseFile(SContext& _ctx, const asset::IAssetLoader::SAssetLoadParams& _params);

	bool parseDataObject(SContext& _ctx, const asset::IAssetLoader::SAssetLoadParams& _params);

	bool parseDataObjectTemplate(SContext& _ctx);

	bool parseDataObjectFrame(SContext& _ctx, asset::ICPUSkinnedMesh::SJoint *parent, const asset::IAssetLoader::SAssetLoadParams& _params);

	bool parseDataObjectTransformationMatrix(SContext& _ctx, core::matrix3x4SIMD&mat, const asset::IAssetLoader::SAssetLoadParams& _params);

	bool parseDataObjectMesh(SContext& _ctx, SXMesh &mesh, const asset::IAssetLoader::SAssetLoadParams& _params);

	bool parseDataObjectSkinWeights(SContext& _ctx, SXMesh &mesh, const asset::IAssetLoader::SAssetLoadParams& _params);

	bool parseDataObjectSkinMeshHeader(SContext& _ctx, SXMesh &mesh);

	bool parseDataObjectMeshNormals(SContext& _ctx, SXMesh &mesh, const asset::IAssetLoader::SAssetLoadParams& _params);

	bool parseDataObjectMeshTextureCoords(SContext& _ctx, SXMesh &mesh);

	bool parseDataObjectMeshVertexColors(SContext& _ctx, SXMesh &mesh);

	bool parseDataObjectMeshMaterialList(SContext& _ctx, SXMesh &mesh);

	bool parseDataObjectMaterial(SContext& _ctx, video::SCPUMaterial& material);

	bool parseDataObjectAnimationSet(SContext& _ctx, const asset::IAssetLoader::SAssetLoadParams& _params);

	bool parseDataObjectAnimation(SContext& _ctx, const asset::IAssetLoader::SAssetLoadParams& _params);

	bool parseDataObjectAnimationKey(SContext& _ctx, asset::ICPUSkinnedMesh::SJoint *joint, const asset::IAssetLoader::SAssetLoadParams& _params);

	bool parseDataObjectTextureFilename(SContext& _ctx, std::string& texturename);

	bool parseUnknownDataObject(SContext& _ctx);

	//! places pointer to next begin of a token, and ignores comments
	void findNextNoneWhiteSpace(SContext& _ctx);

	//! places pointer to next begin of a token, which must be a number,
	// and ignores comments
	void findNextNoneWhiteSpaceNumber(SContext& _ctx);

	//! returns next parseable token. Returns empty string if no token there
	std::string getNextToken(SContext& _ctx);

	//! reads header of dataobject including the opening brace.
	//! returns false if error happened, and writes name of object
	//! if there is one
	bool readHeadOfDataObject(SContext& _ctx, std::string* outname=0);

	//! checks for closing curly brace, returns false if not there
	bool checkForClosingBrace(SContext& _ctx);

	//! checks for one following semicolons, returns false if not there
	bool checkForOneFollowingSemicolons(SContext& _ctx);

	//! checks for two following semicolons, returns false if they are not there
	bool checkForTwoFollowingSemicolons(SContext& _ctx);

	//! reads a x file style string
	bool getNextTokenAsString(SContext& _ctx, std::string& out);

	uint16_t readBinWord(SContext& _ctx);
	uint32_t readBinDWord(SContext& _ctx);
	uint32_t readInt(SContext& _ctx);
	float readFloat(SContext& _ctx);
	bool readVector2(SContext& _ctx, core::vector2df& vec);
	bool readVector3(SContext& _ctx, core::vector3df& vec);
	bool readMatrix(SContext& _ctx, core::matrix3x4SIMD& mat, const asset::IAssetLoader::SAssetLoadParams& _params);
	bool readRGB(SContext& _ctx, video::SColor& color);
	bool readRGBA(SContext& _ctx, video::SColor& color);

	IAssetManager* AssetManager;
	io::IFileSystem* FileSystem;

	template<typename aType>
	static inline void performActionBasedOnOrientationSystem(aType& varToHandle, void (*performOnCertainOrientation)(aType& varToHandle))
	{
		performOnCertainOrientation(varToHandle);
	}
};

} // end namespace asset
} // end namespace irr
#endif // _IRR_COMPILE_WITH_X_LOADER_
#endif
// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_X_MESH_FILE_LOADER_H_INCLUDED__
#define __C_X_MESH_FILE_LOADER_H_INCLUDED__

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_X_LOADER_

#include "IMeshLoader.h"
#include "irrString.h"
#include "CSkinnedMesh.h"


namespace irr
{
namespace io
{
	class IFileSystem;
	class IReadFile;
} // end namespace io
namespace scene
{

class ISceneManager;

//! Meshloader capable of loading x meshes.
class CXMeshFileLoader : public IMeshLoader
{
public:

	//! Constructor
	CXMeshFileLoader(scene::ISceneManager* smgr, io::IFileSystem* fs);

	//! returns true if the file maybe is able to be loaded by this class
	//! based on the file extension (e.g. ".cob")
	virtual bool isALoadableFileExtension(const io::path& filename) const;

	//! creates/loads an animated mesh from the file.
	//! \return Pointer to the created mesh. Returns 0 if loading failed.
	//! If you no longer need the mesh, you should call IAnimatedMesh::drop().
	//! See IReferenceCounted::drop() for more information.
	virtual ICPUMesh* createMesh(io::IReadFile* file);

	struct SXTemplateMaterial
	{
		core::stringc Name; // template name from Xfile
		video::SMaterial Material; // material
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
		// loading process. The IndexCountPerFace array is filled during
		// this triangulation process and stores how much indices belong to
		// every face. This data structure can be ignored, because all data
		// in this structure is triangulated.

		core::stringc Name;

		u32 BoneCount;

		core::array<u16> IndexCountPerFace; // default 3, but could be more

		core::array<SCPUSkinMeshBuffer*> Buffers;

		core::array<SXVertex> Vertices;
		core::array<uint32_t> Colors;
		core::array<core::vector2df> TCoords2;
		core::array<SkinnedVertexIntermediateData> VertexSkinWeights;

		core::array<u32> Indices;

		core::array<u32> FaceMaterialIndices; // index of material for each face

		core::array<video::SMaterial> Materials; // material array

		s32 AttachedJointID;

		bool HasVertexColors;
	};

private:

	bool load(io::IReadFile* file);

	bool readFileIntoMemory(io::IReadFile* file);

	bool parseFile();

	bool parseDataObject();

	bool parseDataObjectTemplate();

	bool parseDataObjectFrame(ICPUSkinnedMesh::SJoint *parent);

	bool parseDataObjectTransformationMatrix(core::matrix4x3 &mat);

	bool parseDataObjectMesh(SXMesh &mesh);

	bool parseDataObjectSkinWeights(SXMesh &mesh);

	bool parseDataObjectSkinMeshHeader(SXMesh &mesh);

	bool parseDataObjectMeshNormals(SXMesh &mesh);

	bool parseDataObjectMeshTextureCoords(SXMesh &mesh);

	bool parseDataObjectMeshVertexColors(SXMesh &mesh);

	bool parseDataObjectMeshMaterialList(SXMesh &mesh);

	bool parseDataObjectMaterial(video::SMaterial& material);

	bool parseDataObjectAnimationSet();

	bool parseDataObjectAnimation();

	bool parseDataObjectAnimationKey(ICPUSkinnedMesh::SJoint *joint);

	bool parseDataObjectTextureFilename(core::stringc& texturename);

	bool parseUnknownDataObject();

	//! places pointer to next begin of a token, and ignores comments
	void findNextNoneWhiteSpace();

	//! places pointer to next begin of a token, which must be a number,
	// and ignores comments
	void findNextNoneWhiteSpaceNumber();

	//! returns next parseable token. Returns empty string if no token there
	core::stringc getNextToken();

	//! reads header of dataobject including the opening brace.
	//! returns false if error happened, and writes name of object
	//! if there is one
	bool readHeadOfDataObject(core::stringc* outname=0);

	//! checks for closing curly brace, returns false if not there
	bool checkForClosingBrace();

	//! checks for one following semicolons, returns false if not there
	bool checkForOneFollowingSemicolons();

	//! checks for two following semicolons, returns false if they are not there
	bool checkForTwoFollowingSemicolons();

	//! reads a x file style string
	bool getNextTokenAsString(core::stringc& out);

	void readUntilEndOfLine();

	u16 readBinWord();
	u32 readBinDWord();
	u32 readInt();
	f32 readFloat();
	bool readVector2(core::vector2df& vec);
	bool readVector3(core::vector3df& vec);
	bool readMatrix(core::matrix4& mat);
	bool readRGB(video::SColor& color);
	bool readRGBA(video::SColor& color);

	ISceneManager* SceneManager;
	io::IFileSystem* FileSystem;

	core::array<ICPUSkinnedMesh::SJoint*> *AllJoints;

	CCPUSkinnedMesh* AnimatedMesh;

	c8* Buffer;
	const c8* P;
	c8* End;
	// counter for number arrays in binary format
	u32 BinaryNumCount;
	u32 Line;
	io::path FilePath;

	ICPUSkinnedMesh::SJoint *CurFrame;

	core::array<SXMesh*> Meshes;

	core::array<SXTemplateMaterial> TemplateMaterials;

	u32 MajorVersion;
	u32 MinorVersion;
	bool BinaryFormat;
	c8 FloatSize;
};

} // end namespace scene
} // end namespace irr
#endif // _IRR_COMPILE_WITH_X_LOADER_
#endif

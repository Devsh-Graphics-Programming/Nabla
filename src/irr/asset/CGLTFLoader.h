// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifdef _IRR_COMPILE_WITH_GLTF_LOADER_

#include "irr/asset/ICPUImageView.h"
#include "irr/asset/IAssetLoader.h"

namespace irr
{
	namespace asset
	{
		//! glTF Loader capable of loading .gltf files
		/*
			glTF bridges the gap between 3D content creation tools and modern 3D applications 
			by providing an efficient, extensible, interoperable format for the transmission and loading of 3D content.
		*/

		class CGLTFLoader final : public asset::IAssetLoader
		{
			protected:
				virtual ~CGLTFLoader() {}

			public:
				CGLTFLoader(asset::IAssetManager* _m_assetMgr) : assetManager(_m_assetMgr) {}

				bool isALoadableFileFormat(io::IReadFile* _file) const override;

				const char** getAssociatedFileExtensions() const override
				{
					static const char* extensions[]{ "gltf", nullptr };
					return extensions;
				}

				uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

				asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

			private:

				struct CGLTFHeader
				{
					uint32_t version;
					std::optional<uint32_t> minVersion;
					std::optional<std::string> generator;
					std::optional<std::string> copyright;
				};

				enum SGLTFPrimitiveTopology
				{
					SGLTFPT_POINTS,
					SGLTFPT_LINES,
					SGLTFPT_LINE_LOOP,
					SGLTFPT_LINE_STRIP,
					SGLTFPT_TRIANGLES,
					SGLTFPT_TRIANGLE_STRIP,
					SGLTFPT_TRIANGLE_FAN
				};

				/*
					A node in the node hierarchy. A node can have either a matrix or any combination of 
					translation/rotation/scale (TRS) properties.
				*/

				struct SGLTFNode
				{
					std::optional<std::string> name;	//!< The user-defined name of this object.
					std::optional<std::vector<uint32_t>> children;	//!< The indices of this node's children.
					std::optional<uint32_t> mesh;		//!< The index of the mesh in this node.
					std::optional<uint32_t> camera;		//!< The index of the camera referenced by this node.
					std::optional<uint32_t> skin;		//!< The index of the skin referenced by this node.
					std::optional<uint32_t> weights;	//!< The weights of the instantiated Morph Target. Number of elements must match number of Morph Targets of used mesh.

					struct SGLTFNTransformationTRS
					{
						core::vector3df_SIMD translation;	//!< The node's translation along the x, y, and z axes.
						core::vector3df_SIMD scale;			//!< The node's non-uniform scale, given as the scaling factors along the x, y, and z axes.
						core::vector4df_SIMD rotation;		//!< The node's unit quaternion rotation in the order (x, y, z, w), where w is the scalar.
					};

					union SGLTFTransformation
					{
						SGLTFTransformation() {}
						~SGLTFTransformation() {}

						SGLTFNTransformationTRS trs;	
						core::matrix4SIMD matrix;		//!< A floating-point 4x4 transformation matrix stored in column-major order.
					} transformation;

					bool validate()
					{
						if (mesh.has_value() && mesh.value() < 0)
							return false;

						if (camera.has_value() && camera.value() < 0)
							return false;

						if (skin.has_value() && skin.value() < 0)
							return false;
					}
				};

				/*
					A set of primitives to be rendered. A node can contain one mesh. 
					A node's transform places the mesh in the scene.
				*/

				struct SGLTFMesh
				{
					struct SPrimitive
					{
						std::unordered_map<std::string, uint32_t> attributes;	//!< A dictionary object, where each key corresponds to mesh attribute semantic and each value is the index of the accessor containing attribute's data.
						std::optional<uint32_t> indices;						//!< The index of the accessor that contains the indices.
						std::optional<uint32_t> material;						//!< The index of the material to apply to this primitive when rendering.
						std::optional<uint32_t> mode;							//!< The type of primitives to render.
						std::optional<std::unordered_map<std::string, uint32_t>> targets;	//!< An array of Morph Targets, each Morph Target is a dictionary mapping attributes (only POSITION, NORMAL, and TANGENT supported) to their deviations in the Morph Target.
					};

					std::vector<SPrimitive> primitives;					//!< An array of primitives, each defining geometry to be rendered with a material.
					std::optional<std::vector<uint32_t>> weights;		//!< Array of weights to be applied to the Morph Targets.
					std::optional<std::string> name;					//!< The user-defined name of this object.
				};	

				struct SGLTFAccessor
				{
					enum SCompomentType
					{
						SCT_BYTE = 5120,
						SCT_UNSIGNED_BYTE = 5121,
						SCT_SHORT = 5122,
						SCT_UNSIGNED_SHORT = 5123,
						SCT_UNSIGNED_INT = 5125,
						SCT_FLOAT = 5126
					};
				};

				struct SGLTFData
				{
					std::vector<SGLTFNode> nodes;
					std::vector<SGLTFMesh> meshes;
				};

				asset::IAssetManager* const assetManager;
		};
	}
}

#endif // _IRR_COMPILE_WITH_GLTF_LOADER_

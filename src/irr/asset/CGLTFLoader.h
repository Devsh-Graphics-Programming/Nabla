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

				struct SGLTFNode
				{
					std::optional<std::string> name;
					std::optional<std::vector<uint32_t>> children;
					std::optional<uint32_t> mesh;
					std::optional<uint32_t> camera;
					std::optional<uint32_t> skin;
					std::optional<uint32_t> weights;

					struct SGLTFNTransformationTRS
					{
						core::vector3df_SIMD translation;	//!< translation, local coordinate system
						core::vector3df_SIMD scale;			//!< scale, local coordinate system
						core::vector4df_SIMD rotation;		//!< quaternion, local coordinate system
					};

					union SGLTFTransformation
					{
						SGLTFTransformation() {}
						~SGLTFTransformation() {}

						SGLTFNTransformationTRS trs;
						core::matrix4SIMD matrix;
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
				};

				asset::IAssetManager* const assetManager;
		};
	}
}

#endif // _IRR_COMPILE_WITH_GLTF_LOADER_

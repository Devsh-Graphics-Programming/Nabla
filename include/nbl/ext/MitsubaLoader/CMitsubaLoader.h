// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_MITSUBA_LOADER_H_INCLUDED__
#define __C_MITSUBA_LOADER_H_INCLUDED__

#include "matrix4SIMD.h"
#include "nbl/asset/asset.h"
#include "nbl/system/path.h"
#include "nbl/asset/utils/ICPUVirtualTexture.h"

#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"
#include "nbl/ext/MitsubaLoader/CMitsubaMetadata.h"
#include "nbl/ext/MitsubaLoader/CElementShape.h"
#include "nbl/ext/MitsubaLoader/SContext.h"


namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

class CElementBSDF;
class CMitsubaMaterialCompilerFrontend;

class CMitsubaLoader : public asset::IRenderpassIndependentPipelineLoader
{
		friend class CMitsubaMaterialCompilerFrontend;
	public:
		//! Constructor
		CMitsubaLoader(asset::IAssetManager* _manager, system::ISystem* _system);

		void initialize() override;

	protected:
		system::ISystem* m_system;

		//! Destructor
		virtual ~CMitsubaLoader() = default;

		static core::smart_refctd_ptr<asset::ICPUPipelineLayout> createPipelineLayout(asset::IAssetManager* _manager, asset::ICPUVirtualTexture* _vt);

		//
		core::vector<SContext::shape_ass_type>	getMesh(SContext& ctx, uint32_t hierarchyLevel, CElementShape* shape, const system::logger_opt_ptr& logger);
		core::vector<SContext::shape_ass_type>	loadShapeGroup(SContext& ctx, uint32_t hierarchyLevel, const CElementShape::ShapeGroup* shapegroup, const core::matrix3x4SIMD& relTform, const system::logger_opt_ptr& _logger);
		SContext::shape_ass_type				loadBasicShape(SContext& ctx, uint32_t hierarchyLevel, CElementShape* shape, const core::matrix3x4SIMD& relTform, const system::logger_opt_ptr& logger);
		
		SContext::tex_ass_type					cacheTexture(SContext& ctx, uint32_t hierarchyLevel, const CElementTexture* texture, bool _restore = false);

		SContext::bsdf_type getBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf, const system::logger_opt_ptr& logger);
		SContext::bsdf_type genBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf, const system::logger_opt_ptr& logger);

		template <typename Iter>
		core::smart_refctd_ptr<asset::ICPUDescriptorSet> createDS0(const SContext& _ctx, asset::ICPUPipelineLayout* _layout, const asset::material_compiler::CMaterialCompilerGLSLBackendCommon::result_t& _compResult, Iter meshBegin, Iter meshEnd);

	public:
		//! Check if the file might be loaded by this class
		/** Check might look into the file.
		\param file File handle to check.
		\return True if file seems to be loadable. */
		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const override;

		//! Returns an array of string literals terminated by nullptr
		const char** getAssociatedFileExtensions() const override;

		//! Returns the assets loaded by the loader
		/** Bits of the returned value correspond to each IAsset::E_TYPE
		enumeration member, and the return value cannot be 0. */
		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH/*|asset::IAsset::ET_SCENE|asset::IAsset::ET_IMPLEMENTATION_SPECIFIC_METADATA*/; }

		//! Loads an asset from an opened file, returns nullptr in case of failure.
		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};

}
}
}
#endif
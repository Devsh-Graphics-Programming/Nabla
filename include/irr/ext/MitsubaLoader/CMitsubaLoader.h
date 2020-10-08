#ifndef __C_MITSUBA_LOADER_H_INCLUDED__
#define __C_MITSUBA_LOADER_H_INCLUDED__

#include "matrix4SIMD.h"
#include "irr/asset/asset.h"
#include "IFileSystem.h"
#include "irr/asset/ICPUVirtualTexture.h"

#include "irr/ext/MitsubaLoader/CSerializedLoader.h"
#include "irr/ext/MitsubaLoader/CGlobalMitsubaMetadata.h"
#include "irr/ext/MitsubaLoader/CElementShape.h"
#include "irr/ext/MitsubaLoader/CMitsubaPipelineMetadata.h"
#include "irr/ext/MitsubaLoader/CMitsubaMetadata.h"
#include "irr/ext/MitsubaLoader/SContext.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class CElementBSDF;
class CMitsubaMaterialCompilerFrontend;

class CMitsubaLoader : public asset::IAssetLoader
{
		friend class CMitsubaMaterialCompilerFrontend;
	public:
		//! Constructor
		CMitsubaLoader(asset::IAssetManager* _manager);

	protected:
		asset::IAssetManager* m_manager;

#include "irr/irrpack.h"
		//compatible with std430 and std140
		struct SInstanceData
		{
			core::matrix3x4SIMD tform;//mat4x3
			//elements (0,3) and (1,3) are (first,count) for bsdf_instrStream (remainder and pdf stream in case of RT backend)
			core::matrix3x4SIMD normalMatrix;//mat4x3
			std::pair<uint32_t,uint32_t> prefetch_instrStream;
			std::pair<uint32_t, uint32_t> nprecomp_instrStream;
			std::pair<uint32_t, uint32_t> genchoice_instrStream;
			uint64_t emissive;//uvec2, rgb19e7
		} PACK_STRUCT;
#include "irr/irrunpack.h"

		//! Destructor
		virtual ~CMitsubaLoader() = default;

		//
		core::vector<SContext::shape_ass_type>	getMesh(SContext& ctx, uint32_t hierarchyLevel, CElementShape* shape);
		core::vector<SContext::shape_ass_type>	loadShapeGroup(SContext& ctx, uint32_t hierarchyLevel, const CElementShape::ShapeGroup* shapegroup, const core::matrix3x4SIMD& relTform);
		SContext::shape_ass_type				loadBasicShape(SContext& ctx, uint32_t hierarchyLevel, CElementShape* shape, const core::matrix3x4SIMD& relTform);
		
		SContext::tex_ass_type					getTexture(SContext& ctx, uint32_t hierarchyLevel, const CElementTexture* texture);

		SContext::bsdf_type getBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf);
		SContext::bsdf_type genBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf);

		template <typename Iter>
		core::smart_refctd_ptr<asset::ICPUDescriptorSet> createDS0(const SContext& _ctx, const asset::material_compiler::CMaterialCompilerGLSLBackendCommon::result_t& _compResult, Iter meshBegin, Iter meshEnd);

		core::smart_refctd_ptr<CMitsubaPipelineMetadata> createPipelineMetadata(core::smart_refctd_ptr<asset::ICPUDescriptorSet>&& _ds0, const asset::ICPUPipelineLayout* _layout);

	public:
		//! Check if the file might be loaded by this class
		/** Check might look into the file.
		\param file File handle to check.
		\return True if file seems to be loadable. */
		bool isALoadableFileFormat(io::IReadFile* _file) const override;

		//! Returns an array of string literals terminated by nullptr
		const char** getAssociatedFileExtensions() const override;

		//! Returns the assets loaded by the loader
		/** Bits of the returned value correspond to each IAsset::E_TYPE
		enumeration member, and the return value cannot be 0. */
		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH/*|asset::IAsset::ET_SCENE|asset::IAsset::ET_IMPLEMENTATION_SPECIFIC_METADATA*/; }

		//! Loads an asset from an opened file, returns nullptr in case of failure.
		asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};

}
}
}
#endif
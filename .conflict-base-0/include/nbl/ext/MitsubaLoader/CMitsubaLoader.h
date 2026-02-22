// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_LOADER_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_LOADER_H_INCLUDED_


#include "nbl/asset/asset.h"

#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/SContext.h"


namespace nbl::ext::MitsubaLoader
{


class CElementBSDF;
class CMitsubaMaterialCompilerFrontend;

#if 0 // TODO
//#include "nbl/builtin/glsl/ext/MitsubaLoader/instance_data_struct.glsl"
#define uint uint32_t
#define uvec2 uint64_t
#define mat4x3 nbl::core::matrix3x4SIMD
#define nbl_glsl_MC_material_data_t asset::material_compiler::material_data_t
struct nbl_glsl_ext_Mitsuba_Loader_instance_data_t
{
	struct vec3
	{
		float x, y, z;
	};
	mat4x3 tform;
	vec3 normalMatrixRow0;
	uint padding0;
	vec3 normalMatrixRow1;
	uint padding1;
	vec3 normalMatrixRow2;
	uint determinantSignBit;
	nbl_glsl_MC_material_data_t material;
};
#undef uint
#undef uvec2
#undef mat4x3
#undef nbl_glsl_MC_material_data_t
using instance_data_t = nbl_glsl_ext_Mitsuba_Loader_instance_data_t;
#endif

class CMitsubaLoader final : public asset::ISceneLoader
{
//		friend class CMitsubaMaterialCompilerFrontend;

		const ParserManager m_parser;
		core::smart_refctd_ptr<system::ISystem> m_system;

		//! Destructor
		virtual ~CMitsubaLoader() = default;

#if 0
		void									cacheTexture(SContext& ctx, uint32_t hierarchyLevel, const CElementTexture* texture, const CMitsubaMaterialCompilerFrontend::E_IMAGE_VIEW_SEMANTIC semantic);
		void cacheEmissionProfile(SContext& ctx, const CElementEmissionProfile* profile);

		SContext::bsdf_type getBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf, const CElementEmitter* emitter, core::matrix4SIMD tform);
		SContext::bsdf_type genBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf);

		template <typename Iter>
		core::smart_refctd_ptr<asset::ICPUDescriptorSet> createDS0(const SContext& _ctx, asset::ICPUPipelineLayout* _layout, const asset::material_compiler::CMaterialCompilerGLSLBackendCommon::result_t& _compResult, Iter meshBegin, Iter meshEnd);
#endif
	public:
		//! Constructor
		inline CMitsubaLoader(core::smart_refctd_ptr<system::ISystem>&& _system) : m_parser(), m_system(std::move(_system)) {}

		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger=nullptr) const override;

		inline const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "xml", nullptr };
			return ext;
		}

		//! Loads an asset from an opened file, returns nullptr in case of failure.
		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override=nullptr, uint32_t _hierarchyLevel=0u) override;
};

}
#endif
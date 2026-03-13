// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_LOADER_CONTEXT_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_LOADER_CONTEXT_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/interchange/CIESProfileLoader.h"

#include "nbl/ext/MitsubaLoader/CMitsubaMetadata.h"
//#include "nbl/ext/MitsubaLoader/CMitsubaMaterialCompilerFrontend.h"


namespace nbl::ext::MitsubaLoader
{
class CMitsubaLoader;

struct SContext final
{
	public:
		using interm_getAssetInHierarchy_t = asset::SAssetBundle(const char*, const uint16_t);

		SContext(
			const asset::IAssetLoader::SAssetLoadContext& _params,
			asset::IAssetLoader::IAssetLoaderOverride* _override,
			CMitsubaMetadata* _metadata
		);

		using shape_ass_type = core::smart_refctd_ptr<const asset::ICPUGeometryCollection>;
		shape_ass_type loadBasicShape(const CElementShape* shape);
		// the `shape` will have to be `Type::SHAPEGROUP`
		shape_ass_type loadShapeGroup(const CElementShape* shape);

		// Mitsuba XML Materials do not support emission from a BSDF node (i.e. emitter behind a coating or glass screen), its purely additive and cannot be backface emitting
		using material_t = asset::material_compiler3::CFrontendIR::typed_pointer_type<const asset::material_compiler3::CFrontendIR::CLayer>; // TODO: change to true IR
		material_t getMaterial(const CElementBSDF* bsdf, const CElementEmitter* frontFaceEmitter, const core::string& debugName, system::ISystem* debugFileWriter=nullptr);

		inline void writeFrontendForestDot3(system::ISystem* system, const system::path& filepath)
		{
			asset::material_compiler3::CFrontendIR::SDotPrinter printer = {frontIR.get(),frontIR->getMaterials()};
			writeDot3File(system,filepath,printer);
		}

		inline void transferMetadata()
		{
			meta->setGeometryCollectionMeta(std::move(shapeCache));
			meta->setGeometryCollectionMeta(std::move(groupCache));
		}

		const asset::IAssetLoader::SAssetLoadContext inner;
		asset::IAssetLoader::IAssetLoaderOverride* override_;
		std::function<interm_getAssetInHierarchy_t> interm_getAssetInHierarchy;
		std::function<interm_getAssetInHierarchy_t> interm_getImageViewInHierarchy;
		CMitsubaMetadata* meta;
		core::smart_refctd_ptr<asset::ICPUScene> scene;

	private:
		using frontend_ir_t = asset::material_compiler3::CFrontendIR;
		using frontend_material_t = frontend_ir_t::typed_pointer_type<const frontend_ir_t::CLayer>;
		// not `frontend_ir_t::CEmitter` because the color factor gets multiplied in
		using frontend_emitter_t = frontend_ir_t::typed_pointer_type<const frontend_ir_t::CMul>;
		frontend_material_t genMaterial(const CElementBSDF* bsdf, system::ISystem* debugFileWriter);
		frontend_emitter_t genEmitter(const CElementEmitter* emitter, system::ISystem* debugFileWriter);
		//
		void writeDot3File(system::ISystem* system, const system::path& filepath, frontend_ir_t::SDotPrinter& printer);
		//
		hlsl::float32_t2x3 getParameters(const std::span<frontend_ir_t::SParameter> out, const CElementTexture::FloatOrTexture& src);
		hlsl::float32_t2x3 getParameters(const std::span<frontend_ir_t::SParameter,3> out, const CElementTexture::SpectrumOrTexture& src);
		frontend_ir_t::SParameter getTexture(const CElementTexture* tex, hlsl::float32_t2x3* outUvTransform);
		frontend_ir_t::SParameter genProfile(const CElementEmissionProfile* profile);

		//
		core::unordered_map<const CElementShape*,CMitsubaMetadata::SGeometryCollectionMetaPair> shapeCache;
		//
		core::unordered_map<const CElementShape::ShapeGroup*,CMitsubaMetadata::SGeometryCollectionMetaPair> groupCache;
		//
		core::unordered_map<const CElementEmitter*,frontend_emitter_t> emitterCache;
		core::unordered_map<const CElementBSDF*,frontend_material_t> bsdfCache;
		core::unordered_map<const CElementEmissionProfile*,frontend_ir_t::SParameter> profileCache;

#if 0 // stuff that belongs in the Material Compiler backend
		//image, sampler
		using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>,core::smart_refctd_ptr<asset::ICPUSampler>>;
		//image, scale 
		core::map<core::smart_refctd_ptr<asset::ICPUImage>,float> derivMapCache;
#endif
		core::smart_refctd_ptr<frontend_ir_t> frontIR;
		// common frontend nodes
		frontend_ir_t::typed_pointer_type<const frontend_ir_t::CDeltaTransmission> deltaTransmission;
		// Common Debug Names
		enum class ECommonDebug : uint16_t
		{
			Albedo,
			MitsubaExtraFactor,
			Count
		};
		frontend_ir_t::obj_pool_type::typed_pointer_type<const frontend_ir_t::CDebugInfo> commonDebugNames[uint16_t(ECommonDebug::Count)];
};

}
#endif
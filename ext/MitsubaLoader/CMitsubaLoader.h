#ifndef __C_MITSUBA_LOADER_H_INCLUDED__
#define __C_MITSUBA_LOADER_H_INCLUDED__

#include "matrix4SIMD.h"
#include "irr/asset/asset.h"
#include "IFileSystem.h"

#include "../../ext/MitsubaLoader/CSerializedLoader.h"
#include "../../ext/MitsubaLoader/CGlobalMitsubaMetadata.h"
#include "../../ext/MitsubaLoader/CElementShape.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

namespace bsdf
{
	using instr_t = uint64_t;

	enum E_OPCODE : uint8_t
	{
		OP_DIFFUSE,
		OP_DIFFTRANS,
		OP_DIELECTRIC,
		OP_CONDUCTOR,
		OP_PLASTIC,
		OP_TRANSPARENT,
		OP_WARD,
		OP_INVALID,
		//all below are meta (have children)
		OP_COATING,
		OP_BLEND,
		OP_BUMPMAP,
	};

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_OPCODE_WIDTH = 4u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_OPCODE_MASK = 0xfu;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BITFIELDS_SHIFT = INSTR_OPCODE_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BITFIELDS_WIDTH = 6u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_REFL_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_REFL_TEX = INSTR_OPCODE_WIDTH + 0u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_ALPHA_U_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_ALPHA_U_TEX = INSTR_OPCODE_WIDTH + 2u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_ALPHA_V_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_ALPHA_V_TEX = INSTR_OPCODE_WIDTH + 3u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_SPEC_TRANS_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_SPEC_TRANS_TEX = INSTR_OPCODE_WIDTH + 0u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_NDF = 0x3u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_NDF = INSTR_OPCODE_WIDTH + 0u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_WARD_VARIANT = 0x3u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_WARD_VARIANT = INSTR_OPCODE_WIDTH + 0u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_FAST_APPROX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_FAST_APPROX = INSTR_OPCODE_WIDTH + 1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_NONLINEAR = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_NONLINEAR = INSTR_OPCODE_WIDTH + 4u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_SIGMA_A_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_SIGMA_A_TEX = INSTR_OPCODE_WIDTH + 4u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_WEIGHT_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_WEIGHT_TEX = INSTR_OPCODE_WIDTH + 0u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_TWOSIDED = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_TWOSIDED = INSTR_OPCODE_WIDTH + 5u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BUMPMAP_REG_WIDTH = 2u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BUMPMAP_REG_SHIFT = INSTR_OPCODE_WIDTH + INSTR_BITFIELDS_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BUMPMAP_REG_MASK = 0x03u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BSDF_BUF_OFFSET_WIDTH = 20u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BSDF_BUF_OFFSET_SHIFT = INSTR_OPCODE_WIDTH + INSTR_BITFIELDS_WIDTH + INSTR_BUMPMAP_REG_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BSDF_BUF_OFFSET_MASK = 0xfffffu;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_WIDTH = 8u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_MASK = 0xffu;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_DST_SHIFT = INSTR_BSDF_BUF_OFFSET_SHIFT + INSTR_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH*0u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_DST_SHIFT = INSTR_BSDF_BUF_OFFSET_SHIFT + INSTR_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH*1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_DST_SHIFT = INSTR_BSDF_BUF_OFFSET_SHIFT + INSTR_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH*2u;


	//padding which make sure struct has sizeof at least min_desired_size but aligned to 4*sizeof(uint32_t)=16
	//both size and desired_size are meant to be expressed in 4 byte units (i.e. to express size of 8, `size` should be 2)
	template<size_t size, size_t min_desired_size>
	using padding_t = uint32_t[ ((min_desired_size + 3ull) & ~(3ull)) - size ];
	//in 4 byte units
	constexpr size_t max_bsdf_struct_size = 8ull;

#include "irr/irrpack.h"
	struct alignas(16) SAllDiffuse
	{
		//if flag decides to use alpha texture, {alpha[0..1]} is bindless texture ID (bit-cast to uvec2)
		//otherwise alpha[0] is single-float alpha
		uint32_t alpha[2];
		//if flag decides to use reflectance texture, `reflectance` is bindless texture ID (bit-cast to uvec2)
		//otherwise `reflectance` is constant reflectance in RGB19E7 format
		uint64_t reflectance;
		padding_t<4, max_bsdf_struct_size> padding;
	} PACK_STRUCT;
	struct alignas(16) SDiffuseTransmitter
	{
		//if flag decides to use transmittance texture, `transmittance` is bindless texture ID (bit-cast to uvec2)
		//otherwise `transmittance` is constant transmittance
		uint64_t transmittance;
		padding_t<2, max_bsdf_struct_size> padding;
	} PACK_STRUCT;
	struct alignas(16) SAllDielectric
	{
		float eta;
		//if NDF is Ashikhmin-Shirley:
		//	if flag decides to use alpha_u texture, alpha_u[0..1] is bindless texture ID
		//	otherwise alpha_u[0] is constant single-float alpha_u
		//	if flag decides to use alpha_v texture, alpha_v[0..1] is bindless texture ID
		//	otherwise alpha_v[0] is constant single-float alpha_v
		//otherwise (different NDF)
		//	if flag decides to use alpha texture, alpha_u[0..1] is bindless texture ID
		//	otherwise alpha_u[0] is constant single-float alpha
		uint32_t alpha_u[2];
		uint32_t alpha_v[2];
		padding_t<5, max_bsdf_struct_size> padding;
	} PACK_STRUCT;
	struct alignas(16) SAllConductor
	{
		//same as for SAllDielectric::alpha_u,alpha_v
		uint32_t alpha_u[2];
		uint32_t alpha_v[2];
		//ior[0] real part of eta in RGB19E7 format
		//ior[1] is imaginary part of eta in RGB19E7 format
		uint64_t eta[2];
		//padding_t<9, max_bsdf_struct_size> padding;
	} PACK_STRUCT;
	struct alignas(16) SAllPlastic
	{
		float eta;
		//same as for SAllDielectric::alpha_u,alpha_v
		uint32_t alpha_u[2];
		uint32_t alpha_v[2];
		padding_t<5, max_bsdf_struct_size> padding;
	} PACK_STRUCT;
	struct alignas(16) SAllCoating
	{
		//thickness and eta encoded as 2x float16, thickness on bits 0:15, eta on bits 16:31
		uint32_t thickness_eta;
		//same as for SAllDielectric::alpha_u,alpha_v
		uint32_t alpha_u[2];
		uint32_t alpha_v[2];
		//rgb in RGB19E& format or texture id (flag decides)
		uint64_t sigmaA;
		padding_t<7, max_bsdf_struct_size> padding;
	} PACK_STRUCT;
	/*
	struct alignas(16) SPhong
	{
		uint32_t bitfields;
		//if flag decides to use exponent texture, exponent[0..1] is bindless texture ID
		//otherwise exponent[0] is constant exponent
		float exponent[2];
		//same as SAllDielectric::specReflectance
		float specReflectance[3];
		//same as SAllPlastic::diffReflectance
		float diffReflectance[3];
	} PACK_STRUCT;
	*/
	struct alignas(16) SWard
	{
		//same as for SAllDielectric::alpha_u,alpha_v
		uint32_t alpha_u[2];
		uint32_t alpha_v[2];
		padding_t<4, max_bsdf_struct_size> padding;
	} PACK_STRUCT;
	struct alignas(16) SBumpMap
	{
		//bindless texture id
		uint64_t bumpmap;
		padding_t<2, max_bsdf_struct_size> padding;
	} PACK_STRUCT;
	struct alignas(16) SMixture
	{
		//two weights resulting from translation of mixture into multiple blend BSDFs (encoded as 2x float16)
		uint32_t weights;
		padding_t<1, max_bsdf_struct_size> padding;
	} PACK_STRUCT;
	struct alignas(16) SBlend
	{
		//per-channel blend factor encoded as RGB19E7 (RGB instead of single-float in order to encode MASK bsdf as BLEND with fully transparent)
		//if flag decides to use weight texture, `weight` is bindless texture ID
		//otherwise `weight` is constant RGB19E7 blend weight
		uint64_t weight;
		padding_t<2, max_bsdf_struct_size> padding;
	} PACK_STRUCT;
#include "irr/irrunpack.h"

	union SBSDFUnion
	{
		SAllDiffuse diffuse;
		SDiffuseTransmitter diffuseTransmitter;
		SAllDielectric dielectric;
		SAllConductor conductor;
		SAllPlastic plastic;
		SAllCoating coating;
		SBumpMap bumpmap;
		SWard ward;
		SMixture mixture;
		SBlend blend;
	};
}

class CElementBSDF;

struct NastyTemporaryBitfield
{
#define MITS_TWO_SIDED		0x80000000u
#define MITS_USE_TEXTURE	0x40000000u
#define MITS_BUMPMAP		0x20000000u
	uint32_t _bitfield;
};

class CMitsubaLoader : public asset::IAssetLoader
{
	public:
		//! Constructor
		CMitsubaLoader(asset::IAssetManager* _manager);

	protected:
		asset::IAssetManager* manager;

		struct SContext
		{
			const asset::IGeometryCreator* creator;
			const asset::IMeshManipulator* manipulator;
			const asset::IAssetLoader::SAssetLoadParams params;
			asset::IAssetLoader::IAssetLoaderOverride* override;
			CGlobalMitsubaMetadata* globalMeta;

			//
			using group_ass_type = core::smart_refctd_ptr<asset::ICPUMesh>;
			core::map<const CElementShape::ShapeGroup*, group_ass_type> groupCache;
			//
			using shape_ass_type = core::smart_refctd_ptr<asset::ICPUMesh>;
			core::map<const CElementShape*, shape_ass_type> shapeCache;
			//! TODO: change to CPU graphics pipeline
			using bsdf_ass_type = core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>;
			core::map<const CElementBSDF*, bsdf_ass_type> pipelineCache;
			//! TODO: even later when texture changes come, might have to return not only a combined sampler but some GLSL sampling code due to the "scale" and offset XML nodes
			using tex_ass_type = std::pair<core::smart_refctd_ptr<asset::ICPUImageView>,core::smart_refctd_ptr<asset::ICPUSampler> >;
			core::unordered_map<const CElementTexture*, tex_ass_type> textureCache;

			core::vector<bsdf::SBSDFUnion> bsdfBuffer;
			core::vector<bsdf::instr_t> instrBuffer;
			uint32_t instrBufOffset;
		};

		//! Destructor
		virtual ~CMitsubaLoader() = default;

		//
		SContext::shape_ass_type	getMesh(SContext& ctx, uint32_t hierarchyLevel, CElementShape* shape);
		SContext::group_ass_type	loadShapeGroup(SContext& ctx, uint32_t hierarchyLevel, const CElementShape::ShapeGroup* shapegroup);
		SContext::shape_ass_type	loadBasicShape(SContext& ctx, uint32_t hierarchyLevel, CElementShape* shape);

		SContext::bsdf_ass_type		getBSDF(SContext& ctx, uint32_t hierarchyLevel, const CElementBSDF* bsdf);
		
		SContext::tex_ass_type		getTexture(SContext& ctx, uint32_t hierarchyLevel, const CElementTexture* texture);

		void genBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf);

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
#ifndef __C_MITSUBA_LOADER_H_INCLUDED__
#define __C_MITSUBA_LOADER_H_INCLUDED__

#include "matrix4SIMD.h"
#include "irr/asset/asset.h"
#include "IFileSystem.h"
#include "irr/asset/ICPUVirtualTexture.h"

#include "../../ext/MitsubaLoader/CSerializedLoader.h"
#include "../../ext/MitsubaLoader/CGlobalMitsubaMetadata.h"
#include "../../ext/MitsubaLoader/CElementShape.h"
#include "../../ext/MitsubaLoader/CMitsubaPipelineMetadata.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

namespace bsdf
{
	using STextureData = asset::ICPUVirtualTexture::SMasterTextureData;
	// texture presence flag (flags are encoded in instruction) tells whether to use VT data or constant (depending on situation RGB encoded as rgb19e7 or single float32 value)
	union STextureDataOrConstant
	{
		STextureData texData;
		uint64_t constant_rgb19e7;
		uint32_t constant_f32;
	};
	static_assert(sizeof(STextureDataOrConstant)==sizeof(uint64_t), "STextureDataOrConstant is not 8 bytes for some reason!");

	using instr_t = uint64_t;

	enum E_OPCODE : uint8_t
	{
		//brdf
		OP_DIFFUSE,
		OP_ROUGHDIFFUSE,
		OP_CONDUCTOR,
		OP_ROUGHCONDUCTOR,
		OP_PLASTIC,
		OP_ROUGHPLASTIC,
		OP_WARD,
		OP_COATING,
		OP_ROUGHCOATING,
		//bsdf
		OP_DIFFTRANS,
		OP_DIELECTRIC,
		OP_ROUGHDIELECTRIC,
		//blend
		OP_BLEND,
		//specials
		OP_BUMPMAP,
		OP_SET_GEOM_NORMAL,
		OP_INVALID,

		OPCODE_COUNT
	};
	inline uint32_t getNumberOfSrcRegsForOpcode(E_OPCODE _op)
	{
		if (_op==OP_BLEND)
			return 2u;
		else if (_op==OP_BUMPMAP || _op==OP_COATING || _op==OP_ROUGHCOATING)
			return 1u;
		return 0u;
	}

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t REGISTER_COUNT = 16u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_OPCODE_WIDTH = 4u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_OPCODE_MASK = 0xfu;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_OPCODE_SHIFT = 0u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BITFIELDS_SHIFT = INSTR_OPCODE_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BITFIELDS_WIDTH = 9u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_REFL_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_REFL_TEX = INSTR_OPCODE_WIDTH + 0u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_PLASTIC_REFL_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_PLASTIC_REFL_TEX = INSTR_OPCODE_WIDTH + 5u;
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
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_TWOSIDED = INSTR_OPCODE_WIDTH + 6u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_MASKFLAG = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_MASKFLAG = INSTR_OPCODE_WIDTH + 7u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_OPACITY_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_OPACITY_TEX = INSTR_OPCODE_WIDTH + 8u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BSDF_BUF_OFFSET_WIDTH = 19u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BSDF_BUF_OFFSET_SHIFT = INSTR_OPCODE_WIDTH + INSTR_BITFIELDS_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BSDF_BUF_OFFSET_MASK = 0x7ffffu;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_WIDTH = 8u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_MASK = 0xffu;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_DST_SHIFT = INSTR_BSDF_BUF_OFFSET_SHIFT + INSTR_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH*0u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_SRC1_SHIFT = INSTR_BSDF_BUF_OFFSET_SHIFT + INSTR_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH*1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_SRC2_SHIFT = INSTR_BSDF_BUF_OFFSET_SHIFT + INSTR_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH*2u;

	//this has no influence on instruction execution, but useful during traversal creation/processing
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_NORMAL_ID_WIDTH = 8u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_NORMAL_ID_MASK	= 0xffu;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_NORMAL_ID_SHIFT = INSTR_REG_SRC2_SHIFT + INSTR_REG_WIDTH;


	inline E_OPCODE getOpcode(const instr_t& i)
	{
		return static_cast<E_OPCODE>(i & INSTR_OPCODE_MASK);
	}
	inline uint32_t getNormalId(const instr_t& i)
	{
		return i >> INSTR_NORMAL_ID_SHIFT;
	}
	inline core::vector3du32_SIMD getRegisters(const instr_t& i)
	{
		return core::vector3du32_SIMD(
			(i>>INSTR_REG_DST_SHIFT),
			(i>>INSTR_REG_SRC1_SHIFT),
			(i>>INSTR_REG_SRC2_SHIFT)
		) & core::vector3du32_SIMD(INSTR_REG_MASK);
	}
	inline bool isTwosided(const instr_t& i)
	{
		return (i>>BITFIELDS_SHIFT_TWOSIDED) & 1u;
	}
	inline bool isMasked(const instr_t& i)
	{
		return (i>>BITFIELDS_SHIFT_MASKFLAG) & 1u;
	}

#include "irr/irrpack.h"
	struct alignas(16) SAllDiffuse
	{
		//if flag decides to use alpha texture, alpha_u.texData is tex data for VT
		//otherwise alpha_u.constant_f32 is constant single-float alpha
		STextureDataOrConstant alpha;
		STextureDataOrConstant reflectance;
		STextureDataOrConstant opacity;
		//multiplication factor for texture samples
		//RGB19E7 format
		//.x - alpha scale, .y - reflectance scale, .z - opacity scale
		uint64_t textureScale;
	} PACK_STRUCT;
	struct alignas(16) SDiffuseTransmitter
	{
		STextureDataOrConstant transmittance;
		STextureDataOrConstant opacity;
		//multiplication factor for texture samples
		//[0] - transmittance scale, [1] - opacity scale
		float textureScale[2];
	} PACK_STRUCT;
	struct alignas(16) SAllDielectric
	{
		float eta;
		//if NDF is Ashikhmin-Shirley:
		//	if flag decides to use alpha_u texture, alpha_u.texData is tex data for VT
		//	otherwise alpha_u.constant_f32 is constant single-float alpha_u
		//	if flag decides to use alpha_v texture, alpha_v.texData is tex data for VT
		//	otherwise alpha_v.constant_f32 is constant single-float alpha_v
		//otherwise (different NDF)
		//	if flag decides to use alpha texture, alpha_u.texData is tex data for VT
		//	otherwise alpha_u.constant_f32 is constant single-float alpha
		STextureDataOrConstant alpha_u;
		STextureDataOrConstant alpha_v;
		STextureDataOrConstant opacity;
		//multiplication factor for texture samples
		//RGB19E7 format
		//.x - alpha_u scale, .y - alpha_v scale, .z - opacity scale
		uint64_t textureScale;
	} PACK_STRUCT;
	struct alignas(16) SAllConductor
	{
		//same as for SAllDielectric::alpha_u,alpha_v
		STextureDataOrConstant alpha_u;
		STextureDataOrConstant alpha_v;
		//ior[0] real part of eta in RGB19E7 format
		//ior[1] is imaginary part of eta in RGB19E7 format
		uint64_t eta[2];
		STextureDataOrConstant opacity;
		//multiplication factor for texture samples
		//RGB19E7 format
		//.x - alpha_u scale, .y - alpha_v scale, .z - opacity scale
		uint64_t textureScale;
	} PACK_STRUCT;
	struct alignas(16) SAllPlastic
	{
		float eta;
		STextureDataOrConstant alpha;
		STextureDataOrConstant opacity;
		STextureDataOrConstant reflectance;
		//multiplication factor for texture samples
		//RGB19E7 format
		//.x - alpha scale,.y - opacity scale, .z - refl scale
		uint64_t textureScale;
	} PACK_STRUCT;
	struct alignas(16) SAllCoating
	{
		//thickness and eta encoded as 2x float16, thickness on bits 0:15, eta on bits 16:31
		uint32_t thickness_eta;
		STextureDataOrConstant alpha;
		//rgb in RGB19E7 format or texture data for VT (flag decides)
		STextureDataOrConstant sigmaA;
		STextureDataOrConstant opacity;
		//multiplication factor for texture samples
		//RGB19E7 format
		//.x - alpha scale, .y - sigmaA scale, .z - opacity scale
		uint64_t textureScale;
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
		STextureDataOrConstant alpha_u;
		STextureDataOrConstant alpha_v;
		STextureDataOrConstant opacity;
		//multiplication factor for texture samples
		//RGB19E7 format
		//.x - alpha u scale, .y - alpha v scale, .z - opacity scale
		uint64_t textureScale;
	} PACK_STRUCT;
	struct alignas(16) SBumpMap
	{
		//texture data for VT
		STextureData bumpmap;
		float textureScale;
	} PACK_STRUCT;
	struct alignas(16) SBlend
	{
		//per-channel single-float32 blend factor
		//2 weights in order to encode MIXTURE bsdf as tree of BLENDs
		//if flag decides to use weight texture, `weightL.texData` is texture data for VT and `weightR` is then irrelevant
		//otherwise `weightL.constant_f32` and `weightR` are constant float32 blend weights. Left has to be multiplied by weightL and right operand has to be multiplied by weightR.
		STextureDataOrConstant weightL;
		float weightR;
		float textureScale;
	} PACK_STRUCT;
#include "irr/irrunpack.h"

	union SBSDFUnion
	{
		SBSDFUnion() : bumpmap{STextureData::invalid()} {}

		SAllDiffuse diffuse;
		SDiffuseTransmitter diffuseTransmitter;
		SAllDielectric dielectric;
		SAllConductor conductor;
		SAllPlastic plastic;
		SAllCoating coating;
		SBumpMap bumpmap;
		SWard ward;
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
		asset::IAssetManager* m_manager;

#include "irr/irrpack.h"
		//compatible with std430 and std140
		struct SInstanceData
		{
			core::matrix3x4SIMD tform;//mat4x3
			core::matrix3x4SIMD normalMatrix;//mat4x3
			uint64_t emissive;//uvec2, rgb19e7
		} PACK_STRUCT;
#include "irr/irrunpack.h"
		struct SContext
		{
			SContext(
				const asset::IGeometryCreator* _geomCreator,
				const asset::IMeshManipulator* _manipulator,
				const asset::IAssetLoader::SAssetLoadParams& _params,
				asset::IAssetLoader::IAssetLoaderOverride* _override,
				CGlobalMitsubaMetadata* _metadata
			);

			const asset::IGeometryCreator* creator;
			const asset::IMeshManipulator* manipulator;
			const asset::IAssetLoader::SAssetLoadParams params;
			asset::IAssetLoader::IAssetLoaderOverride* override;
			CGlobalMitsubaMetadata* globalMeta;

			_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_PAGE_TABLE_LAYERS = 64u;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_PAGE_SZ_LOG2 = 7u;//128
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_PHYSICAL_PAGE_TEX_TILES_PER_DIM_LOG2 = 4u;//16
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_PHYSICAL_PAGE_TEX_LAYERS = 20u;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_PAGE_PADDING = 8u;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_MAX_ALLOCATABLE_TEX_SZ_LOG2 = 12u;//4096

			core::smart_refctd_ptr<asset::ICPUVirtualTexture> VT;

			//
			using group_ass_type = core::smart_refctd_ptr<asset::ICPUMesh>;
			core::map<const CElementShape::ShapeGroup*, group_ass_type> groupCache;
			//
			using shape_ass_type = core::smart_refctd_ptr<asset::ICPUMesh>;
			core::map<const CElementShape*, shape_ass_type> shapeCache;
			//image, sampler, scale
			using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>,core::smart_refctd_ptr<asset::ICPUSampler>,float>;
			core::unordered_map<const CElementTexture*, tex_ass_type> textureCache;
			using VT_data_type = std::pair<bsdf::STextureData, float>;
			core::unordered_map<const CElementTexture*, VT_data_type> VTallocDataCache;

			core::vector<bsdf::SBSDFUnion> bsdfBuffer;
			core::vector<bsdf::instr_t> instrBuffer;

			struct bsdf_type
			{
				using instr_offset_count = std::pair<uint32_t, uint32_t>;

				instr_offset_count postorder;
				instr_offset_count preorder;
			};
			//caches instr buffer instr-wise offset (.first) and instruction count (.second) for each bsdf node
			core::unordered_map<const CElementBSDF*,bsdf_type> instrStreamCache;

			struct SPipelineCacheKey
			{
				asset::SVertexInputParams vtxParams;
				asset::SPrimitiveAssemblyParams primParams;

				inline bool operator==(const SPipelineCacheKey& rhs) const
				{
					return memcmp(&vtxParams, &rhs.vtxParams, sizeof(vtxParams))==0 && memcmp(&primParams, &rhs.primParams, sizeof(primParams))==0;
				}

				struct hash
				{
					inline size_t operator()(const SPipelineCacheKey& k) const
					{
						constexpr size_t BYTESZ = sizeof(k.vtxParams) + sizeof(k.primParams);
						uint8_t mem[BYTESZ]{};
						uint8_t* ptr = mem;
						memcpy(ptr, &k.vtxParams, sizeof(k.vtxParams));
						ptr += sizeof(k.vtxParams);
						memcpy(ptr, &k.primParams, sizeof(k.primParams));
						ptr += sizeof(k.primParams);

						return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(mem), BYTESZ));
					}
				};
			};
			core::unordered_map<SPipelineCacheKey, core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>, SPipelineCacheKey::hash> pipelineCache;

#include "irr/irrpack.h"
			struct SBSDFDataCacheKey
			{
				const CElementBSDF* bsdf;
				const CElementBSDF* maskParent;
				float mix2blend_weight;

				inline bool operator==(const SBSDFDataCacheKey& rhs) const { return memcmp(this,&rhs,sizeof(*this))==0; }
				struct hash
				{
					inline size_t operator()(const SBSDFDataCacheKey& k) const
					{
						return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(&k), sizeof(k)));
					}
				};
			} PACK_STRUCT;
#include "irr/irrunpack.h"
			//caches indices/offsets into `bsdfBuffer`
			core::unordered_map<SBSDFDataCacheKey, uint32_t, SBSDFDataCacheKey::hash> bsdfDataCache;
		};

		//! Destructor
		virtual ~CMitsubaLoader() = default;

		//
		SContext::shape_ass_type				getMesh(SContext& ctx, uint32_t hierarchyLevel, CElementShape* shape);
		SContext::group_ass_type				loadShapeGroup(SContext& ctx, uint32_t hierarchyLevel, const CElementShape::ShapeGroup* shapegroup);
		SContext::shape_ass_type				loadBasicShape(SContext& ctx, uint32_t hierarchyLevel, CElementShape* shape);
		
		SContext::tex_ass_type					getTexture(SContext& ctx, uint32_t hierarchyLevel, const CElementTexture* texture);
		SContext::VT_data_type					getVTallocData(SContext& ctx, const CElementTexture* texture, uint32_t texHierLvl);

		bsdf::SBSDFUnion bsdfNode2bsdfStruct(SContext& _ctx, const CElementBSDF* _node, uint32_t _texHierLvl, float _mix2blend_weight = 0.f, const CElementBSDF* _parentMask = nullptr);
		SContext::bsdf_type getBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf);
		SContext::bsdf_type genBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf);

		template <typename Iter>
		core::smart_refctd_ptr<asset::ICPUDescriptorSet> createDS0(const SContext& _ctx, Iter meshBegin, Iter meshEnd);

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
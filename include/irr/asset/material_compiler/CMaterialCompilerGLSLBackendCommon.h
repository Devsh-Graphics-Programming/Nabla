#ifndef __C_MITSUBA_MATERIAL_COMPILER_RAYTRACING_BACKEND_H_INCLUDED__
#define __C_MITSUBA_MATERIAL_COMPILER_RAYTRACING_BACKEND_H_INCLUDED__

#include <irr/asset/material_compiler/IR.h>
#include <irr/asset/ICPUVirtualTexture.h>
#include "os.h"

namespace irr
{
namespace asset
{
namespace material_compiler
{

namespace instr_stream
{
	using instr_t = uint64_t;

	enum E_OPCODE : uint8_t
	{
		//brdf
		OP_DIFFUSE,
		OP_CONDUCTOR,
		OP_PLASTIC,
		OP_COATING,
		//bsdf
		OP_DIFFTRANS,
		OP_DIELECTRIC,
		//blend
		OP_BLEND,
		//specials
		OP_BUMPMAP,
		OP_SET_GEOM_NORMAL,
		OP_INVALID,
		OP_NOOP,

		OPCODE_COUNT
	};

	//bitfields common for all or more than 1 instruction stream
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_OPCODE_WIDTH = 4u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_OPCODE_MASK = 0xfu;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_OPCODE_SHIFT = 0u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BITFIELDS_SHIFT = INSTR_OPCODE_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BITFIELDS_WIDTH = 9u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_REFL_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_REFL_TEX = INSTR_OPCODE_WIDTH + 5u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_ALPHA_U_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_ALPHA_U_TEX = INSTR_OPCODE_WIDTH + 0u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_ALPHA_V_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_ALPHA_V_TEX = INSTR_OPCODE_WIDTH + 3u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_SPEC_TRANS_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_TRANS_TEX = INSTR_OPCODE_WIDTH + 0u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_NDF = 0x3u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_WIDTH_NDF = 2u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_NDF = INSTR_OPCODE_WIDTH + 1u;

	//_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_FAST_APPROX = 0x1u;
	//_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_WIDTH_FAST_APPROX = 1u;
	//_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_FAST_APPROX = INSTR_OPCODE_WIDTH + 1u;

	//_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_NONLINEAR = 0x1u;
	//_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_WIDTH_NONLINEAR = 1u;
	//_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_NONLINEAR = INSTR_OPCODE_WIDTH + 4u;

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

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_BSDF_BUF_OFFSET_MASK = 0x7ffffu;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_BSDF_BUF_OFFSET_WIDTH = 19u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_BSDF_BUF_OFFSET_SHIFT = INSTR_OPCODE_WIDTH + INSTR_BITFIELDS_WIDTH;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_NORMAL_ID_WIDTH = 8u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_NORMAL_ID_MASK = 0xffu;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_NORMAL_ID_SHIFT = 56u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BUMPMAP_SRC_REG_WIDTH = 8u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BUMPMAP_SRC_REG_MASK = 0xffu;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BUMPMAP_SRC_REG_SHIFT = 40u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t MAX_REGISTER_COUNT = 72u;

	enum E_NDF
	{
		NDF_BECKMANN	= 0b00,
		NDF_GGX			= 0b01,
		NDF_PHONG		= 0b10,
		NDF_AS			= 0b11
	};

	inline E_OPCODE getOpcode(const instr_t& i)
	{
		return static_cast<E_OPCODE>(core::bitfieldExtract(i, INSTR_OPCODE_SHIFT, INSTR_OPCODE_WIDTH));
	}

	inline bool isTwosided(const instr_t& i)
	{
		return static_cast<bool>( core::bitfieldExtract(i, BITFIELDS_SHIFT_TWOSIDED, 1) );
	}
	inline bool isMasked(const instr_t& i)
	{
		return static_cast<bool>( core::bitfieldExtract(i, BITFIELDS_SHIFT_MASKFLAG, 1) );
	}

	inline uint32_t getNormalId(const instr_t& i)
	{
		return i >> INSTR_NORMAL_ID_SHIFT;
	}

	using VTID = asset::ICPUVirtualTexture::SMasterTextureData;
#include "irr/irrpack.h"
	struct STextureData {
		VTID vtid = VTID::invalid();
		union {
			uint32_t prefetch_reg;//uint32
			uint32_t scale = 0u;//float
		};

		bool operator==(const STextureData& rhs) const { return memcmp(this,&rhs,sizeof(rhs))==0; }
		struct hash
		{
			std::size_t operator()(const STextureData& t) const { return std::hash<uint64_t>{}(reinterpret_cast<const uint64_t&>(t.vtid)) ^ std::hash<uint32_t>{}(t.scale); }
		};
	} PACK_STRUCT;

	struct STextureOrConstant
	{
		void setConst(float f) { std::fill(constant, constant+3, reinterpret_cast<uint32_t&>(f)); }
		void setConst(const float* fv) { memcpy(constant, fv, sizeof(constant)); }
		void setTexture(const VTID& _vtid, float _scale)
		{
			tex.vtid = _vtid;
			core::uintBitsToFloat(tex.scale) = _scale;
		}

		union
		{
			STextureData tex;
			uint32_t constant[3];//3x float
		};
	} PACK_STRUCT;
#include "irr/irrunpack.h"

#include "irr/irrpack.h"
	struct SAllDiffuse
	{
		STextureOrConstant alpha;
		STextureOrConstant reflectance;
		STextureOrConstant opacity;
	} PACK_STRUCT;
	struct SDiffuseTransmitter
	{
		STextureOrConstant transmittance;
		STextureOrConstant dummy;
		STextureOrConstant opacity;
	} PACK_STRUCT;
	struct SAllDielectric
	{
		STextureOrConstant alpha_u;
		STextureOrConstant alpha_v;
		STextureOrConstant opacity;
		float eta;
	} PACK_STRUCT;
	struct SAllConductor
	{
		STextureOrConstant alpha_u;
		STextureOrConstant alpha_v;
		STextureOrConstant opacity;
		//3d complex IoR, rgb19e7 format, [0]=real, [1]=imaginary
		uint64_t eta[2];
	} PACK_STRUCT;
	struct SAllPlastic
	{
		STextureOrConstant alpha;
		STextureOrConstant reflectance;
		STextureOrConstant opacity;
		float eta;
	} PACK_STRUCT;
	struct SAllCoating
	{
		STextureOrConstant alpha;
		STextureOrConstant sigmaA;
		STextureOrConstant opacity;
		//thickness and eta encoded as 2x float16, thickness on bits 0:15, eta on bits 16:31
		uint32_t thickness_eta;
	} PACK_STRUCT;
	struct SBumpMap
	{
		//texture data for VT
		STextureData bumpmap;
	} PACK_STRUCT;
	struct SBlend
	{
		STextureOrConstant weight;
	} PACK_STRUCT;
#include "irr/irrunpack.h"

	union alignas(16) SBSDFUnion
	{
		_IRR_STATIC_INLINE_CONSTEXPR size_t MAX_TEXTURES = 3ull;

		SBSDFUnion() : bumpmap{} {}

		SAllDiffuse diffuse;
		SDiffuseTransmitter difftrans;
		SAllDielectric dielectric;
		SAllConductor conductor;
		SAllPlastic plastic;
		SAllCoating coating;
		SBumpMap bumpmap;
		SBlend blend;
		STextureOrConstant param[MAX_TEXTURES];
	};

	struct SContext
	{
		core::unordered_map<const IR::INode*, size_t> bsdfDataIndexMap;
		core::vector<instr_stream::SBSDFUnion> bsdfData;

		using VTallocKey = std::pair<const asset::ICPUImageView*, const asset::ICPUSampler*>;
		struct VTallocKeyHash
		{
			inline std::size_t operator() (const VTallocKey& k) const
			{
				return std::hash<VTallocKey::first_type>{}(k.first) ^ std::hash<VTallocKey::second_type>{}(k.second);
			}
		};
		core::unordered_map<VTallocKey, VTID, VTallocKeyHash> VTallocMap;

		core::smart_refctd_ptr<asset::ICPUVirtualTexture> vt;
	};

	using traversal_t = core::vector<instr_t>;

namespace remainder_and_pdf
{
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_WIDTH = 8u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_MASK = 0xffu;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_DST_SHIFT = BITFIELDS_BSDF_BUF_OFFSET_SHIFT + BITFIELDS_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH*0u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_SRC1_SHIFT = BITFIELDS_BSDF_BUF_OFFSET_SHIFT + BITFIELDS_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH*1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_SRC2_SHIFT = BITFIELDS_BSDF_BUF_OFFSET_SHIFT + BITFIELDS_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH*2u;
}
namespace gen_choice
{
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_RIGHT_JUMP_WIDTH = 8u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_RIGHT_JUMP_MASK = 0xffu;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_RIGHT_JUMP_SHIFT = BITFIELDS_BSDF_BUF_OFFSET_SHIFT + BITFIELDS_BSDF_BUF_OFFSET_WIDTH;
}
namespace tex_prefetch
{
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_0_SHIFT = BITFIELDS_SHIFT_NDF;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_1_SHIFT = BITFIELDS_SHIFT_NDF+1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_2_SHIFT = BITFIELDS_SHIFT_TWOSIDED;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_COUNT = 3u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_REG_CNT_WIDTH = 2u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT = INSTR_NORMAL_ID_SHIFT;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_1_REG_CNT_SHIFT = BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT + BITFIELDS_FETCH_TEX_REG_CNT_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_2_REG_CNT_SHIFT = BITFIELDS_FETCH_TEX_1_REG_CNT_SHIFT + BITFIELDS_FETCH_TEX_REG_CNT_WIDTH;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_0_SHIFT = remainder_and_pdf::INSTR_REG_DST_SHIFT;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_1_SHIFT = remainder_and_pdf::INSTR_REG_SRC1_SHIFT;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_2_SHIFT = remainder_and_pdf::INSTR_REG_SRC2_SHIFT;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_WIDTH = remainder_and_pdf::INSTR_REG_WIDTH;

	inline uint32_t getTexFetchFlags(instr_t i)
	{
		auto flags = core::bitfieldExtract(i, BITFIELDS_FETCH_TEX_0_SHIFT, 1);
		flags |= core::bitfieldExtract(i, BITFIELDS_FETCH_TEX_1_SHIFT, 1)<<1;
		flags |= core::bitfieldExtract(i, BITFIELDS_FETCH_TEX_2_SHIFT, 1)<<2;

		return static_cast<uint32_t>(flags);
	}
}
namespace normal_precomp
{
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_DST_SHIFT = remainder_and_pdf::INSTR_REG_DST_SHIFT;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_WIDTH = remainder_and_pdf::INSTR_REG_WIDTH;
}
}

class CMaterialCompilerGLSLBackendCommon
{
protected:
	core::unordered_map<uint32_t, uint32_t> createBsdfDataIndexMapForPrefetchedTextures(instr_stream::SContext* _ctx, const instr_stream::traversal_t& _tex_prefetch_stream, const core::unordered_map<instr_stream::STextureData, uint32_t, instr_stream::STextureData::hash>& _tex2reg) const;

	void adjustBSDFDataIndices(instr_stream::traversal_t& _stream, const core::unordered_map<uint32_t, uint32_t>& _ix2ix) const
	{
		using namespace instr_stream;
		for (instr_t& i : _stream) {
			auto found = _ix2ix.find(core::bitfieldExtract(i, BITFIELDS_BSDF_BUF_OFFSET_SHIFT, BITFIELDS_BSDF_BUF_OFFSET_WIDTH));
			if (found != _ix2ix.end())
				i = core::bitfieldInsert<instr_t>(i, found->second, BITFIELDS_BSDF_BUF_OFFSET_SHIFT, BITFIELDS_BSDF_BUF_OFFSET_WIDTH);
		}
	}
	void setSourceRegForBumpmaps(instr_stream::traversal_t& _stream, uint32_t _regNumOffset)
	{
		using namespace instr_stream;
		for (instr_t& i : _stream) {
			if (getOpcode(i)==OP_BUMPMAP)
			{
				const uint32_t n_id = getNormalId(i);
				i = core::bitfieldInsert<instr_t>(i, _regNumOffset+n_id, INSTR_BUMPMAP_SRC_REG_SHIFT, INSTR_BUMPMAP_SRC_REG_WIDTH);
			}
		}
	}

public:
	struct result_t
	{
		struct instr_streams_t
		{
			struct stream_t
			{
				uint32_t first;
				uint32_t count;
			};
			stream_t rem_and_pdf;
			stream_t gen_choice;
			stream_t tex_prefetch;
			stream_t norm_precomp;
		};

		instr_stream::traversal_t instructions;

		//TODO flags like alpha tex always present etc..
		bool noPrefetchStream;
		bool noNormPrecompStream;
		uint32_t usedRegisterCount;
		bool prefetch_sameNumOfChannels;
		uint32_t prefetch_numOfChannels;

		core::unordered_set<instr_stream::E_OPCODE> opcodes;

		//one element for each input IR root node
		core::unordered_map<const IR::INode*, instr_streams_t> streams;

		void debugPrint(const instr_streams_t& _streams, const instr_stream::SContext* _ctx) const;
		//common for rem_and_pdf and gen_choice instructions
		void debugPrintInstr(instr_stream::instr_t instr, const instr_stream::SContext* _ctx) const;
	};

	result_t compile(instr_stream::SContext* _ctx, IR* _ir, bool _computeGenChoiceStream = true);
};

}}}

#endif
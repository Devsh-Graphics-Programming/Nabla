#ifndef __C_MITSUBA_MATERIAL_COMPILER_GLSL_BACKEND_COMMON_H_INCLUDED__
#define __C_MITSUBA_MATERIAL_COMPILER_GLSL_BACKEND_COMMON_H_INCLUDED__

#include <irr/asset/material_compiler/IR.h>
#include <irr/asset/ICPUVirtualTexture.h>

#include <ostream>

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

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_1ST_PARAM_TEX = INSTR_OPCODE_WIDTH + 0u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_2ND_PARAM_TEX = INSTR_OPCODE_WIDTH + 3u;
	//3rd param is always opacity

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_REFL_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_REFL_TEX = INSTR_OPCODE_WIDTH + 3u;
	static_assert(BITFIELDS_SHIFT_REFL_TEX==BITFIELDS_SHIFT_2ND_PARAM_TEX);

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_ALPHA_U_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_ALPHA_U_TEX = INSTR_OPCODE_WIDTH + 0u;
	static_assert(BITFIELDS_SHIFT_ALPHA_U_TEX==BITFIELDS_SHIFT_1ST_PARAM_TEX);

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_ALPHA_V_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_ALPHA_V_TEX = INSTR_OPCODE_WIDTH + 3u;
	static_assert(BITFIELDS_SHIFT_ALPHA_V_TEX==BITFIELDS_SHIFT_2ND_PARAM_TEX);

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_SPEC_TRANS_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_TRANS_TEX = INSTR_OPCODE_WIDTH + 0u;
	static_assert(BITFIELDS_SHIFT_TRANS_TEX==BITFIELDS_SHIFT_1ST_PARAM_TEX);

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
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_SIGMA_A_TEX = INSTR_OPCODE_WIDTH + 3u;
	static_assert(BITFIELDS_SHIFT_SIGMA_A_TEX==BITFIELDS_SHIFT_2ND_PARAM_TEX);

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_WEIGHT_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_WEIGHT_TEX = INSTR_OPCODE_WIDTH + 0u;
	static_assert(BITFIELDS_SHIFT_WEIGHT_TEX==BITFIELDS_SHIFT_1ST_PARAM_TEX);

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
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_2_REG_CNT_MASK = 0b11;
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
	struct result_t;
	struct SContext;
protected:
	_IRR_STATIC_INLINE_CONSTEXPR const char* OPCODE_NAMES[OPCODE_COUNT]{
		"OP_DIFFUSE",
		"OP_CONDUCTOR",
		"OP_PLASTIC",
		"OP_COATING",
		"OP_DIFFTRANS",
		"OP_DIELECTRIC",
		"OP_BLEND",
		"OP_BUMPMAP",
		"OP_SET_GEOM_NORMAL",
		"OP_INVALID",
		"OP_NOOP"
	};
	static std::string genPreprocDefinitions(const result_t& _res)
	{
		using namespace std::string_literals;

		std::string defs;
		defs += "\n#define REG_COUNT " + std::to_string(_res.usedRegisterCount);

		for (E_OPCODE op : _res.opcodes)
			defs += "\n#define "s + OPCODE_NAMES[op] + " " + std::to_string(op);
		defs += "\n#define OP_MAX_BRDF " + std::to_string(OP_COATING);
		defs += "\n#define OP_MAX_BSDF " + std::to_string(OP_DIELECTRIC);

		defs += "\n#define NDF_BECKMANN " + std::to_string(NDF_BECKMANN);
		defs += "\n#define NDF_GGX " + std::to_string(NDF_GGX);
		defs += "\n#define NDF_PHONG " + std::to_string(NDF_PHONG);
		defs += "\n#define NDF_AS " + std::to_string(NDF_AS);

		constexpr size_t size_of_uvec4 = 16ull;
		defs += "\n#define sizeof_bsdf_data " + std::to_string((sizeof(SBSDFUnion)+size_of_uvec4-1u)/size_of_uvec4);

		if (!_res.noPrefetchStream)
			defs += "\n#define TEX_PREFETCH_STREAM";
		if (!_res.noNormPrecompStream)
			defs += "\n#define NORM_PRECOMP_STREAM";
		if (_res.prefetch_sameNumOfChannels)
		{
			assert(_res.prefetch_numOfChannels==1u || _res.prefetch_numOfChannels==3u);
			if (_res.prefetch_numOfChannels==1u)
				defs += "\n#define PREFETCH_REGS_ALWAYS_1";
			else if (_res.prefetch_numOfChannels==3u)
				defs += "\n#define PREFETCH_REGS_ALWAYS_3";
		}

		//instruction bitfields
		defs += "\n#define INSTR_OPCODE_MASK " + std::to_string(INSTR_OPCODE_MASK);
		defs += "\n#define INSTR_BSDF_BUF_OFFSET_SHIFT " + std::to_string(BITFIELDS_BSDF_BUF_OFFSET_SHIFT);
		defs += "\n#define INSTR_BSDF_BUF_OFFSET_MASK " + std::to_string(BITFIELDS_BSDF_BUF_OFFSET_MASK);
		defs += "\n#define INSTR_NDF_SHIFT " + std::to_string(BITFIELDS_SHIFT_NDF);
		defs += "\n#define INSTR_NDF_MASK " + std::to_string(BITFIELDS_MASK_NDF);
		defs += "\n#define INSTR_ALPHA_U_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_ALPHA_U_TEX);
		defs += "\n#define INSTR_ALPHA_V_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_ALPHA_V_TEX);
		defs += "\n#define INSTR_REFL_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_REFL_TEX);
		defs += "\n#define INSTR_TRANS_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_TRANS_TEX);
		defs += "\n#define INSTR_SIGMA_A_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_SIGMA_A_TEX);
		defs += "\n#define INSTR_WEIGHT_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_WEIGHT_TEX);
		defs += "\n#define INSTR_TWOSIDED_SHIFT " + std::to_string(BITFIELDS_SHIFT_TWOSIDED);
		defs += "\n#define INSTR_MASKFLAG_SHIFT " + std::to_string(BITFIELDS_SHIFT_MASKFLAG);
		defs += "\n#define INSTR_1ST_PARAM_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_1ST_PARAM_TEX);
		defs += "\n#define INSTR_2ND_PARAM_TEX_SHIFT " + std::to_string(BITFIELDS_SHIFT_2ND_PARAM_TEX);
		defs += "\n#define INSTR_NORMAL_ID_SHIFT " + std::to_string(INSTR_NORMAL_ID_SHIFT);
		defs += "\n#define INSTR_NORMAL_ID_MASK " + std::to_string(INSTR_NORMAL_ID_MASK);
		//remainder_and_pdf
		{
			using namespace remainder_and_pdf;

			defs += "\n#define INSTR_REG_MASK " + std::to_string(INSTR_REG_MASK);
			defs += "\n#define INSTR_REG_DST_SHIFT " + std::to_string(INSTR_REG_DST_SHIFT);
			defs += "\n#define INSTR_REG_SRC1_SHIFT " + std::to_string(INSTR_REG_SRC1_SHIFT);
			defs += "\n#define INSTR_REG_SRC2_SHIFT " + std::to_string(INSTR_REG_SRC2_SHIFT);
		}
		//tex_prefetch
		{
			using namespace tex_prefetch;

			defs += "\n#define INSTR_FETCH_FLAG_TEX_0_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_0_SHIFT);
			defs += "\n#define INSTR_FETCH_FLAG_TEX_1_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_1_SHIFT);
			defs += "\n#define INSTR_FETCH_FLAG_TEX_2_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_2_SHIFT);

			defs += "\n#define INSTR_FETCH_TEX_0_REG_CNT_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT);
			defs += "\n#define INSTR_FETCH_TEX_1_REG_CNT_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_1_REG_CNT_SHIFT);
			defs += "\n#define INSTR_FETCH_TEX_2_REG_CNT_SHIFT " + std::to_string(BITFIELDS_FETCH_TEX_2_REG_CNT_SHIFT);
			defs += "\n#define INSTR_FETCH_TEX_REG_CNT_MASK " + std::to_string(BITFIELDS_FETCH_TEX_2_REG_CNT_MASK);
		}
		defs += "\n";
	}

	core::unordered_map<uint32_t, uint32_t> createBsdfDataIndexMapForPrefetchedTextures(SContext* _ctx, const instr_stream::traversal_t& _tex_prefetch_stream, const core::unordered_map<instr_stream::STextureData, uint32_t, instr_stream::STextureData::hash>& _tex2reg) const;

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

	//common for rem_and_pdf and gen_choice instructions
	void debugPrintInstr(std::ostream& _out, instr_stream::instr_t instr, const result_t& _res, const SContext* _ctx) const;

public:
	struct SContext
	{
		//users should not touch this
		core::vector<instr_stream::SBSDFUnion>* pBsdfData;
		core::unordered_map<const IR::INode*, size_t> bsdfDataIndexMap;

		using VTallocKey = std::pair<const asset::ICPUImageView*, const asset::ICPUSampler*>;
		struct VTallocKeyHash
		{
			inline std::size_t operator() (const VTallocKey& k) const
			{
				return std::hash<VTallocKey::first_type>{}(k.first) ^ std::hash<VTallocKey::second_type>{}(k.second);
			}
		};
		core::unordered_map<VTallocKey, instr_stream::VTID, VTallocKeyHash> VTallocMap;

		//must be initialized by user
		core::smart_refctd_ptr<asset::ICPUVirtualTexture> vt;
	};

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
		core::vector<instr_stream::SBSDFUnion> bsdfData;

		//TODO flags like alpha tex always present etc..
		bool noPrefetchStream;
		bool noNormPrecompStream;
		uint32_t usedRegisterCount;
		bool prefetch_sameNumOfChannels;
		uint32_t prefetch_numOfChannels;

		core::unordered_set<instr_stream::E_OPCODE> opcodes;

		//one element for each input IR root node
		core::unordered_map<const IR::INode*, instr_streams_t> streams;

		std::string fragmentShaderSource;
	};

	void debugPrint(std::ostream& _out, const result_t::instr_streams_t& _streams, const result_t& _res, const SContext* _ctx) const;

	result_t compile(SContext* _ctx, IR* _ir, bool _computeGenChoiceStream = true);
};

}}}

#endif
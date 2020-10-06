// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_MITSUBA_MATERIAL_COMPILER_GLSL_BACKEND_COMMON_H_INCLUDED__
#define __NBL_ASSET_C_MITSUBA_MATERIAL_COMPILER_GLSL_BACKEND_COMMON_H_INCLUDED__

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
		OP_THINDIELECTRIC,
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
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_3RD_PARAM_TEX = INSTR_OPCODE_WIDTH + 8u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_4TH_PARAM_TEX = INSTR_OPCODE_WIDTH + 4u;


	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_OPACITY_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_OPACITY_TEX = BITFIELDS_SHIFT_3RD_PARAM_TEX;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_REFL_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_REFL_TEX = BITFIELDS_SHIFT_4TH_PARAM_TEX;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_ALPHA_U_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_ALPHA_U_TEX = BITFIELDS_SHIFT_1ST_PARAM_TEX;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_ALPHA_V_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_ALPHA_V_TEX = BITFIELDS_SHIFT_2ND_PARAM_TEX;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_SPEC_TRANS_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_TRANS_TEX = BITFIELDS_SHIFT_1ST_PARAM_TEX;

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
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_SIGMA_A_TEX = BITFIELDS_SHIFT_4TH_PARAM_TEX;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_WEIGHT_TEX = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_WEIGHT_TEX = BITFIELDS_SHIFT_1ST_PARAM_TEX;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_TWOSIDED = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_TWOSIDED = INSTR_OPCODE_WIDTH + 6u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_MASK_MASKFLAG = 0x1u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_MASKFLAG = INSTR_OPCODE_WIDTH + 7u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_BSDF_BUF_OFFSET_SHIFT = INSTR_OPCODE_WIDTH + INSTR_BITFIELDS_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_BSDF_BUF_OFFSET_WIDTH = 32u-BITFIELDS_BSDF_BUF_OFFSET_SHIFT;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_BSDF_BUF_OFFSET_MASK = (1u<<BITFIELDS_BSDF_BUF_OFFSET_WIDTH) - 1u;

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

		NDF_COUNT = 3
	};

	inline bool opHasSpecular(E_OPCODE op)
	{
		switch (op)
		{
		case OP_DIELECTRIC:
		case OP_CONDUCTOR:
		case OP_PLASTIC:
		case OP_COATING:
			return true;
		default: return false;
		}
	}

	inline E_OPCODE getOpcode(const instr_t& i)
	{
		return static_cast<E_OPCODE>(core::bitfieldExtract(i, INSTR_OPCODE_SHIFT, INSTR_OPCODE_WIDTH));
	}

	inline E_NDF getNDF(const instr_t& i)
	{
		return static_cast<E_NDF>(core::bitfieldExtract(i, BITFIELDS_SHIFT_NDF, BITFIELDS_WIDTH_NDF));
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

	inline uint32_t getBSDFDataIx(const instr_t& i)
	{
		return static_cast<uint32_t>(
			core::bitfieldExtract(i, BITFIELDS_BSDF_BUF_OFFSET_SHIFT, BITFIELDS_BSDF_BUF_OFFSET_WIDTH)
		);
	}

	inline void setBSDFDataIx(instr_t& i, uint32_t ix)
	{
		i = core::bitfieldInsert<instr_t>(i, ix, BITFIELDS_BSDF_BUF_OFFSET_SHIFT, BITFIELDS_BSDF_BUF_OFFSET_WIDTH);
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
		STextureOrConstant dummy;
		STextureOrConstant opacity;
		STextureOrConstant reflectance;
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
		STextureOrConstant dummy;
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
		STextureOrConstant alpha_u;
		STextureOrConstant alpha_v;
		STextureOrConstant opacity;
		STextureOrConstant reflectance;
		float eta;
	} PACK_STRUCT;
	struct SAllCoating
	{
		STextureOrConstant alpha_u;
		STextureOrConstant alpha_v;
		STextureOrConstant opacity;
		STextureOrConstant sigmaA;
		//thickness and eta encoded as 2x float16, thickness on bits 0:15, eta on bits 16:31
		uint32_t thickness_eta;
	} PACK_STRUCT;
	struct SBumpMap
	{
		//texture data for VT
		STextureData derivmap;
	} PACK_STRUCT;
	struct SBlend
	{
		STextureOrConstant weight;
	} PACK_STRUCT;
#include "irr/irrunpack.h"

#define _TEXTURE_INDEX(s,m) offsetof(s,m)/sizeof(STextureOrConstant)
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t ALPHA_U_TEX_IX = _TEXTURE_INDEX(SAllDielectric,alpha_u);
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t ALPHA_V_TEX_IX = _TEXTURE_INDEX(SAllDielectric, alpha_v);
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t REFLECTANCE_TEX_IX = _TEXTURE_INDEX(SAllDiffuse, reflectance);
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t TRANSMITTANCE_TEX_IX = _TEXTURE_INDEX(SDiffuseTransmitter, transmittance);
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t SIGMA_A_TEX_IX = _TEXTURE_INDEX(SAllCoating, sigmaA);
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t WEIGHT_TEX_IX = _TEXTURE_INDEX(SBlend, weight);
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t OPACITY_TEX_IX = _TEXTURE_INDEX(SAllDiffuse, opacity);
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t DERIV_MAP_TEX_IX = _TEXTURE_INDEX(SBumpMap, derivmap);

#undef _TEXTURE_INDEX

	constexpr size_t sizeof_uvec4 = 16ull;
	union alignas(sizeof_uvec4) SBSDFUnion
	{
		_IRR_STATIC_INLINE_CONSTEXPR size_t MAX_TEXTURES = 4ull;

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
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_3_SHIFT = BITFIELDS_SHIFT_MASKFLAG;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_COUNT = 4u;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_WIDTH = 6u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_0_SHIFT = 32u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_1_SHIFT = BITFIELDS_REG_0_SHIFT + BITFIELDS_REG_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_2_SHIFT = BITFIELDS_REG_1_SHIFT + BITFIELDS_REG_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_3_SHIFT = BITFIELDS_REG_2_SHIFT + BITFIELDS_REG_WIDTH;

	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_REG_CNT_WIDTH = 2u;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_REG_CNT_MASK = 0b11;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT = BITFIELDS_REG_3_SHIFT + BITFIELDS_REG_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_1_REG_CNT_SHIFT = BITFIELDS_FETCH_TEX_0_REG_CNT_SHIFT + BITFIELDS_FETCH_TEX_REG_CNT_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_2_REG_CNT_SHIFT = BITFIELDS_FETCH_TEX_1_REG_CNT_SHIFT + BITFIELDS_FETCH_TEX_REG_CNT_WIDTH;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_FETCH_TEX_3_REG_CNT_SHIFT = BITFIELDS_FETCH_TEX_2_REG_CNT_SHIFT + BITFIELDS_FETCH_TEX_REG_CNT_WIDTH;
	static_assert(BITFIELDS_FETCH_TEX_3_REG_CNT_SHIFT+BITFIELDS_FETCH_TEX_REG_CNT_WIDTH <= 8u*sizeof(instr_t));

	inline uint32_t getTexFetchFlags(instr_t i)
	{
		auto flags = core::bitfieldExtract(i, BITFIELDS_FETCH_TEX_0_SHIFT, 1);
		flags |= core::bitfieldExtract(i, BITFIELDS_FETCH_TEX_1_SHIFT, 1)<<1;
		flags |= core::bitfieldExtract(i, BITFIELDS_FETCH_TEX_2_SHIFT, 1)<<2;
		flags |= core::bitfieldExtract(i, BITFIELDS_FETCH_TEX_3_SHIFT, 1)<<3;

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
public:
	struct result_t;
	struct SContext;

protected:
	_IRR_STATIC_INLINE_CONSTEXPR const char* OPCODE_NAMES[instr_stream::OPCODE_COUNT]{
		"OP_DIFFUSE",
		"OP_CONDUCTOR",
		"OP_PLASTIC",
		"OP_COATING",
		"OP_DIFFTRANS",
		"OP_DIELECTRIC",
		"OP_THINDIELECTRIC",
		"OP_BLEND",
		"OP_BUMPMAP",
		"OP_SET_GEOM_NORMAL",
		"OP_INVALID",
		"OP_NOOP"
	};
	_IRR_STATIC_INLINE_CONSTEXPR const char* NDF_NAMES[instr_stream::NDF_COUNT]{
		"NDF_BECKMANN",
		"NDF_GGX",
		"NDF_PHONG"
	};
	static std::string genPreprocDefinitions(const result_t& _res, bool _genChoiceStream);

	core::unordered_map<uint32_t, uint32_t> createBsdfDataIndexMapForPrefetchedTextures(SContext* _ctx, const instr_stream::traversal_t& _tex_prefetch_stream, const core::unordered_map<instr_stream::STextureData, uint32_t, instr_stream::STextureData::hash>& _tex2reg) const;

	void adjustBSDFDataIndices(instr_stream::traversal_t& _stream, const core::unordered_map<uint32_t, uint32_t>& _ix2ix) const
	{
		using namespace instr_stream;
		for (instr_t& i : _stream) {
			auto found = _ix2ix.find(getBSDFDataIx(i));
			if (found != _ix2ix.end())
				setBSDFDataIx(i, found->second);
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
		bool allIsotropic;
		bool noTwosided;
		uint32_t usedRegisterCount;
		uint32_t globalPrefetchFlags;
		uint32_t globalPrefetchRegCountFlags;

		core::unordered_set<instr_stream::E_OPCODE> opcodes;
		core::unordered_set<instr_stream::E_NDF> NDFs;

		//one element for each input IR root node
		core::unordered_map<const IR::INode*, instr_streams_t> streams;

		//has to go after #version and before required user-provided descriptors and functions
		std::string fragmentShaderSource_declarations;
		//has to go after required user-provided descriptors and functions and before the rest of shader (especially entry point function)
		std::string fragmentShaderSource;
	};

	void debugPrint(std::ostream& _out, const result_t::instr_streams_t& _streams, const result_t& _res, const SContext* _ctx) const;

	result_t compile(SContext* _ctx, IR* _ir, bool _computeGenChoiceStream = true);
};

}}}

#endif
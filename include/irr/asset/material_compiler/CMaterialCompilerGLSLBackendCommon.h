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

template <typename stack_el_t>
class ITraversalGenerator;

class CMaterialCompilerGLSLBackendCommon
{
public:
	struct instr_stream
	{
		using instr_t = uint64_t;

		using instr_id_t = uint32_t;

		enum E_OPCODE : uint8_t
		{
			//brdf
			OP_DIFFUSE,
			OP_CONDUCTOR,
			OP_MAX_BRDF = OP_CONDUCTOR,
			//bsdf
			OP_DIFFTRANS,
			OP_DIELECTRIC,
			OP_THINDIELECTRIC,
			OP_MAX_BSDF = OP_THINDIELECTRIC,
			//blend
			OP_COATING,
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
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_MAX_PARAMETER_COUNT = 4u;
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_SHIFT_PARAM_TEX[INSTR_MAX_PARAMETER_COUNT] = {
			BITFIELDS_SHIFT_1ST_PARAM_TEX,
			BITFIELDS_SHIFT_2ND_PARAM_TEX,
			BITFIELDS_SHIFT_3RD_PARAM_TEX,
			BITFIELDS_SHIFT_4TH_PARAM_TEX
		};

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

		_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_ID_WIDTH = 24u;
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_ID_MASK	 = 0xffffffu;
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_ID_SHIFT = 32u;

		_IRR_STATIC_INLINE_CONSTEXPR uint32_t MAX_REGISTER_COUNT = 72u;

		enum E_NDF
		{
			NDF_BECKMANN	= 0b00,
			NDF_GGX			= 0b01,
			NDF_PHONG		= 0b10,

			NDF_COUNT = 3
		};

		inline static bool opHasSpecular(E_OPCODE op)
		{
			switch (op)
			{
			case OP_DIELECTRIC:
			case OP_CONDUCTOR:
			case OP_COATING:
				return true;
			default: return false;
			}
		}

		inline static void setOpcode(instr_t& i, E_OPCODE op)
		{
			i = core::bitfieldInsert<instr_t>(i, static_cast<instr_t>(op), INSTR_OPCODE_SHIFT, INSTR_OPCODE_WIDTH);
		}

		inline static E_OPCODE getOpcode(const instr_t& i)
		{
			return static_cast<E_OPCODE>(core::bitfieldExtract(i, INSTR_OPCODE_SHIFT, INSTR_OPCODE_WIDTH));
		}

		inline static bool opIsBRDF(E_OPCODE op)
		{
			return op <= OP_MAX_BRDF;
		}

		inline static bool opIsBSDF(E_OPCODE op)
		{
			return op <= OP_MAX_BSDF && !opIsBRDF(op);
		}

		inline static E_NDF getNDF(const instr_t& i)
		{
			return static_cast<E_NDF>(core::bitfieldExtract(i, BITFIELDS_SHIFT_NDF, BITFIELDS_WIDTH_NDF));
		}

		inline static bool isTwosided(const instr_t& i)
		{
			return static_cast<bool>( core::bitfieldExtract(i, BITFIELDS_SHIFT_TWOSIDED, 1) );
		}
		inline static bool isMasked(const instr_t& i)
		{
			return static_cast<bool>( core::bitfieldExtract(i, BITFIELDS_SHIFT_MASKFLAG, 1) );
		}

		inline static uint32_t getNormalId(const instr_t& i)
		{
			return i >> INSTR_NORMAL_ID_SHIFT;
		}

		inline static uint32_t getBSDFDataIx(const instr_t& i)
		{
			return static_cast<uint32_t>(
				core::bitfieldExtract(i, BITFIELDS_BSDF_BUF_OFFSET_SHIFT, BITFIELDS_BSDF_BUF_OFFSET_WIDTH)
			);
		}

		inline static void setBSDFDataIx(instr_t& i, uint32_t ix)
		{
			i = core::bitfieldInsert<instr_t>(i, ix, BITFIELDS_BSDF_BUF_OFFSET_SHIFT, BITFIELDS_BSDF_BUF_OFFSET_WIDTH);
		}

		inline static instr_id_t getInstrId(const instr_t& i)
		{
			return core::bitfieldExtract(i, INSTR_ID_SHIFT, INSTR_ID_WIDTH);
		}

		inline static void setInstrId(instr_t& i, instr_id_t id)
		{
			i = core::bitfieldInsert<instr_t>(i, id, INSTR_ID_SHIFT, INSTR_ID_WIDTH);
		}

		// more like max param number plus one (includes dummies)
		inline static uint32_t getParamCount(E_OPCODE op)
		{
			switch (op)
			{
			case OP_DIFFUSE: [[fallthrough]];
			case OP_COATING:
				return 4u;
			case OP_CONDUCTOR: [[fallthrough]];
			case OP_DIFFTRANS: [[fallthrough]];
			case OP_DIELECTRIC: [[fallthrough]];
			case OP_THINDIELECTRIC:
				return 3u;
			case OP_BLEND: [[fallthrough]];
			case OP_BUMPMAP:
				return 1u;
			default: 0u;
			}
		}

		inline static uint32_t getRegisterCountForParameter(E_OPCODE op, uint32_t n)
		{
	#define SWITCH_REG_CNT_FOR_PARAM_NUM(rc0, rc1, rc2, rc3) \
	switch (n)\
	{\
	case 0u: return rc0;\
	case 1u: return rc1;\
	case 2u: return rc2;\
	case 3u: return rc3;\
	}

			switch (op)
			{
			case OP_DIFFUSE:
				SWITCH_REG_CNT_FOR_PARAM_NUM(1, 0, 3, 3)
				break;
			case OP_DIFFTRANS:
				SWITCH_REG_CNT_FOR_PARAM_NUM(3, 0, 3, 0)
				break;
			case OP_DIELECTRIC: [[fallthrough]];
			case OP_THINDIELECTRIC: [[fallthrough]];
			case OP_CONDUCTOR:
				SWITCH_REG_CNT_FOR_PARAM_NUM(1, 1, 3, 0)
				break;
			case OP_COATING:
				SWITCH_REG_CNT_FOR_PARAM_NUM(1, 1, 3, 3)
				break;
			case OP_BUMPMAP:
				SWITCH_REG_CNT_FOR_PARAM_NUM(2, 0, 0, 0)
				break;
			case OP_BLEND:
				SWITCH_REG_CNT_FOR_PARAM_NUM(1, 0, 0, 0)
				break;
			default:
				return 0u;
			}
			return 0u;
	#undef SWITCH_REG_CNT_FOR_PARAM_NUM
		}

		using VTID = asset::ICPUVirtualTexture::SMasterTextureData;
	#include "irr/irrpack.h"
		struct STextureData {
			STextureData() = default;

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

		_IRR_STATIC_INLINE_CONSTEXPR size_t sizeof_uvec4 = 16ull;

		struct intermediate
		{
			struct STextureOrConstant
			{
				void setConst(float f) { std::fill(constant, constant + 3, reinterpret_cast<uint32_t&>(f)); }
				void setConst(const float* fv) { memcpy(constant, fv, sizeof(constant)); }
				void setTexture(const VTID& _vtid, float _scale)
				{
					tex.vtid = _vtid;
					core::uintBitsToFloat(tex.scale) = _scale;
				}
				core::vector3df_SIMD getConst() const
				{
					auto ret = core::vector3df_SIMD(reinterpret_cast<const float*>(constant));
					ret.w = 0.f;
					return ret;
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
				uint64_t eta;
			} PACK_STRUCT;
			struct SAllConductor
			{
				STextureOrConstant alpha_u;
				STextureOrConstant alpha_v;
				STextureOrConstant opacity;
				STextureOrConstant dummy;
				//3d complex IoR, rgb19e7 format, [0]=real, [1]=imaginary
				uint64_t eta[2];
			} PACK_STRUCT;
			struct SAllCoating
			{
				STextureOrConstant dummy1;
				STextureOrConstant dummy2;
				STextureOrConstant opacity;
				STextureOrConstant sigmaA;
				uint64_t eta;
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

			union alignas(sizeof_uvec4) SBSDFUnion
			{
				_IRR_STATIC_INLINE_CONSTEXPR size_t MAX_TEXTURES = INSTR_MAX_PARAMETER_COUNT;

				SBSDFUnion() : bumpmap{} {}

				SAllDiffuse diffuse;
				SDiffuseTransmitter difftrans;
				SAllDielectric dielectric;
				SAllConductor conductor;
				SAllCoating coating;
				SBumpMap bumpmap;
				SBlend blend;
				struct {
					STextureOrConstant param[MAX_TEXTURES];
					uint64_t extras[2];
				} PACK_STRUCT common;
			};
	#include "irr/irrunpack.h"
		};

	#include "irr/irrpack.h"
		struct STextureOrConstant
		{
			inline void setConst(core::vector3df_SIMD c) { constant = core::rgb32f_to_rgb19e7(c.pointer); }
			inline void setPrefetchReg(uint32_t r) { prefetch = r; }

			inline core::vector3df_SIMD getConst() const 
			{ 
				auto c = core::rgb19e7_to_rgb32f(constant);
				return core::vector3df_SIMD(c.x, c.y, c.z);
			}

			union
			{
				//rgb19e7
				uint64_t constant;
				uint32_t prefetch;
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
			uint64_t eta;
		} PACK_STRUCT;
		struct SAllConductor
		{
			STextureOrConstant alpha_u;
			STextureOrConstant alpha_v;
			STextureOrConstant opacity;
			STextureOrConstant dummy;
			//3d complex IoR, rgb19e7 format, [0]=real, [1]=imaginary
			uint64_t eta[2];
		} PACK_STRUCT;
		struct SAllCoating
		{
			STextureOrConstant dummy1;
			STextureOrConstant dummy2;
			STextureOrConstant opacity;
			STextureOrConstant sigmaA;
			uint64_t eta;
		} PACK_STRUCT;
		struct SBumpMap
		{
			uint32_t derivmap_prefetch_reg;
		} PACK_STRUCT;
		struct SBlend
		{
			STextureOrConstant weight;
		} PACK_STRUCT;

		union alignas(sizeof_uvec4) SBSDFUnion
		{
			_IRR_STATIC_INLINE_CONSTEXPR size_t MAX_TEXTURES = INSTR_MAX_PARAMETER_COUNT;

			SBSDFUnion() : bumpmap{} {}

			SAllDiffuse diffuse;
			SDiffuseTransmitter difftrans;
			SAllDielectric dielectric;
			SAllConductor conductor;
			SAllCoating coating;
			SBumpMap bumpmap;
			SBlend blend;
			struct {
				STextureOrConstant param[MAX_TEXTURES];
				uint64_t extras[2];
			} PACK_STRUCT common;
		};
	#include "irr/irrunpack.h"

	#define _TEXTURE_INDEX(s,m) offsetof(s,m)/sizeof(STextureOrConstant)
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t ALPHA_U_TEX_IX = _TEXTURE_INDEX(SAllDielectric, alpha_u);
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t ALPHA_V_TEX_IX = _TEXTURE_INDEX(SAllDielectric, alpha_v);
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t REFLECTANCE_TEX_IX = _TEXTURE_INDEX(SAllDiffuse, reflectance);
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t TRANSMITTANCE_TEX_IX = _TEXTURE_INDEX(SDiffuseTransmitter, transmittance);
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t SIGMA_A_TEX_IX = _TEXTURE_INDEX(SAllCoating, sigmaA);
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t WEIGHT_TEX_IX = _TEXTURE_INDEX(SBlend, weight);
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t OPACITY_TEX_IX = _TEXTURE_INDEX(SAllDiffuse, opacity);
		//_IRR_STATIC_INLINE_CONSTEXPR uint32_t DERIV_MAP_TEX_IX = _TEXTURE_INDEX(SBumpMap, derivmap);

	#undef _TEXTURE_INDEX

		using traversal_t = core::vector<instr_t>;

		struct remainder_and_pdf
		{
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_WIDTH = 8u;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_MASK = 0xffu;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_DST_SHIFT = BITFIELDS_BSDF_BUF_OFFSET_SHIFT + BITFIELDS_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH * 0u;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_SRC1_SHIFT = BITFIELDS_BSDF_BUF_OFFSET_SHIFT + BITFIELDS_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH * 1u;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_REG_SRC2_SHIFT = BITFIELDS_BSDF_BUF_OFFSET_SHIFT + BITFIELDS_BSDF_BUF_OFFSET_WIDTH + INSTR_REG_WIDTH * 2u;
		};
		struct gen_choice
		{
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_RIGHT_JUMP_WIDTH = INSTR_NORMAL_ID_WIDTH;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_RIGHT_JUMP_MASK = INSTR_NORMAL_ID_MASK;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_RIGHT_JUMP_SHIFT = INSTR_NORMAL_ID_SHIFT;

			_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_OFFSET_INTO_REMANDPDF_STREAM_WIDTH = INSTR_ID_WIDTH;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_OFFSET_INTO_REMANDPDF_STREAM_MASK  = INSTR_ID_MASK;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_OFFSET_INTO_REMANDPDF_STREAM_SHIFT = INSTR_ID_SHIFT;

			inline static uint32_t getOffsetIntoRemAndPdfStream(const instr_t& i)
			{
				return getInstrId(i);
			}
			inline static void setOffsetIntoRemAndPdfStream(instr_t& i, uint32_t offset)
			{
				setInstrId(i, offset);
			}
		};
		struct tex_prefetch
		{
		#include "irr/irrpack.h"
			struct prefetch_instr_t
			{
				prefetch_instr_t() : qword{ 0ull, 0ull } {}

				_IRR_STATIC_INLINE_CONSTEXPR uint32_t DWORD4_DST_REG_SHIFT = 0u;
				_IRR_STATIC_INLINE_CONSTEXPR uint32_t DWORD4_DST_REG_WIDTH = 8u;
				_IRR_STATIC_INLINE_CONSTEXPR uint32_t DWORD4_REG_CNT_SHIFT = DWORD4_DST_REG_SHIFT + DWORD4_DST_REG_WIDTH;
				_IRR_STATIC_INLINE_CONSTEXPR uint32_t DWORD4_REG_CNT_WIDTH = 2u;

				inline void setDstReg(uint32_t r) { dword4 = core::bitfieldInsert(dword4, r, DWORD4_DST_REG_SHIFT, DWORD4_DST_REG_WIDTH); }
				inline void setRegCnt(uint32_t c) { dword4 = core::bitfieldInsert(dword4, c, DWORD4_REG_CNT_SHIFT, DWORD4_REG_CNT_WIDTH); }

				inline uint32_t getDstReg() const { return core::bitfieldExtract(dword4, DWORD4_DST_REG_SHIFT, DWORD4_DST_REG_WIDTH); }
				inline uint32_t getRegCnt() const { return core::bitfieldExtract(dword4, DWORD4_REG_CNT_SHIFT, DWORD4_REG_CNT_WIDTH); }

				union {
					uint64_t qword[2];
					uint32_t dword[4];
					struct {
						STextureData tex_data;
						uint32_t dword4;
					} PACK_STRUCT;
				};
			} PACK_STRUCT;
		#include "irr/irrunpack.h"
			static_assert(sizeof(prefetch_instr_t) == sizeof_uvec4);

			using prefetch_stream_t = core::vector<prefetch_instr_t>;
		};
		struct normal_precomp
		{
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_DST_SHIFT = remainder_and_pdf::INSTR_REG_DST_SHIFT;
			_IRR_STATIC_INLINE_CONSTEXPR uint32_t BITFIELDS_REG_WIDTH = remainder_and_pdf::INSTR_REG_WIDTH;
		};
	};
	struct result_t;
	struct SContext;

protected:
	_IRR_STATIC_INLINE_CONSTEXPR const char* OPCODE_NAMES[instr_stream::OPCODE_COUNT]{
		"OP_DIFFUSE",
		"OP_CONDUCTOR",
		"OP_DIFFTRANS",
		"OP_DIELECTRIC",
		"OP_THINDIELECTRIC",
		"OP_COATING",
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

	void setSourceRegForBumpmaps(instr_stream::traversal_t& _stream, uint32_t _regNumOffset)
	{
		for (instr_stream::instr_t& i : _stream) {
			if (instr_stream::getOpcode(i)==instr_stream::OP_BUMPMAP)
			{
				const uint32_t n_id = instr_stream::getNormalId(i);
				i = core::bitfieldInsert<instr_stream::instr_t>(i, _regNumOffset+n_id, instr_stream::INSTR_BUMPMAP_SRC_REG_SHIFT, instr_stream::INSTR_BUMPMAP_SRC_REG_WIDTH);
			}
		}
	}

	//common for rem_and_pdf and gen_choice instructions
	void debugPrintInstr(std::ostream& _out, instr_stream::instr_t instr, const result_t& _res, const SContext* _ctx) const;

public:
	class SContext
	{
		template <typename stack_el_t>
		friend class ITraversalGenerator;

		friend class CMaterialCompilerGLSLBackendCommon;

		//users should not touch this
		core::vector<instr_stream::intermediate::SBSDFUnion> bsdfData;
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

	public:
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

			stream_t get_rem_and_pdf() const { return { offset, rem_and_pdf_count }; }
			stream_t get_gen_choice() const { return {offset + rem_and_pdf_count, gen_choice_count}; }
			stream_t get_norm_precomp() const { return { offset + rem_and_pdf_count + gen_choice_count, norm_precomp_count }; }

			uint32_t offset;
			uint32_t rem_and_pdf_count;
			uint32_t gen_choice_count;
			uint32_t norm_precomp_count;

			stream_t get_tex_prefetch() const { return { prefetch_offset, tex_prefetch_count }; }

			uint32_t prefetch_offset;
			uint32_t tex_prefetch_count;
		};

		instr_stream::traversal_t instructions;
		instr_stream::tex_prefetch::prefetch_stream_t prefetch_stream;
		core::vector<instr_stream::SBSDFUnion> bsdfData;

		//TODO flags like alpha tex always present etc..
		bool noPrefetchStream;
		bool noNormPrecompStream;
		bool allIsotropic;
		bool noTwosided;
		bool noBSDF;
		uint32_t usedRegisterCount;
		uint32_t globalPrefetchRegCountFlags;
		uint32_t paramTexPresence[instr_stream::SBSDFUnion::MAX_TEXTURES][2];
		// always same value and the value
		std::pair<bool, core::vector3df_SIMD> paramConstants[instr_stream::SBSDFUnion::MAX_TEXTURES];

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
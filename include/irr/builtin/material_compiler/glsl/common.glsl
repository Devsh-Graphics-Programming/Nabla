#version 430 core

#include <irr/builtin/glsl/virtual_texturing/extensions.glsl>

layout (location = 0) in vec3 WorldPos;
layout (location = 1) flat in uint InstanceIndex;
layout (location = 2) in vec3 Normal;
layout (location = 3) in vec2 UV;

layout (location = 0) out vec4 OutColor;

#define instr_t uvec2
#define reg_t uint
#define REG_COUNT 72 //TODO this must be decided by backend
//TODO insert here other control #defines as well

//in 16-byte/uvec4 units
//layout (constant_id = 0) const uint sizeof_bsdf_data = 3;
#define sizeof_bsdf_data 3

struct bsdf_data_t
{
	uvec4 data[sizeof_bsdf_data];
};

#define _IRR_VT_DESCRIPTOR_SET 0
#define _IRR_VT_PAGE_TABLE_BINDING 0

#define _IRR_VT_FLOAT_VIEWS_BINDING 1 
#define _IRR_VT_FLOAT_VIEWS_COUNT 4
#define _IRR_VT_FLOAT_VIEWS

#define _IRR_VT_INT_VIEWS_BINDING 2
#define _IRR_VT_INT_VIEWS_COUNT 0
#define _IRR_VT_INT_VIEWS

#define _IRR_VT_UINT_VIEWS_BINDING 3
#define _IRR_VT_UINT_VIEWS_COUNT 0
#define _IRR_VT_UINT_VIEWS
#include <irr/builtin/glsl/virtual_texturing/descriptors.glsl>

layout (set = 0, binding = 2, std430) restrict readonly buffer PrecomputedStuffSSBO
{
    uint pgtab_sz_log2;
    float vtex_sz_rcp;
    float phys_pg_tex_sz_rcp[_IRR_VT_MAX_PAGE_TABLE_LAYERS];
    uint layer_to_sampler_ix[_IRR_VT_MAX_PAGE_TABLE_LAYERS];
} precomputed;

layout (set = 0, binding = 3, std430) restrict readonly buffer INSTR_BUF
{
	instr_t data[];
} instr_buf;
layout (set = 0, binding = 4, std430) restrict readonly buffer BSDF_BUF
{
	bsdf_data_t data[];
} bsdf_buf;

uint irr_glsl_VT_layer2pid(in uint layer)
{
    return precomputed.layer_to_sampler_ix[layer];
}
uint irr_glsl_VT_getPgTabSzLog2()
{
    return precomputed.pgtab_sz_log2;
}
float irr_glsl_VT_getPhysPgTexSzRcp(in uint layer)
{
    return precomputed.phys_pg_tex_sz_rcp[layer];
}
float irr_glsl_VT_getVTexSzRcp()
{
    return precomputed.vtex_sz_rcp;
}
#define _IRR_USER_PROVIDED_VIRTUAL_TEXTURING_FUNCTIONS_

#include <irr/builtin/glsl/virtual_texturing/functions.glsl/7/8>

struct stream_t
{
	uint offset;
	uint count;
};
layout (push_constant) uniform Block
{
	stream_t rem_and_pdf;
	stream_t tex_prefetch;
	stream_t norm_precomp;
	stream_t gen_choice;
} PC;

#include <irr/builtin/glsl/utils/vertex.glsl>

layout (set = 1, binding = 0, row_major, std140) uniform UBO {
    irr_glsl_SBasicViewParameters params;
} CamData;

struct InstanceData
{
	mat4x3 tform;
	vec3 normalMatrixRow0;
	uint instrOffset;
	vec3 normalMatrixRow1;
	uint instrCount;
	vec3 normalMatrixRow2;
	uint _padding;//not needed
	uvec2 emissive;
};
layout (set = 0, binding = 5, row_major, std430) readonly restrict buffer InstDataBuffer {
	InstanceData data[];
} InstData;

//put this into some builtin
#define RGB19E7_MANTISSA_BITS 19
#define RGB19E7_MANTISSA_MASK 0x7ffff
#define RGB19E7_EXPONENT_BITS 7
#define RGB19E7_EXP_BIAS 63
vec3 decodeRGB19E7(in uvec2 x)
{
	int exp = int(bitfieldExtract(x.y, 3*RGB19E7_MANTISSA_BITS-32, RGB19E7_EXPONENT_BITS) - RGB19E7_EXP_BIAS - RGB19E7_MANTISSA_BITS);
	float scale = exp2(float(exp));//uintBitsToFloat((uint(exp)+127u)<<23u)
	
	vec3 v;
	v.x = int(bitfieldExtract(x.x, 0, RGB19E7_MANTISSA_BITS))*scale;
	v.y = int(
		bitfieldExtract(x.x, RGB19E7_MANTISSA_BITS, 32-RGB19E7_MANTISSA_BITS) | 
		(bitfieldExtract(x.y, 0, RGB19E7_MANTISSA_BITS-(32-RGB19E7_MANTISSA_BITS))<<(32-RGB19E7_MANTISSA_BITS))
	) * scale;
	v.z = int(bitfieldExtract(x.y, RGB19E7_MANTISSA_BITS-(32-RGB19E7_MANTISSA_BITS), RGB19E7_MANTISSA_BITS)) * scale;
	
	return v;
}

//i think ill have to create some c++ macro or something to create string with those
//becasue it's too fucked up to remember about every change in c++ and have to update everything here
#define INSTR_OPCODE_MASK			0x0fu
#define INSTR_REG_MASK				0xffu
#define INSTR_BSDF_BUF_OFFSET_SHIFT 13
#define INSTR_BSDF_BUF_OFFSET_MASK	0x7ffffu
#define INSTR_NDF_SHIFT 5
#define INSTR_NDF_MASK 0x3u
#define INSTR_ALPHA_U_TEX_SHIFT 4
#define INSTR_ALPHA_V_TEX_SHIFT 7
#define INSTR_REFL_TEX_SHIFT 9
#define INSTR_TRANS_TEX_SHIFT 4
#define INSTR_SIGMA_A_TEX_SHIFT 8
#define INSTR_WEIGHT_TEX_SHIFT 4
#define INSTR_TWOSIDED_SHIFT 10
#define INSTR_MASKFLAG_SHIFT 11
#define INSTR_OPACITY_TEX_SHIFT 12
#define INSTR_1ST_PARAM_TEX_SHIFT 4

#define INSTR_NORMAL_ID_SHIFT 56u
#define INSTR_NORMAL_ID_MASK  0xffu

uint instr_getNormalId(in instr_t instr)
{
	return (instr.y>>(INSTR_NORMAL_ID_SHIFT-32u)) & INSTR_NORMAL_ID_MASK;
}
uint instr_getOpcode(in instr_t instr)
{
	return instr.x&INSTR_OPCODE_MASK;
}
uint instr_getBSDFbufOffset(in instr_t instr)
{
	return (instr.x>>INSTR_BSDF_BUF_OFFSET_SHIFT) & INSTR_BSDF_BUF_OFFSET_MASK;
}
uint instr_getNDF(in instr_t instr)
{
	return (instr.x>>INSTR_NDF_SHIFT) & INSTR_NDF_MASK;
}
bool instr_getAlphaUTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_ALPHA_U_TEX_SHIFT)) != 0u;
}
bool instr_getAlphaVTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_ALPHA_V_TEX_SHIFT)) != 0u;
}
bool instr_getReflectanceTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_REFL_TEX_SHIFT)) != 0u;
}
bool instr_getPlasticReflTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_PLASTIC_REFL_TEX_SHIFT)) != 0u;
}
uint instr_getWardVariant(in instr_t instr)
{
	return (instr.x>>INSTR_WARD_VARIANT_SHIFT) & INSTR_WARD_VARIANT_MASK;
}
bool instr_getFastApprox(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_FAST_APPROX_SHIFT)) != 0u;
}
bool instr_getNonlinear(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_NONLINEAR_SHIFT)) != 0u;
}
bool instr_getSigmaATexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_SIGMA_A_TEX_SHIFT)) != 0u;
}
bool instr_getWeightTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_WEIGHT_TEX_SHIFT)) != 0u;
}
bool instr_getTwosided(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_TWOSIDED_SHIFT)) != 0u;
}
bool instr_getMaskFlag(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_MASKFLAG_SHIFT)) != 0u;
}
bool instr_getOpacityTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_OPACITY_TEX_SHIFT)) != 0u;
}
bool instr_getTransmittanceTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_TRANS_TEX_SHIFT)) != 0u;
}
bool instr_get1stParamTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_1ST_PARAM_TEX_SHIFT)) != 0u;
}

//returns: x=dst, y=src1, z=src2
uvec3 instr_decodeRegisters(in instr_t instr)
{
	uvec3 regs = uvec3(instr.y, (instr.y>>8), (instr.y>>16));
	return regs & uvec3(INSTR_REG_MASK);
}
#define REG_DST(r)	r.x
#define REG_SRC1(r)	r.y
#define REG_SRC2(r)	r.z

bsdf_data_t fetchBSDFDataForInstr(in instr_t instr)
{
	uint ix = instr_getBSDFbufOffset(instr);
	return bsdf_buf.data[ix];
}

//remember to keep it compliant with c++ enum!!
#define OP_DIFFUSE			0u
#define OP_CONDUCTOR		1u
#define OP_PLASTIC			2u
#define OP_COATING			3u
#define OP_MAX_BRDF			OP_COATING
#define OP_DIFFTRANS		4u
#define OP_DIELECTRIC		5u
#define OP_MAX_BSDF			OP_DIELECTRIC
#define OP_BLEND			6u
#define OP_BUMPMAP			7u
#define OP_SET_GEOM_NORMAL	8u
#define OP_INVALID			9u
#define OP_NOOP				10u

#define NDF_BECKMANN	0u
#define NDF_GGX			1u
#define NDF_PHONG		2u
#define NDF_AS			3u

#include <irr/builtin/glsl/bxdf/common.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/fresnel/fresnel.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/fresnel_correction.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ndf/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ashikhmin_shirley.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/beckmann_smith.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl>
#include <irr/builtin/glsl/bump_mapping/utils.glsl>

irr_glsl_BSDFAnisotropicParams currBSDFParams;
irr_glsl_AnisotropicViewSurfaceInteraction currInteraction;
reg_t registers[REG_COUNT];

vec3 textureOrRGBconst(in uvec3 data, in bool texPresenceFlag)
{
	return 
#ifdef TEX_PREFETCH_STREAM
	texPresenceFlag ? 
		uintBitsToFloat(uvec3(registers[data.z],registers[data.z+1u],registers[data.z+2u])) :
#endif
		uintBitsToFloat(data);
}
float textureOrRconst(in uvec3 data, in bool texPresenceFlag)
{
	return 
#ifdef TEX_PREFETCH_STREAM
	texPresenceFlag ?
		uintBitsToFloat(registers[data.z]) :
#endif
		uintBitsToFloat(data.x);
}

void writeReg(uint n, float v)
{
	registers[n] = floatBitsToUint(v);
}
void writeReg(uint n, vec3 v)
{
	writeReg(n   ,v.x);
	writeReg(n+1u,v.y);
	writeReg(n+2u,v.z);
}

void setCurrBSDFParams(in vec3 n, in vec3 L)
{
	vec3 campos = irr_glsl_SBasicViewParameters_GetEyePos(CamData.params.NormalMatAndEyePos);
	irr_glsl_IsotropicViewSurfaceInteraction interaction = irr_glsl_calcFragmentShaderSurfaceInteraction(campos, WorldPos, n);
	currInteraction = irr_glsl_calcAnisotropicInteraction(interaction);
	irr_glsl_BSDFIsotropicParams isoparams = irr_glsl_calcBSDFIsotropicParams(interaction, L);
	currBSDFParams = irr_glsl_calcBSDFAnisotropicParams(isoparams, currInteraction);
}

#ifdef OP_DIFFUSE
void instr_execute_DIFFUSE(in instr_t instr, in uvec3 regs, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		vec3 refl = textureOrRGBconst(data.data[1].zw, instr_getReflectanceTexPresence(instr));
		vec3 diffuse = irr_glsl_lambertian_cos_eval(currBSDFParams.isotropic,currInteraction.isotropic) * refl;
		registers[REG_DST(regs)] = diffuse;
	}
	else registers[REG_DST(regs)] = reg_t(0.0);
}
#endif
#ifdef OP_DIFFTRANS
void instr_execute_DIFFTRANS(in instr_t instr, in uvec3 regs, in mat2 dUV, in vec3 trans, in bsdf_data_t data)
{
	registers[REG_DST(regs)] = reg_t(1.0, 0.0, 0.0);
}
#endif
#ifdef OP_DIELECTRIC
void instr_execute_DIELECTRIC(in instr_t instr, in uvec3 regs, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		vec3 eta = vec3(uintBitsToFloat(data.data[2].x));
		vec3 diffuse = irr_glsl_lambertian_cos_eval(currBSDFParams.isotropic,currInteraction.isotropic) * vec3(0.89);
		diffuse *= irr_glsl_diffuseFresnelCorrectionFactor(eta,eta*eta) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currInteraction.isotropic.NdotV)) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currBSDFParams.isotropic.NdotL));
		registers[REG_DST(regs)] = diffuse;
	}
	else registers[REG_DST(regs)] = vec3(0.0);
}
#endif
#ifdef OP_CONDUCTOR
void instr_execute_CONDUCTOR(in instr_t instr, in uvec3 regs, in mat2 dUV, in float DG, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		mat2x3 eta2;
		eta2[0] = decodeRGB19E7(data.data[2].xy); eta2[0]*=eta2[0];
		eta2[1] = decodeRGB19E7(data.data[2].zw); eta2[1]*=eta2[1];
		vec3 fr = irr_glsl_fresnel_conductor(eta2[0],eta2[1],currBSDFParams.isotropic.VdotH);
		registers[REG_DST(regs)] = DG*fr;
	}
	else registers[REG_DST(regs)] = reg_t(0.0);
}
#endif
#ifdef OP_PLASTIC
void instr_execute_PLASTIC(in instr_t instr, in uvec3 regs, in mat2 dUV, in vec3 scale, in float a2, in float DG, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		vec3 eta = vec3(uintBitsToFloat(data.data[2].x));
		vec3 eta2 = eta*eta;
		vec3 refl = textureOrRGBconst(data.data[1].zw, instr_getPlasticReflTexPresence(instr));

		vec3 diffuse = irr_glsl_oren_nayar_cos_eval(currBSDFParams.isotropic, currInteraction.isotropic, a2) * refl;
		diffuse *= irr_glsl_diffuseFresnelCorrectionFactor(eta,eta2) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currInteraction.isotropic.NdotV)) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currBSDFParams.isotropic.NdotL));
		vec3 fr = irr_glsl_fresnel_dielectric(eta,currBSDFParams.isotropic.VdotH);
		vec3 specular = DG*fr;

		registers[REG_DST(regs)] = specular + diffuse;
	}
	else registers[REG_DST(regs)] = reg_t(0.0);
}
#endif
#ifdef OP_COATING
void instr_execute_COATING(in instr_t instr, in uvec3 regs, in mat2 dUV, in vec3 scale, in bsdf_data_t data)
{
	vec2 thickness_eta = unpackHalf2x16(data.data[2].x);
	vec3 sigmaA = textureOrRGBconst(data.data[1].zw, instr_getSigmaATexPresence(instr));
	registers[REG_DST(regs)] = reg_t(1.0, 0.0, 0.0);
}
#endif
#ifdef OP_BUMPMAP
void instr_execute_BUMPMAP(in instr_t instr, in vec3 L)
{
	vec3 n = normal_registers[instr_getNormalId(instr)];
	setCurrBSDFParams(n, L);
}
#endif
#ifdef OP_SET_GEOM_NORMAL
//executed at most once
void instr_execute_SET_GEOM_NORMAL(in vec3 L)
{
	setCurrBSDFParams(normalize(Normal), L);
}
#endif
#ifdef OP_BLEND
void instr_execute_BLEND(in instr_t instr, in uvec3 regs, in bsdf_data_t data)
{
	bool weightTexPresent = instr_getWeightTexPresence(instr);
	float w = textureOrRconst(data.data[0].zw, weightTexPresent);

	registers[REG_DST(regs)] = mix(registers[REG_SRC1(regs)], registers[REG_SRC2(regs)], w);
}
#endif

vec3 fetchTex(in uvec3 texid, in vec2 uv, in mat2 dUV)
{
	float scale = uintBitsToFloat(texid.z);

	return irr_glsl_vTextureGrad(texid.xy, uv, dUV).rgb*scale;
}

#define INSTR_FETCH_FLAGS_SHIFT 4u
#define INSTR_FETCH_TEX_0_REG_CNT_SHIFT 7u
#define INSTR_FETCH_TEX_1_REG_CNT_SHIFT 9u
#define INSTR_FETCH_TEX_2_REG_CNT_SHIFT 11u
#define INSTR_FETCH_TEX_REG_CNT_MASK    0x03u
#ifdef TEX_PREFETCH_STREAM
void runTexPrefetchStream(in mat2 dUV)
{
	for (uint i = 0u; i < PC.tex_prefetch.offset; ++i)
	{
		instr_t instr = instr_buf.data[PC.tex_prefetch.offset+i];

		uvec3 regcnt = (instr.xxx >> uvec3(INSTR_FETCH_TEX_0_REG_CNT_SHIFT,INSTR_FETCH_TEX_1_REG_CNT_SHIFT,INSTR_FETCH_TEX_2_REG_CNT_SHIFT)) & uvec3(INSTR_FETCH_TEX_REG_CNT_MASK);
		bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);

		uint fetchFlags = (instr.x>>INSTR_FETCH_FLAGS_SHIFT);
		uvec3 regs = instr_decodeRegisters(instr);
		if ((fetchFlags & 0x01u) == 0x01u)
		{
			vec3 val = fetchTex(bsdf_data.data[0].xyz, UV, dUV);
			if (regcnt.x==1u)
				writeReg(regs.x, val.x);
			else
				writeReg(regs.x, val);
		}
		if ((fetchFlags & 0x02u) == 0x02u)
		{
			uvec3 texid = uvec3(bsdf_data.data[0].w,bsdf_data.data[1].xy);
			vec3 val = fetchTex(texid,UV,dUV);
			if (regcnt.y==1u)
				writeReg(regs.y, val.x);
			else
				writeReg(regs.y, val);
		}
		if ((fetchFlags & 0x04u) == 0x04u)
		{
			uvec3 texid = uvec3(bsdf_data.data[1].zw, bsdf_data.data[2].x);
			vec3 val = fetchTex(texid,UV,dUV);
			if (regcnt.z==1u)
				writeReg(regs.z, val.x);
			else
				writeReg(regs.z, val);
		}
	}
}
#endif

#ifdef NORM_PRECOMP_STREAM
void runNormalPrecompStream(in mat2 dUV)
{
	for (uint i = 0u; i < PC.norm_precomp.offset; ++i)
	{
		instr_t instr = instr_buf.data[PC.norm_precomp.offset+i];

		bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);

		uint reg = instr_getNormalId(instr);

		uvec2 bm = data.data[0].xy;
		float scale = uintBitsToFloat(data.data[0].z);
		//dirty trick for getting height map derivatives in divergent workflow
		vec2 dHdScreen = vec2(
			irr_glsl_vTextureGrad(bm, UV+0.5*dUV[0], dUV).x - irr_glsl_vTextureGrad(bm, UV-0.5*dUV[0], dUV).x,
			irr_glsl_vTextureGrad(bm, UV+0.5*dUV[1], dUV).x - irr_glsl_vTextureGrad(bm, UV-0.5*dUV[1], dUV).x
		) * scale;
		normal_registers[reg] = irr_glsl_perturbNormal_heightMap(currInteraction.isotropic.N, currInteraction.isotropic.V.dPosdScreen, dHdScreen);
	}
}
#endif
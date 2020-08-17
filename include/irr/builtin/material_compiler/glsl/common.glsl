#include <irr/builtin/glsl/virtual_texturing/extensions.glsl>

layout (location = 0) in vec3 WorldPos;
layout (location = 1) flat in uint InstanceIndex;
layout (location = 2) in vec3 Normal;
layout (location = 3) in vec2 UV;

layout (location = 0) out vec4 OutColor;

#define instr_t uvec2
#define reg_t uint
#define params_t mat3x3
#define bxdf_eval_t vec3
#define REG_COUNT 72 //TODO this must be decided by backend
//TODO insert here other control #defines as well

//in 16-byte/uvec4 units
//layout (constant_id = 0) const uint sizeof_bsdf_data = 4;
#define sizeof_bsdf_data 4

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

#include <irr/builtin/glsl/utils/common.glsl>

layout (set = 1, binding = 0, row_major, std140) uniform UBO {
    irr_glsl_SBasicViewParameters params;
} CamData;

struct InstanceData
{
	mat4x3 tform;
	vec3 normalMatrixRow0;
	uint bsdf_instrOffset;
	vec3 normalMatrixRow1;
	uint bsdf_instrCount;
	vec3 normalMatrixRow2;
	uint _padding;//not needed
	uvec2 prefetch_instrStream;
	uvec2 nprecomp_instrStream;
	uvec2 genchoice_instrStream;
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
#define INSTR_REFL_TEX_SHIFT 7
#define INSTR_TRANS_TEX_SHIFT 4
#define INSTR_SIGMA_A_TEX_SHIFT 7
#define INSTR_WEIGHT_TEX_SHIFT 4
#define INSTR_TWOSIDED_SHIFT 10
#define INSTR_MASKFLAG_SHIFT 11
#define INSTR_OPACITY_TEX_SHIFT 12
#define INSTR_1ST_PARAM_TEX_SHIFT 4
#define INSTR_2ND_PARAM_TEX_SHIFT 7

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

bool instr_get1stParamTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_1ST_PARAM_TEX_SHIFT)) != 0u;
}
bool instr_get2ndParamTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_2ND_PARAM_TEX_SHIFT)) != 0u;
}

bool instr_getAlphaUTexPresence(in instr_t instr)
{
	return instr_get1stParamTexPresence(instr);
}
bool instr_getAlphaVTexPresence(in instr_t instr)
{
	return instr_get2ndParamTexPresence(instr);
}
bool instr_getReflectanceTexPresence(in instr_t instr)
{
	return instr_get2ndParamTexPresence(instr);
}
bool instr_getSigmaATexPresence(in instr_t instr)
{
	return instr_get2ndParamTexPresence(instr);
}
bool instr_getTransmittanceTexPresence(in instr_t instr)
{
	return instr_get1stParamTexPresence(instr);
}
bool instr_getWeightTexPresence(in instr_t instr)
{
	return instr_get1stParamTexPresence(instr);
}
bool instr_getOpacityTexPresence(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_OPACITY_TEX_SHIFT)) != 0u;
}

bool instr_getTwosided(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_TWOSIDED_SHIFT)) != 0u;
}
bool instr_getMaskFlag(in instr_t instr)
{
	return (instr.x&(1u<<INSTR_MASKFLAG_SHIFT)) != 0u;
}

#define INSTR_REG_DST_SHIFT 32
#define INSTR_REG_SRC1_SHIFT 40
#define INSTR_REG_SRC2_SHIFT 48

//returns: x=dst, y=src1, z=src2
//works with tex prefetch instructions as well (x=reg0,y=reg1,z=reg2)
uvec3 instr_decodeRegisters(in instr_t instr)
{
	uvec3 regs = instr.yyy >> (uvec3(INSTR_REG_DST_SHIFT,INSTR_REG_SRC1_SHIFT,INSTR_REG_SRC2_SHIFT)-32u);
	return regs & uvec3(INSTR_REG_MASK);
}
#define REG_DST(r)	(r).x
#define REG_SRC1(r)	(r).y
#define REG_SRC2(r)	(r).z
#define REG_PREFETCH0(r) REG_DST(r)
#define REG_PREFETCH1(r) REG_SRC1(r)
#define REG_PREFETCH2(r) REG_SRC2(r)

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

bool op_isBRDF(in uint op)
{
	return op<=OP_MAX_BRDF;
}
bool op_isBSDF(in uint op)
{
	return !op_isBRDF(op) && op<=OP_MAX_BSDF;
}
bool op_hasSpecular(in uint op)
{
	return op_isBSDF(op) && op!=OP_DIFFUSE && op!=OP_DIFFTRANS;
}

#define NDF_BECKMANN	0u
#define NDF_GGX			1u
#define NDF_PHONG		2u
#define NDF_AS			3u

#include <irr/builtin/glsl/bxdf/common.glsl>
#include <irr/builtin/glsl/bxdf/fresnel.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/fresnel_correction.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ndf/ggx.glsl>
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

bvec3 instr_getTexPresence(in instr_t i)
{
	return bvec3 p(
		instr_get1stParamTexPresence(instr),
		instr_get2ndParamTexPresence(instr),
		instr_getOpacityTexPresence(instr)
	);
}
params_t instr_getParameters(in instr_t i, in bsdf_data_t data)
{
	params_t p;
	bvec3 presence = instr_getTexPresence(i);
	p[0] = textureOrRGBconst(data.data[0].xyz, presence.x);
	p[1] = textureOrRGBconst(uvec3(data.data[0].w,data.data[1].xy), presence.y);
	p[2] = textureOrRGBconst(uvec3(data.data[1].zw,data.data[2].x), presence.z);

	return p;
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
float readReg1(uint n)
{
	return uintBitsToFloat( registers[n] );
}
vec3 readReg3(uint n)
{
	return vec3(
		readReg(n), readReg(n+1u), readReg(n+2u)
	);
}

void setCurrBSDFParams(in vec3 n, in vec3 L)
{
	vec3 campos = irr_glsl_SBasicViewParameters_GetEyePos(CamData.params.NormalMatAndEyePos);
	irr_glsl_IsotropicViewSurfaceInteraction interaction = irr_glsl_calcFragmentShaderSurfaceInteraction(campos, WorldPos, n);
	currInteraction = irr_glsl_calcAnisotropicInteraction(interaction);
	irr_glsl_BSDFIsotropicParams isoparams = irr_glsl_calcBSDFIsotropicParams(interaction, L);
	currBSDFParams = irr_glsl_calcBSDFAnisotropicParams(isoparams, currInteraction);
}

float getAlpha(in params_t p)
{
	return p[0].x;
}
vec3 getReflectance(in params_t p)
{
	return p[1];
}
vec3 getOpacity(in params_t p)
{
	return p[2];
}
float getAlphaV(in params_t p)
{
	return p[1].x;
}
vec3 getSigmaA(in params_t p)
{
	return getReflectance(data,texPresence);
}
float getBlendWeight(in params_t p)
{
	return getAlpha(data,texPresence);
}
vec3 getTransmittance(in params_t p)
{
	return p[0];
}


#ifdef OP_DIFFUSE
void instr_execute_DIFFUSE(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		vec3 refl = getReflectance(params);
		float a = getAlpha(params);
		vec3 diffuse = irr_glsl_oren_nayar_cos_eval(currBSDFParams.isotropic,currInteraction.isotropic,a*a) * refl;
		writeReg(REG_DST(regs), diffuse);
	}
	else writeReg(REG_DST(regs), bxdf_eval_t(0.0));
}
#endif
#ifdef OP_DIFFTRANS
void instr_execute_DIFFTRANS(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	vec3 tr = getTransmittance(params);
	writeReg(REG_DST(regs), bxdf_eval_t(1.0,0.0,0.0));
}
#endif
#ifdef OP_DIELECTRIC
void instr_execute_DIELECTRIC(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		//float au = getAlpha(params);
		//float av = getAlphaV(params);
		vec3 eta = vec3(uintBitsToFloat(data.data[2].y));
		vec3 diffuse = irr_glsl_lambertian_cos_eval(currBSDFParams.isotropic,currInteraction.isotropic) * vec3(0.89);
		diffuse *= irr_glsl_diffuseFresnelCorrectionFactor(eta,eta*eta) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currInteraction.isotropic.NdotV)) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currBSDFParams.isotropic.NdotL));
		writeReg(REG_DST(regs), diffuse);
	}
	else writeReg(REG_DST(regs), bxdf_eval_t(0.0));
}
#endif
#ifdef OP_CONDUCTOR
void instr_execute_CONDUCTOR(in instr_t instr, in uvec3 regs, in float DG, in params_t params, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		//float au = getAlpha(params);
		//float av = getAlphaV(params);
		mat2x3 eta;
		eta[0] = decodeRGB19E7(data.data[2].yz);
		eta[1] = decodeRGB19E7(uvec2(data.data[2].w,data.data[3].x));
		vec3 fr = irr_glsl_fresnel_conductor(eta[0],eta[1],currBSDFParams.isotropic.VdotH);
		writeReg(REG_DST(regs), DG*fr);
	}
	else writeReg(REG_DST(regs), bxdf_eval_t(0.0));
}
#endif
#ifdef OP_PLASTIC
void instr_execute_PLASTIC(in instr_t instr, in uvec3 regs, in float DG, in params_t params, in bsdf_data_t data)
{
	if (currBSDFParams.isotropic.NdotL>FLT_MIN)
	{
		vec3 eta = vec3(uintBitsToFloat(data.data[2].y));
		vec3 eta2 = eta*eta;
		vec3 refl = getReflectance(params);
		float a2 = getAlpha(params);
		a2*=a2;

		vec3 diffuse = irr_glsl_oren_nayar_cos_eval(currBSDFParams.isotropic, currInteraction.isotropic, a2) * refl;
		diffuse *= irr_glsl_diffuseFresnelCorrectionFactor(eta,eta2) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currInteraction.isotropic.NdotV)) * (vec3(1.0)-irr_glsl_fresnel_dielectric(eta, currBSDFParams.isotropic.NdotL));
		vec3 fr = irr_glsl_fresnel_dielectric(eta,currBSDFParams.isotropic.VdotH);
		vec3 specular = DG*fr;

		writeReg(REG_DST(regs), specular+diffuse);
	}
	writeReg(REG_DST(regs), bxdf_eval_t(0.0));
}
#endif
#ifdef OP_COATING
void instr_execute_COATING(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	//vec2 thickness_eta = unpackHalf2x16(data.data[2].y);
	//vec3 sigmaA = getSigmaA(params);
	//float a = getAlpha(params);
	writeReg(REG_DST(regs), bxdf_eval_t(1.0,0.0,0.0));
}
#endif
#ifdef OP_BUMPMAP
void instr_execute_BUMPMAP(in instr_t instr, in vec3 L)
{
	vec3 n = readReg3( REG_SRC1(instr_decodeRegisters(instr)) );
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
void instr_execute_BLEND(in instr_t instr, in uvec3 regs, in params_t params, in bsdf_data_t data)
{
	float w = getBlendWeight(params);
	bxdf_eval_t bxdf1 = readReg3(REG_SRC1(regs));
	bxdf_eval_t bxdf2 = readReg3(REG_SRC2(regs));

	bxdf_eval_t blend = mix(bxdf1, bxdf2, w);
	writeReg(REG_DST(regs), blend);
}
#endif

vec3 fetchTex(in uvec3 texid, in vec2 uv, in mat2 dUV)
{
	float scale = uintBitsToFloat(texid.z);

	return irr_glsl_vTextureGrad(texid.xy, uv, dUV).rgb*scale;
}

#define INSTR_FETCH_FLAG_TEX_0_SHIFT INSTR_NDF_SHIFT
#define INSTR_FETCH_FLAG_TEX_1_SHIFT (INSTR_NDF_SHIFT+1)
#define INSTR_FETCH_FLAG_TEX_2_SHIFT INSTR_TWOSIDED_SHIFT
uint instr_getTexFetchFlags(in instr_t i)
{
	uint flags = bitfieldExtract(i.x,INSTR_FETCH_FLAG_TEX_0_SHIFT,1);
	flags |= bitfieldExtract(i.x,INSTR_FETCH_FLAG_TEX_1_SHIFT,1)<<1;
	flags |= bitfieldExtract(i.x,INSTR_FETCH_FLAG_TEX_2_SHIFT,1)<<2;

	return flags;
}

#define INSTR_FETCH_TEX_0_REG_CNT_SHIFT 56
#define INSTR_FETCH_TEX_1_REG_CNT_SHIFT 58
#define INSTR_FETCH_TEX_2_REG_CNT_SHIFT 60
#define INSTR_FETCH_TEX_REG_CNT_MASK    0x03u
#ifdef TEX_PREFETCH_STREAM
void runTexPrefetchStream(uvec2 stream, in mat2 dUV)
{
	for (uint i = 0u; i < stream.y; ++i)
	{
		instr_t instr = instr_buf.data[stream.x+i];

		uvec3 regcnt = ( instr.yyy >> (uvec3(INSTR_FETCH_TEX_0_REG_CNT_SHIFT,INSTR_FETCH_TEX_1_REG_CNT_SHIFT,INSTR_FETCH_TEX_2_REG_CNT_SHIFT)-uvec3(32)) ) & uvec3(INSTR_FETCH_TEX_REG_CNT_MASK);
		bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);

		uint fetchFlags = instr_getTexFetchFlags(instr);
		uvec3 regs = instr_decodeRegisters(instr);
		if ((fetchFlags & 0x01u) == 0x01u)
		{
			vec3 val = fetchTex(bsdf_data.data[0].xyz, UV, dUV);
			uint reg = REG_PREFETCH0(regs);
			if (regcnt.x==1u)
				writeReg(reg, val.x);
			else
				writeReg(reg, val);
		}
		if ((fetchFlags & 0x02u) == 0x02u)
		{
			uvec3 texid = uvec3(bsdf_data.data[0].w,bsdf_data.data[1].xy);
			vec3 val = fetchTex(texid,UV,dUV);
			uint reg = REG_PREFETCH1(regs);
			if (regcnt.y==1u)
				writeReg(reg, val.x);
			else
				writeReg(reg, val);
		}
		if ((fetchFlags & 0x04u) == 0x04u)
		{
			uvec3 texid = uvec3(bsdf_data.data[1].zw, bsdf_data.data[2].x);
			vec3 val = fetchTex(texid,UV,dUV);
			uint reg = REG_PREFETCH2(regs);
			if (regcnt.z==1u)
				writeReg(reg, val.x);
			else
				writeReg(reg, val);
		}
	}
}
#endif

#ifdef NORM_PRECOMP_STREAM
void runNormalPrecompStream(in uvec2 stream, in mat2 dUV)
{
	for (uint i = 0u; i < stream.y; ++i)
	{
		instr_t instr = instr_buf.data[stream.x+i];

		bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);

		uint reg = REG_DST(instr_decodeRegisters(instr));

		uvec2 bm = data.data[0].xy;
		float scale = uintBitsToFloat(data.data[0].z);
		//dirty trick for getting height map derivatives in divergent workflow
		vec2 dHdScreen = vec2(
			irr_glsl_vTextureGrad(bm, UV+0.5*dUV[0], dUV).x - irr_glsl_vTextureGrad(bm, UV-0.5*dUV[0], dUV).x,
			irr_glsl_vTextureGrad(bm, UV+0.5*dUV[1], dUV).x - irr_glsl_vTextureGrad(bm, UV-0.5*dUV[1], dUV).x
		) * scale;
		writeReg(reg,
			irr_glsl_perturbNormal_heightMap(currInteraction.isotropic.N, currInteraction.isotropic.V.dPosdScreen, dHdScreen)
		);
	}
}
#endif

void runEvalStream(in uvec2 stream, in vec3 L)
{
	for (uint i = 0u; i < stream.y; ++i)
	{
		instr_t instr = instr_buf.data[stream.x+i];
		uint op = instr_getOpcode(instr);

		//speculative execution
		bsdf_data_t bsdf_data = fetchBSDFDataForInstr(instr);
		params_t params = instr_getParameters(instr);
		float bxdf_eval_scalar_part;
		uint ndf = instr_getNDF(instr);
		float a = getAlpha(params);
		float a2 = a*a;
		float ay = getAlphaV(params);
		float ay2 = ay*ay;

		if (op_hasSpecular(op))
		{
			if (ndf==NDF_GGX) {
				bxdf_eval_scalar_part = irr_glsl_ggx_height_correlated_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, a2);
			}
			else if (ndf==NDF_BECKMANN) {
				bxdf_eval_scalar_part = irr_glsl_beckmann_smith_height_correlated_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, a2);
			}
			else if (ndf==NDF_PHONG) {
				float n = irr_glsl_alpha2_to_phong_exp(a2);
				bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(currBSDFParams.isotropic, currInteraction.isotropic, n, a2);
			}
			else if (ndf==NDF_AS) {
				float nx = irr_glsl_alpha2_to_phong_exp(a2);
				float ny = irr_glsl_alpha2_to_phong_exp(ay2);
				bxdf_eval_scalar_part = irr_glsl_blinn_phong_cos_eval_DG(currBSDFParams, currInteraction, nx, ny, a2, ay2);
			}
		}

		uvec3 regs = instr_decodeRegisters(instr);
		if (op==OP_DIFFUSE) {
			instr_execute_DIFFUSE(instr, regs, params, bsdf_data);
		}
		else if (op==OP_CONDUCTOR) {
			instr_execute_CONDUCTOR(instr, regs, bxdf_eval_scalar_part, params, bsdf_data);
		}
		else if (op==OP_PLASTIC) {
			instr_execute_PLASTIC(instr, regs, bxdf_eval_scalar_part, params, bsdf_data);
		}
		else if (op==OP_COATING) {
			instr_execute_COATING(instr, regs, params, bsdf_data);
		}
		else if (op==OP_DIFFTRANS) {
			instr_execute_DIFFTRANS(instr, regs, params, bsdf_data);
		}
		else if (op==OP_DIELECTRIC) {
			instr_execute_DIELECTRIC(instr, regs, params, bsdf_data);
		}
		else if (op==OP_BLEND) {
			instr_execute_BLEND(instr, regs, params, bsdf_data);
		}
		else if (op==OP_BUMPMAP) {
			instr_execute_BUMPMAP(instr, L);
		}
		else if (op==OP_SET_GEOM_NORMAL) {
			instr_execute_SET_GEOM_NORMAL(L);
		}
	}
}
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_
#define _NBL_BUILTIN_MATERIAL_COMPILER_GLSL_RASTERIZATION_IMPL_INCLUDED_

#include <nbl/builtin/glsl/material_compiler/common.glsl>

void nbl_glsl_MC_instr_eval_execute(in nbl_glsl_MC_instr_t instr, in nbl_glsl_MC_precomputed_t precomp, inout nbl_glsl_LightSample s, inout nbl_glsl_MC_microfacet_t microfacet)
{
	const uint op = nbl_glsl_MC_instr_getOpcode(instr);

	#ifdef OP_SET_GEOM_NORMAL
	if (op==OP_SET_GEOM_NORMAL)
		nbl_glsl_MC_instr_execute_SET_GEOM_NORMAL(precomp);
	else
	#endif
	{
		const nbl_glsl_MC_RegID_t regs = nbl_glsl_MC_instr_decodeRegisters(instr);
		#ifdef OP_BUMPMAP
		if (op==OP_BUMPMAP)
		{
			nbl_glsl_MC_instr_execute_BUMPMAP(regs.srcA,precomp);
		}
		else
		#endif
		{			
			// dont worry about pushing this computation into `is_bxdf` branch if OP_COATING not defined, compiler should be able to do that
			const bool is_not_brdf = !nbl_glsl_MC_op_isBRDF(op);
			const float NdotV = nbl_glsl_conditionalAbsOrMax(is_not_brdf,currInteraction.inner.isotropic.NdotV,0.f);
			const float NdotL = nbl_glsl_conditionalAbsOrMax(is_not_brdf,s.NdotL,0.f);
			const bool skip = false
			#ifdef OP_THINDIELECTRIC
				&& (op != OP_THINDIELECTRIC)
			#endif
			#ifdef OP_DELTATRANS
				&& (op != OP_DELTATRANS)
			#endif
			;
			const bool run = !skip && (NdotL > nbl_glsl_FLT_MIN) && (NdotV > nbl_glsl_FLT_MIN);

			// don't worry about unused variables, compiler should be able to spot them
			nbl_glsl_MC_params_t params;
			mat2x3 ior;
			mat2x3 ior2;
			{
				nbl_glsl_MC_bsdf_data_t bsdf_data;
				if (!skip)
				{
					bsdf_data = nbl_glsl_MC_fetchBSDFDataForInstr(instr);
					params = nbl_glsl_MC_instr_getParameters(instr,bsdf_data);
				}
				ior[0] = vec3(1.00001f); // avoid issues from uninitialized memory containing NaNs
				if (run)
				{
					// for dielectrics the IoR is already fetched as oriented
					ior = nbl_glsl_MC_bsdf_data_decodeIoR(bsdf_data,op);
					ior2 = matrixCompMult(ior,ior);
				}
			}


			//
			nbl_glsl_MC_bxdf_spectrum_t result;
			if (nbl_glsl_MC_op_isBXDF(op))
			{
				result = nbl_glsl_MC_instr_bxdf_eval_and_pdf_common(
					instr,op,is_not_brdf,params,ior,ior2,
					#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
					precomp.N,
					#endif
					NdotV,NdotL,
					s,microfacet,run
				).value;
			}
			else
			{
				nbl_glsl_MC_bxdf_spectrum_t srcA,srcB;
				nbl_glsl_MC_readReg(regs.srcA,srcA);
				nbl_glsl_MC_readReg(regs.srcB,srcB);

				vec3 dummySpectrum;
				#ifdef OP_COATING
				#ifdef OP_BLEND
				if (op==OP_COATING)
				#endif
				{
					// TODO: would be cool to use some assumptions about srcA==dst (coating being in the output register's place already) to skip any register writing when fresnel=1
					//vec3 thickness_sigma = params_getSigmaA(params);
					float dummyScalar;
					result = nbl_glsl_MC_coatedDiffuse(srcA,srcB,/*thickness_sigma,*/ior[0],ior2[0],NdotV,NdotL,dummySpectrum,dummyScalar);
				}
				#endif
				#ifdef OP_BLEND
				#ifdef OP_COATING
				else
				#endif
				{
					result = nbl_glsl_MC_instr_execute_BLEND(srcA,srcB,params,dummySpectrum);
				}
				#endif
			}

			nbl_glsl_MC_writeReg(regs.dst,result);
		}
	}
}

nbl_glsl_MC_bxdf_spectrum_t nbl_glsl_MC_runEvalStream(in nbl_glsl_MC_precomputed_t precomp, in nbl_glsl_MC_instr_stream_t stream, in vec3 L)
{
	nbl_glsl_MC_setCurrInteraction(precomp);
	nbl_glsl_LightSample s = nbl_glsl_createLightSample(L,currInteraction.inner);
	nbl_glsl_MC_microfacet_t microfacet;
	microfacet.inner = nbl_glsl_calcAnisotropicMicrofacetCache(currInteraction.inner, s);
	nbl_glsl_MC_finalizeMicrofacet(microfacet);

	for (uint i=0u; i<stream.count; ++i)
	{
		nbl_glsl_MC_instr_t instr = nbl_glsl_MC_fetchInstr(stream.offset+i);

		nbl_glsl_MC_instr_eval_execute(instr, precomp, s, microfacet);

		#if defined(OP_SET_GEOM_NORMAL)||defined(OP_BUMPMAP)
		const uint op = nbl_glsl_MC_instr_getOpcode(instr);
		if (
		#ifdef OP_SET_GEOM_NORMAL
			op==OP_SET_GEOM_NORMAL
		#ifdef OP_BUMPMAP
			||
		#endif
		#endif
		#ifdef OP_BUMPMAP
			op==OP_BUMPMAP
		#endif
		) {
			nbl_glsl_MC_updateLightSampleAfterNormalChange(s);
			nbl_glsl_MC_updateMicrofacetCacheAfterNormalChange(s,microfacet);
		}
		#endif
	}

	//result is always in regs 0,1,2
	nbl_glsl_MC_bxdf_spectrum_t retval;
	nbl_glsl_MC_readReg(0u,retval);
	return retval;
}

#endif
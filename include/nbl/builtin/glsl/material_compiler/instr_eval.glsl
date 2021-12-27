#ifndef NBL_GLSL_MC_INSTR_EVAL_EXECUTE_FUNCNAME
#error "Need to define NBL_GLSL_MC_INSTR_EVAL_EXECUTE_FUNCNAME"
#endif

void NBL_GLSL_MC_INSTR_EVAL_EXECUTE_FUNCNAME(
	in nbl_glsl_MC_instr_t instr,
	in nbl_glsl_MC_precomputed_t precomp,
	in nbl_glsl_LightSample s,
	in nbl_glsl_MC_microfacet_t _microfacet,
	in bool skip
)
{
	const uint op = nbl_glsl_MC_instr_getOpcode(instr);
	const nbl_glsl_MC_RegID_t regs = nbl_glsl_MC_instr_decodeRegisters(instr);

	const bool is_bxdf_or_combiner = nbl_glsl_MC_op_isBXDForCoatOrBlend(op);
	const bool is_bxdf = nbl_glsl_MC_op_isBXDF(op);
	const bool is_not_brdf = !nbl_glsl_MC_op_isBRDF(op);

#if 1
	nbl_glsl_MC_params_t params;
	nbl_glsl_MC_bsdf_data_t bsdf_data;
	mat2x3 ior;
	mat2x3 ior2;
	nbl_glsl_MC_microfacet_t microfacet;

	// TODO: should this include the NdotV term?
	const bool run = !skip;

	if (is_bxdf_or_combiner && run)
	{
		bsdf_data = nbl_glsl_MC_fetchBSDFDataForInstr(instr);
		ior = nbl_glsl_MC_bsdf_data_decodeIoR(bsdf_data, op);
		ior2 = matrixCompMult(ior, ior);
		params = nbl_glsl_MC_instr_getParameters(instr, bsdf_data);
	}

	const float NdotV = nbl_glsl_conditionalAbsOrMax(is_not_brdf, currInteraction.inner.isotropic.NdotV, 0.0);
	const float NdotL = nbl_glsl_conditionalAbsOrMax(is_not_brdf, s.NdotL, 0.0);

	nbl_glsl_MC_eval_pdf_aov_t result;
	result.value = vec3(0.f);
#ifdef GEN_CHOICE_STREAM
	result.pdf = 0.0;
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
	result.aov.albedo = vec3(0.f);
	result.aov.throughputFactor = 0.f;
	result.aov.normal = vec3(0.f);
#endif
#endif
	if (is_bxdf && run && (NdotV > nbl_glsl_FLT_MIN))
	{
		//speculative execution
		uint ndf = nbl_glsl_MC_instr_getNDF(instr);
		float a = nbl_glsl_MC_params_getAlpha(params);
		float a2 = a*a;
#ifdef ALL_ISOTROPIC_BXDFS
		float one_minus_a2 = 1.0 - a2;
#else
		float ay = nbl_glsl_MC_params_getAlphaV(params);
		float ay2 = ay*ay;
#endif
		const vec3 albedo = nbl_glsl_MC_params_getReflectance(params);

#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
		result.aov.normal = precomp.N;
#endif

#if defined(OP_DIFFUSE) || defined(OP_DIFFTRANS)
		if (nbl_glsl_MC_op_isDiffuse(op))
		{
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
			result.aov.albedo = albedo;
#endif
			if (NdotL > nbl_glsl_FLT_MIN)
			{
				float pdf;
				result.value = albedo * nbl_glsl_oren_nayar_cos_remainder_and_pdf_wo_clamps(pdf, a2, s.VdotL, NdotL, NdotV);
				if (is_not_brdf)
					pdf *= 0.5f;
				result.value *= pdf;
				#ifdef GEN_CHOICE_STREAM
				result.pdf = pdf;
				#endif
			}
		}
		else
#endif
#if defined(OP_CONDUCTOR) || defined(OP_DIELECTRIC)
		{
			const float eta = nbl_glsl_MC_colorToScalar(ior[0]);
			const float rcp_eta = 1.0 / eta;

			bool is_valid = true;
			bool refraction = false;
			if (nbl_glsl_isTransmissionPath(currInteraction.inner.isotropic.NdotV, s.NdotL))
			{
				nbl_glsl_calcAnisotropicMicrofacetCache(microfacet.inner, true, currInteraction.inner.isotropic.V.dir, s.L, currInteraction.inner.T, currInteraction.inner.B, currInteraction.inner.isotropic.N, s.NdotL, s.VdotL, eta, rcp_eta);
				nbl_glsl_MC_finalizeMicrofacet(microfacet);
				refraction = true;
			}
			else
				microfacet = _microfacet;

			// TODO: remove the alpha check, implementation should be numerically stable enough to handle roughness tending to 0, also it doesnt do anything for anisotropic roughnesses right now!
			if (nbl_glsl_isValidVNDFMicrofacet(microfacet.inner.isotropic, is_not_brdf, refraction, s.VdotL, eta, rcp_eta) && a > NBL_GLSL_MC_ALPHA_EPSILON)
			{
				const float TdotV2 = currInteraction.TdotV2;
				const float BdotV2 = currInteraction.BdotV2;
				const float NdotV2 = currInteraction.inner.isotropic.NdotV_squared;
				const float NdotL2 = s.NdotL2;

				const float TdotH2 = microfacet.TdotH2;
				const float BdotH2 = microfacet.BdotH2;
				const float NdotH2 = microfacet.inner.isotropic.NdotH2;

				float G1_over_2NdotV = 0.0;
				float G2_over_G1 = 0.0;
				float ndf_val = 0.0;

				BEGIN_CASES(ndf)
#ifdef NDF_GGX
				CASE_BEGIN(ndf, NDF_GGX) {
#ifdef ALL_ISOTROPIC_BXDFS
					G1_over_2NdotV = nbl_glsl_GGXSmith_G1_wo_numerator(NdotV, NdotV2, a2, one_minus_a2);
					G2_over_G1 = nbl_glsl_ggx_smith_G2_over_G1(NdotL, NdotL2, NdotV, NdotV2, a2, one_minus_a2);
					ndf_val = nbl_glsl_ggx_trowbridge_reitz(a2, NdotH2);
#else
					G1_over_2NdotV = nbl_glsl_GGXSmith_G1_wo_numerator(NdotV, TdotV2, BdotV2, NdotV2, a2, ay2);
					G2_over_G1 = nbl_glsl_ggx_smith_G2_over_G1(NdotL, s.TdotL*s.TdotL, s.BdotL*s.BdotL, NdotL2, NdotV, TdotV2, BdotV2, NdotV2, a2, ay2);
					ndf_val = nbl_glsl_ggx_aniso(TdotH2, BdotH2, NdotH2, a, ay, a2, ay2);
#endif
				} CASE_END
#endif

#ifdef NDF_BECKMANN
				CASE_BEGIN(ndf, NDF_BECKMANN) {
#ifdef ALL_ISOTROPIC_BXDFS
					float lambdaV = nbl_glsl_smith_beckmann_Lambda(NdotV2, a2);
					G1_over_2NdotV = nbl_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
					G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, NdotL2, a2);
					ndf_val = nbl_glsl_beckmann(a2, NdotH2);
#else
					float lambdaV = nbl_glsl_smith_beckmann_Lambda(TdotV2, BdotV2, NdotV2, a2, ay2);
					G1_over_2NdotV = nbl_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
					G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, s.TdotL*s.TdotL, s.BdotL*s.BdotL, NdotL2, a2, ay2);
					ndf_val = nbl_glsl_beckmann(a, ay, a2, ay2, TdotH2, BdotH2, NdotH2);
#endif
				} CASE_END
#endif

#ifdef NDF_PHONG
				CASE_BEGIN(ndf, NDF_PHONG) {
#ifdef ALL_ISOTROPIC_BXDFS
					float lambdaV = nbl_glsl_smith_beckmann_Lambda(NdotV2, a2);
					G1_over_2NdotV = nbl_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
					G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, NdotL2, a2);
					ndf_val = nbl_glsl_beckmann(a2, NdotH2);
#else
					float lambdaV = nbl_glsl_smith_beckmann_Lambda(TdotV2, BdotV2, NdotV2, a2, ay2);
					G1_over_2NdotV = nbl_glsl_smith_G1(lambdaV) / (2.0 * NdotV);
					G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(lambdaV + 1.0, s.TdotL*s.TdotL, s.BdotL*s.BdotL, NdotL2, a2, ay2);
					ndf_val = nbl_glsl_beckmann(a, ay, a2, ay2, TdotH2, BdotH2, NdotH2);
#endif
				} CASE_END
#endif
				CASE_OTHERWISE
				{} //else "empty braces"
				END_CASES

#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
#ifdef ALL_ISOTROPIC_BXDFS
					result.aov.throughputFactor = nbl_glsl_MC_aov_t_specularThroughputFactor(a2);
#else
					result.aov.throughputFactor = nbl_glsl_MC_aov_t_specularThroughputFactor(a2, ay2);
#endif
#endif

				const float pdf = nbl_glsl_smith_VNDF_pdf_wo_clamps(ndf_val, G1_over_2NdotV);
				float remainder_scalar_part = G2_over_G1;

				const float VdotH = abs(microfacet.inner.isotropic.VdotH);

				// compute fresnel for the microfacet
				vec3 fr;
				// computing it again for albedo is unfortunately quite expensive, but I have no other choice
#ifdef OP_CONDUCTOR
#ifdef OP_DIELECTRIC
				if (op == OP_CONDUCTOR)
#endif
				{
					fr = nbl_glsl_fresnel_conductor(ior[0],ior[1],VdotH);
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
					result.aov.albedo = nbl_glsl_fresnel_conductor(ior[0],ior[1],NdotV);
#endif
				}
#endif
#ifdef OP_DIELECTRIC
#ifdef OP_CONDUCTOR
				else
#endif
				{
					const float eta2 = eta*eta;
					// TODO: would be nice not to have monochrome dielectrics
					fr = vec3(nbl_glsl_fresnel_dielectric_common(eta2,VdotH));
#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
					result.aov.albedo = vec3(1.0);
#endif
				}
#endif

				float eval_scalar_part = remainder_scalar_part*pdf;
#ifndef NO_BSDF
				if (is_not_brdf)
				{
					float LdotH = microfacet.inner.isotropic.LdotH;
					float VdotHLdotH = microfacet.inner.isotropic.VdotH * LdotH;
					LdotH = abs(LdotH);
#ifdef NDF_GGX
					if (ndf == NDF_GGX)
						eval_scalar_part = nbl_glsl_ggx_microfacet_to_light_measure_transform(eval_scalar_part, NdotL, refraction, VdotH, LdotH, VdotHLdotH, eta);
					else
#endif
						eval_scalar_part = nbl_glsl_microfacet_to_light_measure_transform(eval_scalar_part, NdotV, refraction, VdotH, LdotH, VdotHLdotH, eta);

					float reflectance = nbl_glsl_MC_colorToScalar(fr);
					reflectance = refraction ? (1.0 - reflectance) : reflectance;
					#ifdef GEN_CHOICE_STREAM
					result.pdf = pdf*reflectance;
					#endif
				}
#endif 
				result.value = fr * eval_scalar_part;
			} 
		}
#endif
		{} // empty else for when there are diffuse ops but arent any specular ones


#if GEN_CHOICE_STREAM>=GEN_CHOICE_WITH_AOV_EXTRACTION
		const float aovContrib = 1.f-result.aov.throughputFactor;
		result.aov.albedo *= aovContrib;
		result.aov.normal *= aovContrib;
#endif
	}

	if (!is_bxdf)
	{
		nbl_glsl_MC_eval_pdf_aov_t srcA,srcB;
		nbl_glsl_MC_readReg(regs.srcA, srcA);
		nbl_glsl_MC_readReg(regs.srcB, srcB);
		BEGIN_CASES(op)
#ifdef OP_COATING
		CASE_BEGIN(op, OP_COATING) {
			//vec3 thickness_sigma = params_getSigmaA(params);
			result = nbl_glsl_MC_instr_execute_COATING(srcA, srcB, /*thickness_sigma,*/ ior[0], ior2[0], NdotV, NdotL);
		} CASE_END
#endif
#ifdef OP_BLEND
		CASE_BEGIN(op, OP_BLEND) {
			result = nbl_glsl_MC_instr_execute_BLEND(srcA, srcB, params);
		} CASE_END
#endif
#ifdef OP_BUMPMAP
		CASE_BEGIN(op, OP_BUMPMAP) {
			nbl_glsl_MC_instr_execute_BUMPMAP(regs.srcA, precomp);
		} CASE_END
#endif
#ifdef OP_SET_GEOM_NORMAL
		CASE_BEGIN(op, OP_SET_GEOM_NORMAL) {
			nbl_glsl_MC_instr_execute_SET_GEOM_NORMAL(precomp);
		} CASE_END
#endif
		CASE_OTHERWISE
		{} //else "empty braces"
		END_CASES
	}
#endif

	if (is_bxdf_or_combiner)
		nbl_glsl_MC_writeReg(regs.dst, result);
}

#undef NBL_GLSL_MC_INSTR_EVAL_EXECUTE_FUNCNAME
// Copyright (C) 2022-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/material_compiler3/CFrontendIR.h"

#include "nbl/builtin/hlsl/complex.hlsl"
#include "nbl/builtin/hlsl/portable/vector_t.hlsl"


namespace nbl::asset::material_compiler3
{

constexpr auto ELL_ERROR = nbl::system::ILogger::E_LOG_LEVEL::ELL_ERROR;
using namespace nbl::system;

bool CFrontendIR::CEmitter::invalid(const SInvalidCheckArgs& args) const
{
	// not checking validty of profile because invalid means no emission profile
	// check for NaN and non invertible matrix
	if (profile && !(hlsl::determinant(profileTransform)>hlsl::numeric_limits<hlsl::float32_t>::min))
	{
		args.logger.log("Emission Profile's Transform is not an invertible matrix!");
		return true;
	}
	return false;
}

bool CFrontendIR::CBeer::invalid(const SInvalidCheckArgs& args) const
{
	if (!args.pool->deref(perpTransparency))
	{
		args.logger.log("Perpendicular Transparency node of correct type must be attached, but is %u of type %s",ELL_ERROR,perpTransparency,args.pool->getTypeName(perpTransparency).data());
		return true;
	}
	return false;
}

bool CFrontendIR::CFresnel::invalid(const SInvalidCheckArgs& args) const
{
	if (!args.pool->deref(orientedRealEta))
	{
		args.logger.log("Oriented Real Eta node of correct type must be attached, but is %u of type %s",ELL_ERROR,orientedRealEta,args.pool->getTypeName(orientedRealEta).data());
		return true;
	}
	if (const auto imagEta = args.pool->deref(orientedImagEta); imagEta)
	{
		if (args.isBTDF)
		{
			const auto knotCount = imagEta->getKnotCount();
			// TODO: check all knots have a scale of 0
		}
	}
	else
	{
		args.logger.log("Oriented Imaginary Eta node of correct type must be attached, but is %u of type %s",ELL_ERROR,orientedImagEta,args.pool->getTypeName(orientedImagEta).data());
		return true;
	}
	return false;
}

bool CFrontendIR::COrenNayar::invalid(const SInvalidCheckArgs& args) const
{
	if (!ndParams)
	{	
		args.logger.log("Normal Distribution Parameters are invalid",ELL_ERROR);
		return true;
	}
	return false;
}

bool CFrontendIR::CCookTorrance::invalid(const SInvalidCheckArgs& args) const
{
	if (!ndParams)
	{	
		args.logger.log("Normal Distribution Parameters are invalid",ELL_ERROR);
		return true;
	}
	if (args.isBTDF && !args.pool->deref(orientedRealEta))
	{
		args.logger.log("Cook Torrance BTDF requires the Index of Refraction to compute the refraction direction, but is %u of type %s",ELL_ERROR,orientedRealEta,args.pool->getTypeName(orientedRealEta).data());
		return true;
	}
	return false;
}


auto CFrontendIR::reciprocate(const TypedHandle<const IExprNode> other) -> TypedHandle<IExprNode>
{
	if (const auto* in=deref<CFresnel>({.untyped=other.untyped}); in)
	{
		auto fresnelH = _new<CFresnel>();
		auto* fresnel = deref(fresnelH);
		*fresnel = *in;
		fresnel->reciprocateEtas = ~in->reciprocateEtas;
		return fresnelH;
	}
	assert(false); // unimplemented
	return {};
}

auto CFrontendIR::createNamedFresnel(const std::string_view name) -> TypedHandle<CFresnel>
{
	using complex32_t = hlsl::complex_t<float>;
	using spectral_complex_t = hlsl::portable_vector_t<complex32_t,3>;
	const static core::map<std::string_view,spectral_complex_t> creationLambdas = {
#define SPECTRUM_MACRO(R,G,B,X,Y,Z) spectral_complex_t(complex32_t(R,X),complex32_t(G,Y),complex32_t(B,Z))
		{"a-C",				SPECTRUM_MACRO(1.6855f, 1.065f, 1.727f,		0.0f, 0.009f, 0.0263f)},			// there is no "a-C", but "a-C:H; data from palik"
		{"Ag",				SPECTRUM_MACRO(0.059481f, 0.055090f, 0.046878f,		4.1367f, 3.4574f, 2.8028f)},
		{"Al",				SPECTRUM_MACRO(1.3404f, 0.95151f, 0.68603f,		7.3509f, 6.4542f, 5.6351f)},
		{"AlAs",			SPECTRUM_MACRO(3.1451f, 3.2636f, 3.4543f,		0.0012319f, 0.0039041f, 0.012940f)},
		{"AlAs_palik",		SPECTRUM_MACRO(3.145f, 3.273f, 3.570f,		0.0f, 0.000275f, 1.56f)},
		{"Au",				SPECTRUM_MACRO(0.21415f, 0.52329f, 1.3319f,		3.2508f, 2.2714f, 1.8693f)},
		{"Be",				SPECTRUM_MACRO(3.3884f, 3.2860f, 3.1238f,		3.1692f, 3.1301f, 3.1246f)},
		{"Be_palik",		SPECTRUM_MACRO(3.46f, 3.30f, 3.19f,		3.18f, 3.18f, 3.16f)},
		{"Cr",				SPECTRUM_MACRO(3.2246f, 2.6791f, 2.1411f,		4.2684f, 4.1664f, 3.9300f)},
		{"CsI",				SPECTRUM_MACRO(1.7834f, 1.7978f, 1.8182f,		0.0f, 0.0f, 0.0f)},
		{"CsI_palik",		SPECTRUM_MACRO(1.78006f, 1.79750f, 1.82315,		0.0f, 0.0f, 0.0f)},
		{"Cu",				SPECTRUM_MACRO(0.32075f,1.09860f,1.2469f,		3.17900f,2.59220f,2.4562)},
		{"Cu_palik",		SPECTRUM_MACRO(0.32000f, 1.04f, 1.16f,		3.15000f, 2.59f, 2.4f)},
		{"Cu20",			SPECTRUM_MACRO(2.975f, 3.17f, 3.075f,		0.122f, 0.23f, 0.525f)},  // out of range beyond 2.5 um refractiveindex.info and similar websites, so data applied is same as from palik's data
		{"Cu20_palik",	SPECTRUM_MACRO(2.975f, 3.17f, 3.075f,		0.122f, 0.23f, 0.525f)},
		{"d-C",			SPECTRUM_MACRO(2.4123f, 2.4246f, 2.4349f,		0.0f, 0.0f, 0.0f)},
		{"d-C_palik",		SPECTRUM_MACRO(2.4137f, 2.4272f, 2.4446f,		0.0f, 0.0f, 0.0f)},
		{"Hg",			SPECTRUM_MACRO(1.8847f, 1.4764f, 1.1306f,		5.1147f, 4.5410f, 3.9896f)},
		{"Hg_palik",		SPECTRUM_MACRO(1.850f, 1.460f, 1.100f,		5.100f, 4.600f, 3.990f)},
		//{"HgTe",			SPECTRUM_MACRO(,,,		,,)},						// lack of length wave range for our purpose https://www.researchgate.net/publication/3714159_Dispersion_of_refractive_index_in_degenerate_mercury_cadmium_telluride
		//{"HgTe_palik",		SPECTRUM_MACRO(,,,		,,)},					// the same in palik (wavelength beyond 2 um)			
		{"Ir",			SPECTRUM_MACRO(2.4200f, 2.0795f, 1.7965f,		5.0665f, 4.6125f, 4.1120f)},
		{"Ir_palik",		SPECTRUM_MACRO(2.44f, 2.17f, 1.87f,		4.52f, 4.24f, 3.79f)},
		{"K",			SPECTRUM_MACRO(0.052350f, 0.048270f, 0.042580f,		1.6732f, 1.3919f, 1.1195f)},
		{"K_palik",		SPECTRUM_MACRO(0.0525f, 0.0483f, 0.0427f,		1.67f, 1.39f, 1.12f)},
		{"Li",			SPECTRUM_MACRO(0.14872f, 0.14726f, 0.19236f,		2.9594f, 2.5129f, 2.1144f)},
		{"Li_palik",		SPECTRUM_MACRO(0.218f, 0.2093f, 0.229f,		2.848f, 2.369f, 2.226f)},
		{"MgO",			SPECTRUM_MACRO(1.7357f, 1.7419f, 1.7501f,		0.0f, 0.0f, 0.0f)},
		{"MgO_palik",		SPECTRUM_MACRO(1.7355f, 1.7414f, 1.74975f,		0.0f, 0.0f, 1.55f)},				 // Handbook of optical constants of solids vol 2 page 951, weird k compoment alone, no measurements and resoults
		{"Mo",			SPECTRUM_MACRO(0.76709f, 0.57441f, 0.46711f,		8.5005f, 7.2352f, 6.1383f)},	   	// https://refractiveindex.info/?shelf=main&book=Mo&page=Werner comparing with palik - weird
		{"Mo_palik",		SPECTRUM_MACRO(3.68f, 3.77f, 3.175f,		3.51f, 3.624f, 3.56f)},
		{"Na_palik",			SPECTRUM_MACRO(0.0522f, 0.061f, 0.0667f,		2.535f, 2.196f, 1.861f)},
		{"Nb",			SPECTRUM_MACRO(2.2775f, 2.2225f, 2.0050f,		3.2500f, 3.1325f, 3.0100f)},
		{"Nb_palik",			SPECTRUM_MACRO(2.869f, 2.9235f, 2.738f,		2.867f, 2.8764f, 2.8983f)},
		{"Ni_palik",			SPECTRUM_MACRO(1.921f, 1.744f, 1.651f,		3.615f, 3.168f, 2.753f)},
		{"Rh",			SPECTRUM_MACRO(2.8490f, 2.6410f, 2.4310f,		3.5450f, 3.3150f, 3.1190f)},
		{"Rh_palik",		SPECTRUM_MACRO(2.092f, 1.934f, 1.8256f,		5.472f, 4.902f, 4.5181f)},
		{"Se",			SPECTRUM_MACRO(1.4420f, 1.4759f, 1.4501f,		0.018713f, 0.10233f, 0.18418f)},
		{"Se_palik",		SPECTRUM_MACRO(3.346f, 3.013f, 3.068f,		0.6402f, 0.6711f, 0.553f)},
		{"SiC",			SPECTRUM_MACRO(2.6398f, 2.6677f, 2.7069f,		0.0f, 0.0f, 0.0f)},
		{"SiC_palik",		SPECTRUM_MACRO(2.6412f, 2.6684f, 2.7077f,		0.0f, 0.0f, 0.0f)},
		{"SnTe",			SPECTRUM_MACRO(3.059f, 1.813f, 1.687f,		5.144f, 4.177f, 3.555f)},			   // no data except palik's resources, so data same as palik
		{"SnTe_palik",		SPECTRUM_MACRO(3.059f, 1.813f, 1.687f,		5.144f, 4.177f, 3.555f)},
		{"Ta",			SPECTRUM_MACRO(1.0683f, 1.1379f, 1.2243f,		5.5047f, 4.7432f, 4.0988f)},
		{"Ta_palik",		SPECTRUM_MACRO(1.839f, 2.5875f, 2.8211f,		1.997f, 1.8683f, 2.0514f)},
		{"Te",			SPECTRUM_MACRO(4.1277f, 3.2968f, 2.6239f,		2.5658f, 2.8789f, 2.7673f)},
		{"Te_palik",		SPECTRUM_MACRO(5.8101f, 4.5213f, 3.3682f,		2.9428f, 3.7289f, 3.6783f)},
		{"ThF4",			SPECTRUM_MACRO(1.5113f, 1.5152f, 1.5205f,		0.0f, 0.0f, 0.0f)},
		{"ThF4_palik",		SPECTRUM_MACRO(1.520f, 1.5125f, 1.524f,		0.0f, 0.0f, 0.0f)},
		{"TiC",			SPECTRUM_MACRO(3.0460f, 2.9815f, 2.8864f,		2.6585f, 2.4714f, 2.3987f)},
		{"TiC_palik",		SPECTRUM_MACRO(3.0454f, 2.9763, 2.8674f,		2.6589f, 2.4695f, 2.3959f)},
		{"TiO2",			SPECTRUM_MACRO(2.1362f, 2.1729f, 2.2298f,		0.0f, 0.0f, 0.0f)},
		{"TiO2_palik",		SPECTRUM_MACRO(2.5925f, 2.676f, 2.78f,		0.0f, 0.0f, 0.0f)},
		{"VC",			SPECTRUM_MACRO(3.0033f, 2.8936f, 2.8138f,		2.4981f, 2.3046f, 2.1913f)},
		{"VC_palik",		SPECTRUM_MACRO(3.0038f, 2.8951f, 2.8184f,		2.4923f, 2.3107f, 2.1902f)},
		{"V_palik",		SPECTRUM_MACRO(3.512f, 3.671f, 3.2178f,		2.9337, 3.069f, 3.3667f)},
		{"VN",			SPECTRUM_MACRO(2.3429f, 2.2268f, 2.1550f,		2.4506f, 2.1345f, 1.8753f)},
		{"VN_palik",		SPECTRUM_MACRO(2.3418f, 2.2239f, 2.1539f,		2.4498f, 2.1371f, 1.8776f)},		
		{"W",		SPECTRUM_MACRO(0.96133f, 1.5474f, 2.1930f,		6.2902f, 5.1052f, 5.0325f)},
		{"none",			SPECTRUM_MACRO(0.f,0.f,0.f,		0.f,0.f,0.f)}
#undef SPECTRUM_MACRO
	};
	//
	const auto found = creationLambdas.find(name);
	if (found==creationLambdas.end())
		return {};
	//
	const auto frH = _new<CFrontendIR::CFresnel>();
	auto* fr = deref(frH);
	fr->debugInfo = _new<CNodePool::CDebugInfo>(found->first);
	{
		CSpectralVariable::SCreationParams<3> params = {};
		params.getSemantics() = CSpectralVariable::Semantics::Fixed3_SRGB;
		params.knots.params[0].scale = found->second.x.real();
		params.knots.params[1].scale = found->second.y.real();
		params.knots.params[2].scale = found->second.z.real();
		fr->orientedRealEta = _new<CSpectralVariable>(std::move(params));
	}
	{
		CSpectralVariable::SCreationParams<3> params = {};
		params.getSemantics() = CSpectralVariable::Semantics::Fixed3_SRGB;
		params.knots.params[0].scale = found->second.x.imag();
		params.knots.params[1].scale = found->second.y.imag();
		params.knots.params[2].scale = found->second.z.imag();
		fr->orientedImagEta = _new<CSpectralVariable>(std::move(params));
	}
	return frH;
}

void CFrontendIR::printDotGraph(std::ostringstream& str) const
{
	str << "digraph {\n";

	// TODO: track layering depth and indent accordingly?
	// assign in reverse because we want materials to print in order
	core::vector<TypedHandle<const CLayer>> layerStack(m_rootNodes.rbegin(),m_rootNodes.rend());
	core::stack<TypedHandle<const IExprNode>> exprStack;
	while (!layerStack.empty())
	{
		const auto layerHandle = layerStack.back();
		layerStack.pop_back();
		const auto* layerNode = deref(layerHandle);
		//
		const auto layerID = getNodeID(layerHandle);
		str << "\n\t" << getLabelledNodeID(layerHandle);
		//
		if (layerNode->coated)
		{
			str << "\n\t" << layerID << " -> " << getNodeID(layerNode->coated) << "[label=\"coats\"]\n";
			layerStack.push_back(layerNode->coated);
		}
		auto pushExprRoot = [&](const TypedHandle<const IExprNode> root, const std::string_view edgeLabel)->void
		{
			if (!root)
				return;
			str << "\n\t" << layerID << " -> " << getNodeID(root) << "[label=\"" << edgeLabel << "\"]";
			exprStack.push(root);
		};
		pushExprRoot(layerNode->brdfTop,"Top BRDF");
		pushExprRoot(layerNode->btdf,"BTDF");
		pushExprRoot(layerNode->brdfBottom,"Bottom BRDF");
		while (!exprStack.empty())
		{
			const auto entry = exprStack.top();
			exprStack.pop();
			const auto nodeID = getNodeID(entry);
			str << "\n\t" << getLabelledNodeID(entry);
			const auto* node = deref(entry);
			const auto childCount = node->getChildCount();
			if (childCount)
			{
				str << "\n\t" << nodeID << " -> {";
				for (auto childIx=0; childIx<childCount; childIx++)
				{
					const auto childHandle = node->getChildHandle(childIx);
					if (const auto child=deref(childHandle); child)
					{
						str << getNodeID(childHandle) << " ";
						exprStack.push(childHandle);
					}
				}
				str << "}\n";
			}
			// special printing
			node->printDot(str,nodeID);
		}
	}

	// TODO: print image views

	str << "\n}\n";
}

void CFrontendIR::SParameter::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	sstr << "\n\t" << selfID << "[label=\"scale = " << std::to_string(scale);
	if (view)
	{
		sstr << "\\nchannel = " << std::to_string(viewChannel);
		const auto& viewParams = view->getCreationParameters();
		sstr << "\\nWraps = {" << sampler.TextureWrapU;
		if (viewParams.viewType!=ICPUImageView::ET_1D && viewParams.viewType!=ICPUImageView::ET_1D_ARRAY)
			sstr << "," << sampler.TextureWrapV;
		if (viewParams.viewType==ICPUImageView::ET_3D)
			sstr << "," << sampler.TextureWrapW;
		sstr << "}\\nBorder = " << sampler.BorderColor;
		// don't bother printing the rest, we really don't care much about those
	}
	sstr << "\"]";
	// TODO: do specialized printing for image views (they need to be gathered into a view set -> need a printing context struct)
	/*
	struct SDotPrintContext
	{
		std::ostringstream* sstr;
		core::unordered_map<ICPUImageView*,core::blake3_hash>* usedViews;
		uint16_t indentation = 0;
	};
	*/
	if (view)
		sstr << "\n\t" << selfID << " -> _view_" << std::to_string(reinterpret_cast<const uint64_t&>(view));
}

core::string CFrontendIR::CSpectralVariable::getLabelSuffix() const
{
	if (getKnotCount()<2)
		return "";
	constexpr const char* SemanticNames[] =
	{
		"\\nSemantics = Fixed3_SRGB",
		"\\nSemantics = Fixed3_DCI_P3",
		"\\nSemantics = Fixed3_BT2020",
		"\\nSemantics = Fixed3_AdobeRGB",
		"\\nSemantics = Fixed3_AcesCG"
	};
	auto pWonky = reinterpret_cast<const SCreationParams<2>*>(this+1);
	return SemanticNames[static_cast<uint8_t>(pWonky->getSemantics())];
}
void CFrontendIR::CSpectralVariable::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	auto pWonky = reinterpret_cast<const SCreationParams<1>*>(this+1);
	pWonky->knots.printDot(getKnotCount(),sstr,selfID);
}

void CFrontendIR::CEmitter::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	if (profile)
		profile.printDot(sstr,selfID);
	if (profile.view)
	{
		const auto transformNodeID = selfID+"_pTform";
		sstr << "\n\t" << transformNodeID << " [label=\"";
		printMatrix(sstr,profileTransform);
		sstr << "\"]";
		// connect up
		sstr << "\n\t" << selfID << " -> " << transformNodeID << "[label=\"Profile Transform\"]";
	}
}

void CFrontendIR::CFresnel::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
}

void CFrontendIR::IBxDF::SBasicNDFParams::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	constexpr const char* paramSemantics[] = {
		"dh/du",
		"dh/dv",
		"alpha_u",
		"alpha_v"
	};
	SParameterSet<4>::printDot(sstr,selfID,paramSemantics);
	if (hlsl::determinant(reference)>0.f)
	{
		const auto referenceID = selfID+"_reference";
		sstr << "\n\t" << referenceID << " [label=\"";
		printMatrix(sstr,reference);
		sstr << "\"]";
		sstr << "\n\t" << selfID << " -> " << referenceID << " [label=\"Stretch Reference\"]";
	}
}

void CFrontendIR::COrenNayar::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	ndParams.printDot(sstr,selfID);
}

void CFrontendIR::CCookTorrance::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	ndParams.printDot(sstr,selfID);
}

}
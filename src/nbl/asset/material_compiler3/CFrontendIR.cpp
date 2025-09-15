// Copyright (C) 2022-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/material_compiler3/CFrontendIR.h"


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
	if (!args.pool->deref(orientedImagEta))
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
	assert(false); // unimplemented
	return {};
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
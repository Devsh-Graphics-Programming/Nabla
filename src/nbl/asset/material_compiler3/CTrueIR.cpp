// Copyright (C) 2022-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#define _NBL_ASSET_MATERIAL_COMPILER3_C_TRUE_IR_CPP_
#include "nbl/asset/material_compiler3/CTrueIR.h"

#include "nbl/builtin/hlsl/complex.hlsl"
#include "nbl/builtin/hlsl/portable/vector_t.hlsl"


namespace nbl::asset::material_compiler3
{
constexpr auto ELL_ERROR = nbl::system::ILogger::E_LOG_LEVEL::ELL_ERROR;
constexpr auto ELL_DEBUG = nbl::system::ILogger::E_LOG_LEVEL::ELL_DEBUG;
using namespace nbl::system;

// TODO: Move these notes, they're for the backend.
// 
// Any material (unless its emission only) requires a shading normal to be computed and kept, this should be the first register.
// Materials are primitive agnostic, don't assume triangles, so any attribute interpolation happens in Material System attribute getting callbacks.
// In practice most renderers will apply them on triangles, the advice is to keep triangle indices in own live state raw or compressed (uint32_t base, and then uint16_t diffs).
// To use the material outside of Raytracing or Visibility Buffer one could use HW interpolators, so all interpolated attributes will sit in global input variables.
// Whenever Material Compiler calls `getUV` or `getTBN` its the user's choice how to implement that.
// 
// However whenever HW interpolators are not used, to compute a UV it costs:
// - 3x index buffer fetch (from primitive ID and index buffer BDA)
// - 3x dependent UV attribute fetch
// - 2 FMA per channel to interpolate
// And to compute a TBN its:
// - 3x index buffer fetch (from primitive ID and index buffer BDA)
// - 3x dependent fetch of Tangent (and optional Bitangent), or Rotation Around normal + scales
// - 2 FMA per channel to interpolate
// - 1 reconstruction of a TBN matrix (can be expensive, i.e. from a quaternion or include renormalizations of interpolated tangents)
// For this reason, it makes sense to spill interpolated UVs and Tangent+BitangentScale instead of recomputing.
// The material compiler will keep a bitmask of UVs and TBNs (TBN depends on UV set) already requested and will spill if user marks sets as expensive to recompute.
// 
// If everyone is using the same TBN set, we could back up the interaction and sample because these are just V and L transformed into the TBN.
// In theory this requires us to use anisotropic interactions whenever the isotropic BRDF has a normal perturbation which seems like an overhead.
// However specular AA mitigations which derive roughness from per-pixel surface curvature and path regularizations would force this upon us anyway.
// Problem is that importance sampling requires us to transform a sampled tangent space L into worldspace and that requires knowing the TBN.
// I can see some optimizations for TBN calc where we only recompute the tangent space .xy of L and V for each TBN since N stays constant and NdotX2 can be known.
// Since we must have full TBN for anisotropic BxDF, tangent space V can be computed quicker than fetching it. 
// 
// Pertubed normal can be spilled or refetched. If it gets refetched some coefficients need to be recomputed which allow for taking dot products of V and L with it.
// If the BRDF is anisotropic, a small nested TBN needs construction so perturbed normal is 3 coefficients.
// Spilling as a float16_t3 or float16_t4 is identical memory traffic to a bilinear derivative map tap assuming UNORM8 storage.
// Tangents consistent with unperturbed TBN can be worked out as `B_p = normalize(cross(N_p,T))` and `T_p = cross(B_p,N_p)`, unless `N_p==T` then cross N_p with B instead.
// If tangents want to be rotated/explicitly controlled, then the spilled normal plus a rotation are still 3 coefficients if encoded as a quaternion (painful to decode) or 4 coeffs.
// 
// Already computed shading normals and other BRDF parameters will be computed on demand but also cached (statically or dynamically) in registers.
// Register leftovers can be passed between command streams to not re-fetch or recompute BxDF parameters.
// Register allocation should spill whole quantities (all of RGB of a color or XYZ of a normal) not singular channels.
// Registers spilled should be the ones with least upcoming uses (should assume streams will run as AOV emission, AOV throughputs, Emission, NEE Eval, Generate, Quotient order)
// Relative cost to recompute or refetch should be included in the decision whether to keep around when register allocating.
// Perturbed Normals should be spilled to VRAM instead of recomputed as they require fetching at least a quaternion and a texture sample.
// Spill should be a big circular buffer allocated per-subgroup - hard to control in Pipeline Shaders https://github.com/KhronosGroup/Vulkan-Docs/issues/2717
// So need to allocate worst case spill for all rays in a dispatch (although can do persistent threads and reduce the dispatch size a little)


CTrueIR::CTrueIR(creation_params_type&& params) : CNodePool(std::move(params)), m_basicNodes(this)
{
	reset();
}

CTrueIR::SBasicNodes::SBasicNodes(CTrueIR* ir)
{
	auto& pool = ir->getObjectPool();
	//
	blackHoleBxDF = pool.emplace<CContributorSum>();
	{
		auto* const node = pool.deref(blackHoleBxDF._const_cast());
		node->product = {};
		const bool success = node->recomputeHash(pool);
		assert(success);
		ir->m_uniqueNodes[node->getHash()] = blackHoleBxDF;
	}
	//
	scalarNegation = pool.emplace<CSpectralVariableFactor>(uint8_t(1));
	{
		auto* const node = pool.deref(scalarNegation._const_cast());
		node->setParameter(0,{.scale=-1.f});
		const bool success = node->recomputeHash(pool);
		assert(success);
		ir->m_uniqueNodes[node->getHash()] = scalarNegation;
	}
	// we never compute the hashes on these ones, they're supposed to have invalid hash
	errorLayer = pool.emplace<COrientedLayer>();
	{
		auto* const node = pool.deref(errorLayer._const_cast());
		node->brdfTop = errorBxDF;
		node->firstTransmission = {};
	}
	errorBxDF = pool.emplace<CContributorSum>();
	{
		auto* const node = pool.deref(errorBxDF._const_cast());
		node->product = pool.emplace<CWeightedContributor>();
		{
			auto* const weighted = pool.deref(node->product._const_cast());
			weighted->contributor = pool.emplace<COrenNayar>();
			const auto factorH = pool.emplace<CSpectralVariableFactor>(uint8_t(3));
			{
				auto* const factor = pool.deref(factorH);
				// make a magenta constant color (can do checkerboard of green & magenta in the future with a small texture)
				for (auto i=0; i<3; i++)
					factor->setParameter(i,{.scale = i!=1 ? 1.f:0.f});
			}
			weighted->factor = factorH;
		}
		node->rest = {};
	}
}

uint32_t CTrueIR::deepCopy(typed_pointer_type<const INode>* out,
    const std::span<const typed_pointer_type<const INode>> orig, const CTrueIR* srcIR)
{
	auto& dstPool = getObjectPool();
	// if not explicitly other, then its ours
	if (!srcIR)
		srcIR = this;
	const auto& srcPool = srcIR->getObjectPool();

	core::vector<typed_pointer_type<const INode>> stack;
	stack.reserve(orig.size() + 32);
	for (const auto& o : orig)
	    stack.push_back(o);
	// use a hashmap to not explore whole DAG
	core::unordered_map<typed_pointer_type<const INode>, typed_pointer_type<INode>> substitutions;
	while (!stack.empty())
	{
		const auto entry = stack.back();
		const auto* const node = srcPool.deref(entry);
		if (!node) // this is an error
			return {};
		const auto nodeType = node->getFinalType();
		const auto childCount = node->getChildCount();
		if (auto& copyH = substitutions[entry]; !copyH)
		{
			for (uint8_t c = 0; c < childCount; c++)
			{
				const auto childH = node->getChildHandle(c);
				if (auto child = srcPool.deref(childH); !child)
					continue; // this is not an error
				stack.push_back(childH);
			}

		    // TODO: how to handle coated in CorrelatedTransmission?

			// copy copies everything including child handles
			copyH = node->copy(this);
			if (!copyH)
				continue;	// TODO: handle invalid nodes somehow
		}
		else
		{
			auto* const copy = dstPool.deref(copyH);
			for (uint8_t c = 0; c < childCount; c++)
			{
				const auto childH = node->getChildHandle(c);
				if (!childH)
					continue;
				auto found = substitutions.find(childH);
				assert(found != substitutions.end());
				copy->setChild(c, found->second);
			}
			// TODO: how to handle coated in CorrelatedTransmission?
			stack.pop_back();
		}
	}

	uint32_t invalidCount = 0;
	auto copies = out;
	for (const auto& o : orig)
	{
		auto copy = substitutions[o];
		const auto* const node = dstPool.deref(copy);
		if (!node) // this is invalid
			invalidCount++;
		else
			*copies = copy;
		copies++;
	}
	return invalidCount;
}

void CTrueIR::SDotPrinter::operator()(std::ostringstream& output)
{
	output << "digraph {\n";

	const auto errorBxDF = m_ir->getBasicNodes().errorBxDF;
	const auto errorLayer = m_ir->getBasicNodes().errorLayer;
	auto drainNodeStack = [&]()->void
	{
		while (!nodeStack.empty())
		{
			const auto entry = nodeStack.back();
			// don't print null nodes
			assert(entry);
			nodeStack.pop_back();
			const auto nodeID = m_ir->getNodeID(entry);
			const auto* node = m_ir->getObjectPool().deref(entry);
			if (!node)
				continue;
			output << "\n\t" << m_ir->getLabelledNodeID(entry);
			const auto childCount = node->getChildCount();
			if (childCount)
			{
				for (auto childIx = 0; childIx < childCount; childIx++)
				{
					const auto childHandle = node->getChildHandle(childIx);
					if (const auto child = m_ir->getObjectPool().deref(childHandle); child)
					{
						output << "\n\t" << nodeID << " -> " << m_ir->getNodeID(childHandle) << "[label=\"" << node->getChildName_impl(childIx) << "\"]";
						const auto visited = visitedNodes.find(childHandle);
						if (visited != visitedNodes.end())
							continue;
						nodeStack.push_back(childHandle);
						visitedNodes.insert(childHandle);
					}
				}
			}
			if (node->getFinalType() == INode::EFinalType::CCorellatedTransmission)
			{
				const auto* transmission = dynamic_cast<const CCorellatedTransmission*>(node);
				layerStack.push_back(transmission->coated);
			}
			// special printing
			node->printDot(output, nodeID);
		}
	};
	drainNodeStack();

	while (!layerStack.empty())
	{
		const auto layerHandle = layerStack.back();
		layerStack.pop_back();
		// don't print layer nodes multiple times
		const auto visited = visitedNodes.find(layerHandle);
		if (visited != visitedNodes.end())
			continue;
		visitedNodes.insert(layerHandle);
		const auto* layerNode = m_ir->getObjectPool().deref(layerHandle);
		if (!layerNode)
			continue;
		//
		const auto layerID = m_ir->getNodeID(layerHandle);
		output << "\n\t" << m_ir->getLabelledNodeID(layerHandle);
		//
		auto pushNodeRoot = [&](const typed_pointer_type<const INode> root, const std::string_view edgeLabel)->void
		{
			if (!root)
				return;
			// print the link from the layer to the expression
			output << "\n\t" << layerID << " -> " << m_ir->getNodeID(root) << "[label=\"" << edgeLabel << "\"]";
			// but not the expression again
			const auto visited = visitedNodes.find(root);
			if (visited != visitedNodes.end())
				return;
			nodeStack.push_back(root);
			visitedNodes.insert(root);
			drainNodeStack();
		};
		pushNodeRoot(layerNode->brdfTop, "Top BRDF");
		pushNodeRoot(layerNode->firstTransmission, "Corellated Trans");
	}

	// TODO: print image views

	output << "\n}\n";
}

CTrueIR::SRewriteSession::~SRewriteSession()
{
	if (success)
		return;
	// if session was not a success, then clean up all the nodes
	for (const auto& handle : createdNodes)
		args.dst->getObjectPool()._delete(handle,1);
}

bool CTrueIR::SRewriteSession::rewrite(typed_pointer_type<const COrientedLayer>& oriented)
{
	if (!success)
		return false;

	success = args.src==args.dst;
	// TODO: go through the layers
//	assert(success);
	return success;
}

bool CTrueIR::SRewriteSession::rewriteSingleLayer(typed_pointer_type<const COrientedLayer>& oriented)
{
	if (!success)
		return false;
	// ideas for the pass:
	// - skip replace delta transmissions by the layer undernearth, if null then keep as delta
	// - if BTDF has delta transmissions, then via the sampling property hoist next layer into current layer BRDFs with the DeltaTransmission weights applied
	// - order the `CWeightedContributor` within the `CContributorSum` linked list so emitters are first, and so on (null products go last)
	// observations:
	// - Any V-dependent factors cannot be commonalized across layers, most of the CSE has to be done within a layer

	// TODO: combine layer meta with `retval.metadata`

	// TODO: deduplicate, collect metadata and insert into current IR

	success = args.src==args.dst;
	assert(success);
	return success;
}

void CTrueIR::ISpectralVariableFactor::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	printDotParameterSet(*pWonky(), getKnotCount(), sstr, selfID, {});
}

void CTrueIR::CEmitter::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	if (profile)
		profile.printDot(sstr, selfID);
	if (profile.view)
	{
		const auto transformNodeID = selfID + "_pTform";
		sstr << "\n\t" << transformNodeID << " [label=\"";
		printMatrix(sstr, profileTransform);
		sstr << "\"]";
		// connect up
		sstr << "\n\t" << selfID << " -> " << transformNodeID << "[label=\"Profile Transform\"]";
	}
}

void CTrueIR::COrenNayar::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	ndfParams.printDot(sstr, selfID);
}

void CTrueIR::CCookTorrance::printDot(std::ostringstream& sstr, const core::string& selfID) const
{
	ndfParams.printDot(sstr, selfID);
}

template class CTrueIR::CSpectralVariable<CTrueIR::ISpectralVariableFactor>;
}
// Copyright (C) 2022-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
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

auto CTrueIR::addMaterials(const SAddMaterialsArgs& args) -> core::vector<SMaterialHandle>
{
	const auto logger = args.logger;
	if (!args)
	{
		logger.log("Invalid Arguments to `CTrueIR::addMaterials`",ELL_ERROR);
		return {};
	}
	auto& astPool = args.forest->getObjectPool();

	// debug what we're processing
	auto printSubtree = [&](const CFrontendIR::typed_pointer_type<const CFrontendIR::IExprNode> nodeH)->void
	{
		CFrontendIR::SDotPrinter printer(args.forest);
		printer.exprStack.push(nodeH);
		logger.log("Subtree Dot3 : \n%s\n", ELL_DEBUG,printer().c_str());
	};

//	core::unordered_map<const CFrontendIR::IExprNode*,bool> brdfs;
//	core::unordered_map<const CFrontendIR::IExprNode*,bool> btdfs;

	// Some of the things we must canonicalize:
	// A ( f_0 (B + C) + D f_1 ) = f_0 B A + f_0 C A + f_1 D A
	// Expression nodes really come in 4 variants:
	// - add
	// - mul
	// - complement, which is equivalent to 1 ADD (-1 MUL x)
	// - function/other
	// BRDFs can appear only under ADD and MUL nodes, so if we want to canonicalize:
	// 1. The Add above can be ignored, we form full multiplication chain to the top
	// 2. Adds in sibling nodes (below the last add) cause us to have to add a factored copy to the IR
	// DFS from right-to-left (inverse order of adding children to stack), would cause us to keep postifxes of the multiplier chain every time we descend into ADD.
	// We want to essentially visit the parent ADD node again after dealing with its subtree (in-order traversal) then mul chain can be reset just to the parent.
	// If we perform DFS stack push left-to-right, we'll know the contributor already for all the leaf nodes if we push it onto the stack.
	// Then for all other leaf nodes we can accumulate them in the MUL chain, and adding their weighted contributor whenever we're back at an ADD node (be it the ancestor or sibling/cousin).
	// If the contributor is null or multiplied with a null we can keep draining the stack until we're back at its immediate parent ADD node
	struct SContributor
	{
		// the "active" contributor, basically the leftmost item in the subbranch below and ADD
		typed_pointer_type<const IContributor> contributor;
	};
	core::vector<SContributor> contributorStack;
	// Every time we encounter an AST leaf we must add the current contributor together with all the factors multiplied together
	struct SFactor
	{
		// We only track multiplicative factors, we break down every BRDF equally into the canonical form
		typed_pointer_type<const IFactorLeaf> handle;
		uint8_t negate : 1 = false;
		uint8_t monochrome : 1 = true;
		// extend later when allowing variable buckets
		uint8_t liveSpectralChannels : 3 = 0b111;
	};
	// here we keep the multiplication chain unsorted so its each to add/remove nodes as we encounter them
	core::vector<SFactor> mulChain;
	//
	struct StackEntry
	{
		inline bool notVisited() const {return !visited;}

		const CFrontendIR::IExprNode* node;
		// the ancestor ADD node to go back to if we hit a 0 MUL
		uint16_t nonMulImmediateAncestorStackEnd = 0;
		// the length of the `mulChain` at the time we first visited the node
		uint16_t mulChainLen = 0;
		bool visited = false;
		// only relevant for Add nodes
		bool addContributor = false;
	};
	core::vector<StackEntry> exprStack;
	// Multiplication Chain need to be sorted in a canonical order so its easier to spot them being the same
	auto sortMuls = [](const SFactor& lhs, const SFactor& rhs)->bool
	{
		// monochrome is cheaper
		if (lhs.monochrome!=rhs.monochrome)
			return lhs.monochrome;
		// not doing a complement is cheaper
		if (lhs.handle.value==rhs.handle.value)
			return lhs.negate<rhs.negate;
		// but want negations to show up together in the sorted list so easier to put back together
		return lhs.handle.value<rhs.handle.value;
	};
	core::vector<SFactor> mulChainSortScratch;
	auto getContributors = [&](const CFrontendIR::typed_pointer_type<const CFrontendIR::IExprNode> bxdfRootH)->auto
	{
		typed_pointer_type<const CContributorSum> headH = {};
		if (!bxdfRootH)
			return headH;
		printSubtree(bxdfRootH);

		// scratches are initialized
		assert(mulChain.empty());
		assert(contributorStack.empty());
		exprStack.push_back({.node=astPool.deref(bxdfRootH)});
		typed_pointer_type<CContributorSum> tailH = {};
		while (!exprStack.empty())
		{
			auto& entry = exprStack.back();
			using ast_expr_type_e = CFrontendIR::IExprNode::Type;
			const ast_expr_type_e astExprType = entry.node->getType();
			const bool isContributor = astExprType==CFrontendIR::IExprNode::Type::Contributor;
			//
			if (entry.notVisited())
			{
				if (isContributor)
				{
					// TODO actually make the contributor
					const auto contributorType = 0;
					switch (contributorType)
					{
						case 45:
						{
							// TODO: add to contributorStack
							contributorStack.push_back({.contributor={}});
							break;
						}
						// unsupported contributor
						default:
							logger.log("Unsupported contributor type %d skipping subtree",ELL_ERROR,contributorType);
							return m_errorBxDF;
					}
					// dont want to deal with the contributor again
					exprStack.pop_back();
				}
				else
				{
#if 0 // TODO: Other factors
					// spot prefix being null/zero to stop exploring
					// can make the decision wholly on current factor
					bool continueExploring = true;
					if (continueExploring)
					{
						// make the actual factor
						auto derivedFactorH = getObjectPool().emplace<CConstant>();
						// TODO: find the place where to insert it
						{
							// shtuff
						}
						entry.factor = derivedFactorH;
						// add self after making self
						ancestors.push_back(getObjectPool().deref(entry.factor));
						// TODO: push the children nodes onto the stack
						continue;
					}
#endif
					const bool isAdd = astExprType==ast_expr_type_e::Add;
					if (isAdd)
					{
						entry.addContributor = true;
						// Current Add node will perform the job of the parent add node for this subtree
						if (entry.nonMulImmediateAncestorStackEnd)
						exprStack[entry.nonMulImmediateAncestorStackEnd-1].addContributor = false;
					}
					const bool notMul = astExprType!=ast_expr_type_e::Mul;
					// go through children
					const auto childCount = entry.node->getChildCount();
					// add in reverse so stack processes in order
					for (auto childIx=childCount; childIx; )
					{
						// making sure we visit this node again each time a subtree of an Add node is done
						if (isAdd && childIx!=childCount)
						{
							auto& extraEntry = exprStack.emplace_back(entry);
							extraEntry.visited = true;
							extraEntry.addContributor = true;
						}
						// regular exploration
						exprStack.push_back({
							.node = astPool.deref(entry.node->getChildHandle(--childIx)),
							.nonMulImmediateAncestorStackEnd = notMul ? static_cast<uint16_t>(exprStack.size()):entry.nonMulImmediateAncestorStackEnd
						});
					}
				}
				entry.visited = true;
			}
			else
			{
				assert(!isContributor);
				// do stuff now
				switch (astExprType)
				{
					case ast_expr_type_e::Add:
					{
						// we visit leftmost subtrees first so this is the right order
						{
							auto* const tail = getObjectPool().deref(tailH);
							tailH = getObjectPool().emplace<CContributorSum>();
							if (tailH)
								tail->rest = tailH;
							else
								headH = tailH;
						}
						// add current contributor with weight to BxDF Sum
						{
							const auto weightedH = getObjectPool().emplace<CWeightedContributor>();
							getObjectPool().deref(tailH)->product = weightedH;
							auto* weighted = getObjectPool().deref(weightedH);
							weighted->contributor = contributorStack.back().contributor;
							if (!mulChain.empty())
							{
								const CFactorCombiner::SState combinerState = {
									.type = CFactorCombiner::Type::Mul,
									.childCount = mulChain.size()
								};
								// TODO: create the combiner node
								//const auto factorH = getObjectPool().emplace<CFactorCombiner>();
								{
									// every contributor node gets its own SORTED ancestor prefix
									mulChainSortScratch = mulChain;
									std::sort(mulChainSortScratch.begin(),mulChainSortScratch.end(),sortMuls);
									//auto oit = getObjectPool().deref(factorH)->child;
									//for (const auto& mul : mulChainSortScratch)
										//*(oit++) = mul.handle;
								}
								//weighted->factor = factorH;
							}
						}
						// when we are done we need to reset the mul chain back to its original state
						mulChain.resize(entry.mulChainLen);
						break;
					}
					default:
						break;
				}
				exprStack.pop_back();
			}
		}
		// we got all the AST ADD nodes on the way back out
		assert(mulChain.empty());
		assert(contributorStack.empty());
		return headH;
	};
	//
	core::vector<const CFrontendIR::CLayer*> layerStack;
	auto makeOrientedMaterial = [&](const CFrontendIR::typed_pointer_type<const CFrontendIR::CLayer> rootH)->SMaterial::SOriented
	{
		SMaterial::SOriented retval = {};

		// go down through layers and create all the dependencies
		layerStack.clear();
		for (const auto* layer=astPool.deref(rootH); layer; layer=astPool.deref(layer->coated))
		{
			// TODO: actually re-check the expressions for being null after optimization
			bool noTopReflection = !layer->brdfTop;
			bool noTransmission = !layer->btdf;
			// if there's literally nothing on the top level, you can't get to the next layer to retroreflect from it
			if (noTopReflection && noTransmission)
			{
				logger.log("Skipping current layer and farther ones due to no transmission and reflection",ELL_DEBUG);
				break;
			}
			layerStack.push_back(layer);
			// find out rest of the layers don't matter because they're blocked from being seen, its not a complete check
			if (noTransmission)
			{
				logger.log("Skipping remaining layers due to no transmission",ELL_DEBUG);
				break;
			}
			// Only if we're not in the last layer do we care about the bottom BRDF (you can't hit it otherwise)
			// Note that this won't catch the next layer being a blackhole and needs to be undone if it is
			if (layer->coated && layer->brdfBottom)
			{
				// do stuff with brdfBottom
			}
		}
		if (!layerStack.empty())
			retval.metadata |= SMaterial::EMetadataBits::NotBlackhole;
		// then go back up and make the layers
		while (!layerStack.empty())
		{
			const auto* const inLayer = layerStack.back();
			layerStack.pop_back();
			// allocate a layer
			const auto layerH = getObjectPool().emplace<COrientedLayer>();
			auto* const outLayer = getObjectPool().deref(layerH);
			retval.root = layerH;
			// process the BTDF
			//...
			// process the top BRDF
			outLayer->brdfTop = getContributors(inLayer->brdfTop);
			// if BTDF has delta transmissions, then via the sampling property hoist next layer into current layer BRDFs with the DeltaTransmission weights applied
			// hmm this would require decorrellation... because don't want rest of BTDF to affect
			//...
		}
		// skip replace delta transmissions by the layer undernearth, if null then keep as delta

		// AST is Sum Expression to the BRDF nodes
		// We need to keep the Ancestor prefix as an unrolled linked list
		return retval;
	};

	const auto inputMaterials = args.forest->getMaterials();
	core::vector<SMaterialHandle> retval(inputMaterials.size(), {});
	auto outIt = retval.begin();
	for (const auto& rootH : inputMaterials)
	{
		auto& result = *(outIt++);

		const auto* astRoot = astPool.deref<const CFrontendIR::CLayer>(rootH);
		// no material
		if (!astRoot)
		{
			result = BlackholeMaterialHandle;
			continue;
		}
		SMaterial material = {
			.front = makeOrientedMaterial(rootH),
//			.back = makeOrientedMaterial(rootH) // TODO: reverse AST into another tree
		};

		// TODO: better debug info
		if (const auto* debug=astPool.deref<const CDebugInfo>(astRoot->debugInfo); debug && !debug->data().empty())
		{
			material.debugInfo = getObjectPool().emplace<CNodePool::CDebugInfo>(debug->data().data(),debug->data().size());
		}

		//
		result.value = m_materials.size();
		m_materials.push_back(material);
	}

	return retval;
}


}
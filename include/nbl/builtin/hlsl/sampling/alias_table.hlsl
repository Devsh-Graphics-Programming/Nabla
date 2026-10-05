// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_ALIAS_TABLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_ALIAS_TABLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// Packed alias-entry bit layout shared by every packed variant. One 32-bit
// word holds the redirect index in the low Log2N bits and the stay-
// probability quantized as an unorm in the high (32 - Log2N) bits.
//   u * N  = scaled;  bin = floor(scaled);  remainder = scaled - bin
//   if (remainder < getStayProb(word))  -> result = bin
//   else                                -> result = getTarget(word)
// Quantizing the threshold to (32 - Log2N) bits is precision-neutral: `u`
// already consumed Log2N bits of randomness producing `bin`, so `remainder`
// carries exactly that many bits of discriminatory power.
namespace impl
{
template<uint32_t Log2N>
struct AliasBitDecoder
{
	static uint32_t getTarget(uint32_t word)
	{
		return word & ((1u << Log2N) - 1u);
	}
	template<typename T>
	static T getStayProb(uint32_t word)
	{
		const uint32_t unormMax = (~0u) >> Log2N;
		return T(word >> Log2N) / T(unormMax);
	}
};
} // namespace impl

// 8 B entry used by the NBig == true variant. Embeds the bin's own pdf
// alongside the packed word so the common stay-case needs no extra tap.
template<typename T>
struct PackedAliasEntryB
{
	uint32_t packedWord;	// low Log2N: redirect target; high 32-Log2N: stayProb unorm
	T ownPdf;				// pdf of this bin
};


// NBig == false: 4 B packed word per bin + separate pdf[] array. Per sample
// = one 4 B word load + one unconditional 4 B pdf[] tap indexed by the
// selected bin (either the current bin or its redirect). Total 8 B whether
// the sample stays or aliases. Favours small N.
template<typename T, typename Domain, typename Codomain, typename PackedWordAccessor, typename PdfAccessor, uint32_t Log2N
	NBL_PRIMARY_REQUIRES(
		concepts::UnsignedIntegralScalar<Codomain> &&
		concepts::accessors::GenericReadAccessor<PackedWordAccessor, uint32_t, Codomain> &&
		concepts::accessors::GenericReadAccessor<PdfAccessor, T, Codomain>)
struct PackedAliasTableA
{
	using scalar_type = T;
	using domain_type = Domain;
	using codomain_type = Codomain;
	using density_type = scalar_type;
	using weight_type = density_type;
	using decoder = impl::AliasBitDecoder<Log2N>;
	NBL_CONSTEXPR_STATIC_INLINE bool NBig = false;

	struct cache_type
	{
		density_type pdf;
	};

	static PackedAliasTableA create(NBL_CONST_REF_ARG(PackedWordAccessor) _entryAcc, NBL_CONST_REF_ARG(PdfAccessor) _pdfAcc, codomain_type _size)
	{
		PackedAliasTableA retval;
		retval.entryAcc = _entryAcc;
		retval.pdfAcc = _pdfAcc;
		const scalar_type exact = scalar_type(_size);
		retval.tableSizeMinusUlp = nbl::hlsl::bit_cast<scalar_type>(nbl::hlsl::bit_cast<uint32_t>(exact) - 1u);
		return retval;
	}

	codomain_type generate(const domain_type u) NBL_CONST_MEMBER_FUNC
	{
		const scalar_type scaled = u * tableSizeMinusUlp;
		const codomain_type bin = _static_cast<codomain_type>(scaled);
		const scalar_type remainder = scaled - scalar_type(bin);

		uint32_t packedWord;
		entryAcc.template get<uint32_t, codomain_type>(bin, packedWord);
		return hlsl::select(remainder < decoder::template getStayProb<scalar_type>(packedWord), bin, codomain_type(decoder::getTarget(packedWord)));
	}

	codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		const codomain_type result = generate(u);
		pdfAcc.template get<scalar_type, codomain_type>(result, cache.pdf);
		return result;
	}

	density_type forwardPdf(const domain_type u, NBL_CONST_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		return cache.pdf;
	}

	weight_type forwardWeight(const domain_type u, NBL_CONST_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		return cache.pdf;
	}

	density_type backwardPdf(const codomain_type v) NBL_CONST_MEMBER_FUNC
	{
		scalar_type pdf;
		pdfAcc.template get<scalar_type, codomain_type>(v, pdf);
		return pdf;
	}

	weight_type backwardWeight(const codomain_type v) NBL_CONST_MEMBER_FUNC
	{
		return backwardPdf(v);
	}

	PackedWordAccessor entryAcc;
	PdfAccessor pdfAcc;
	scalar_type tableSizeMinusUlp;
};

// NBig == true: 8 B entry {packedWord, ownPdf} + separate pdf[] array. Per
// sample = one 8 B entry load (covers the common stay case where cache
// already has ownPdf). If the sample aliases, a conditional 4 B pdf[target]
// tap fills the cache. Total 8 B stay, 12 B aliased. Favours large N.
template<typename T, typename Domain, typename Codomain, typename EntryAccessor, typename PdfAccessor, uint32_t Log2N
	NBL_PRIMARY_REQUIRES(
		concepts::UnsignedIntegralScalar<Codomain> &&
		concepts::accessors::GenericReadAccessor<EntryAccessor, PackedAliasEntryB<T>, Codomain> &&
		concepts::accessors::GenericReadAccessor<PdfAccessor, T, Codomain>)
struct PackedAliasTableB
{
	using scalar_type = T;
	using domain_type = Domain;
	using codomain_type = Codomain;
	using density_type = scalar_type;
	using weight_type = density_type;
	using entry_type = PackedAliasEntryB<scalar_type>;
	using decoder = impl::AliasBitDecoder<Log2N>;
	NBL_CONSTEXPR_STATIC_INLINE bool NBig = true;

	struct cache_type
	{
		density_type pdf;
	};

	static PackedAliasTableB create(NBL_CONST_REF_ARG(EntryAccessor) _entryAcc, NBL_CONST_REF_ARG(PdfAccessor) _pdfAcc, codomain_type _size)
	{
		PackedAliasTableB retval;
		retval.entryAcc = _entryAcc;
		retval.pdfAcc = _pdfAcc;
		const scalar_type exact = scalar_type(_size);
		retval.tableSizeMinusUlp = nbl::hlsl::bit_cast<scalar_type>(nbl::hlsl::bit_cast<uint32_t>(exact) - 1u);
		return retval;
	}

	codomain_type generate(const domain_type u) NBL_CONST_MEMBER_FUNC
	{
		const scalar_type scaled = u * tableSizeMinusUlp;
		const codomain_type bin = _static_cast<codomain_type>(scaled);
		const scalar_type remainder = scaled - scalar_type(bin);

		entry_type entry;
		entryAcc.template get<entry_type, codomain_type>(bin, entry);
		return hlsl::select(remainder < decoder::template getStayProb<scalar_type>(entry.packedWord), bin, codomain_type(decoder::getTarget(entry.packedWord)));
	}

	codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		const scalar_type scaled = u * tableSizeMinusUlp;
		const codomain_type bin = _static_cast<codomain_type>(scaled);
		const scalar_type remainder = scaled - scalar_type(bin);

		entry_type entry;
		entryAcc.template get<entry_type, codomain_type>(bin, entry);

		const bool stay = remainder < decoder::template getStayProb<scalar_type>(entry.packedWord);
		
		cache.pdf = entry.ownPdf;
		codomain_type result = bin;
		if (!stay)
		{
			const codomain_type target = codomain_type(decoder::getTarget(entry.packedWord));
			pdfAcc.template get<scalar_type, codomain_type>(target, cache.pdf);
			result = target;
		}
		return result;
	}

	density_type forwardPdf(const domain_type u, NBL_CONST_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		return cache.pdf;
	}

	weight_type forwardWeight(const domain_type u, NBL_CONST_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		return cache.pdf;
	}

	density_type backwardPdf(const codomain_type v) NBL_CONST_MEMBER_FUNC
	{
		scalar_type pdf;
		pdfAcc.template get<scalar_type, codomain_type>(v, pdf);
		return pdf;
	}

	weight_type backwardWeight(const codomain_type v) NBL_CONST_MEMBER_FUNC
	{
		return backwardPdf(v);
	}

	EntryAccessor entryAcc;
	PdfAccessor pdfAcc;
	scalar_type tableSizeMinusUlp;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif

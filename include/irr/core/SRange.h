// Copyright (C) 2019 DevSH Graphics Programming Sp. z O.O.
// This file is part of the "IrrlichtBaW".
// For conditions of distribution and use, see LICENSE.md

#ifndef __IRR_S_RANGE_H_INCLUDED__
#define __IRR_S_RANGE_H_INCLUDED__

#include "irr/core/Types.h"

#include "stddef.h"
#include <type_traits>

/*! \file SRange.h
	\brief File containing SRange utility struct for C++11 range loops
*/

namespace irr
{
namespace core
{

template<typename T, typename IteratorType = std::add_pointer_t<T>, typename ConstIteratorType = std::add_pointer_t<const T> >
struct SRange
{
		inline SRange(const IteratorType& _beg, const IteratorType& _end) : m_begin(_beg), m_end(_end) {}
		inline SRange(IteratorType&& _beg, IteratorType&& _end) : m_begin(std::move(_beg)), m_end(std::move(_end)) {}

		inline IteratorType begin() { return m_begin; }
        inline ConstIteratorType begin() const { return m_begin; }
		inline IteratorType end() { return m_end; }
        inline ConstIteratorType end() const { return m_end; }

		inline size_t size() const {return std::distance(m_begin,m_end);}

	private:
		IteratorType m_begin, m_end;
};


} // end namespace core
} // end namespace irr

#endif

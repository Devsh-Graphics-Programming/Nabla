// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_S_RANGE_H_INCLUDED__
#define __NBL_CORE_S_RANGE_H_INCLUDED__

#include "nbl/core/decl/Types.h"

#include "stddef.h"
#include <type_traits>
#include <utility>
#include <iterator>

/*! \file SRange.h
	\brief File containing SRange utility struct for C++11 range loops
*/

namespace nbl::core
{

template<typename T, typename IteratorType = std::add_pointer_t<T>, typename ConstIteratorType = std::add_pointer_t<const T> >
struct NBL_API SRange
{
	public:
		using iterator_type = IteratorType;
		using const_iterator_type = ConstIteratorType;


		inline SRange(const IteratorType& _beg, const IteratorType& _end) : m_begin(_beg), m_end(_end) {}
		inline SRange(IteratorType&& _beg, IteratorType&& _end) : m_begin(std::move(_beg)), m_end(std::move(_end)) {}

		inline IteratorType begin() const { return m_begin; }
		inline IteratorType end() const { return m_end; }

		template<class Q=ConstIteratorType>
		inline typename std::enable_if<!std::is_same<Q,IteratorType>::value,ConstIteratorType>::type begin() const { return m_begin; }
		template<class Q=ConstIteratorType>
		inline typename std::enable_if<!std::is_same<Q,IteratorType>::value,ConstIteratorType>::type end() const { return m_end; }
		
		inline const T&	operator[](size_t ix) const noexcept { return begin()[ix]; }
		template<typename Q=T>
        inline typename std::enable_if<!std::is_const_v<Q>,T&>::type operator[](size_t ix) noexcept { return begin()[ix]; }
		
		inline size_t size() const {return std::distance(m_begin,m_end);}

		inline bool empty() const { return m_begin==m_end; }

	private:
		IteratorType m_begin, m_end;
};

/*
template< class U, class T, typename IteratorType, typename ConstIteratorType>
inline SRange<U> SRange_static_cast(const SRange<T>& smart_ptr)
{
	SRange<U> other(nullptr,nullptr);
	return other;
}
*/

} // end namespace nbl::core

#endif

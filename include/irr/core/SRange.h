// Copyright (C) 2019 DevSH Graphics Programming Sp. z O.O.
// This file is part of the "IrrlichtBaW".
// For conditions of distribution and use, see LICENSE.md

#ifndef __IRR_S_RANGE_H_INCLUDED__
#define __IRR_S_RANGE_H_INCLUDED__

#include "stddef.h"
#include "irr/core/Types.h"

/*! \file SRange.h
	\brief File containing SRange utility struct for C++11 range loops
*/

namespace irr
{
namespace core
{

	template<typename T>
	struct SRange
	{
			inline SRange(T* _beg, T* _end) : m_begin(_beg), m_end(_end) {}

			inline T* begin() { return m_begin; }
			inline T* end() { return m_end; }

			inline size_t size() const {return m_end-m_begin;}

		private:
			T* m_begin, * m_end;
	};


} // end namespace core
} // end namespace irr

#endif

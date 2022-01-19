// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_REGION_BLOCK_FUNCTOR_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_REGION_BLOCK_FUNCTOR_H_INCLUDED__

#include "nbl/core/declarations.h"

#include <type_traits>

#include "nbl/asset/filters/CBasicImageFilterCommon.h"

namespace nbl
{
namespace asset
{

// fill a section of the image with a uniform value
template<typename Functor, bool ConstImage>
class CRegionBlockFunctorFilter : public CImageFilter<CRegionBlockFunctorFilter<Functor,ConstImage> >
{
	public:
		virtual ~CRegionBlockFunctorFilter() {}

		class CState : public IImageFilter::IState
		{
			public:
				using image_type = typename std::conditional<ConstImage, const ICPUImage, ICPUImage>::type;
				CState(Functor& _functor, image_type* _image, const IImage::SBufferCopy* _regionIterator) :
					functor(_functor), image(_image), regionIterator(_regionIterator) {}
				virtual ~CState() {}

				Functor& functor;
				image_type* image;
				const IImage::SBufferCopy* regionIterator;
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!state->image)
				return false;

			if (!state->regionIterator)
				return false;	
			const auto& regions = state->image->getRegions();
			if (state->regionIterator<regions.begin() || state->regionIterator>=regions.end())
				return false;

			return true;
		}

		template<class ExecutionPolicy>
		static inline bool execute(ExecutionPolicy&& policy, state_type* state)
		{
			if (!validate(state))
				return false;

			CBasicImageFilterCommon::executePerBlock<ExecutionPolicy,Functor>(std::forward<ExecutionPolicy>(policy),state->image,*state->regionIterator,state->functor);

			return true;
		}
		static inline bool execute(state_type* state)
		{
			return execute(std::execution::seq,state);
		}
};

} // end namespace asset
} // end namespace nbl

#endif
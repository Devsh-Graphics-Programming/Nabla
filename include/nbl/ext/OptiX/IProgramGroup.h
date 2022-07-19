// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_EXT_OPTIX_PROGRAM_GROUP_H_INCLUDED__
#define __NBL_EXT_OPTIX_PROGRAM_GROUP_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "optix.h"

namespace nbl
{
namespace ext
{
namespace OptiX
{

class IContext;


class NBL_API IProgramGroup final : public core::IReferenceCounted
{
	public:
		inline OptixProgramGroup getOptiXHandle() {return programGroup;}

	protected:
		friend class OptiX::IContext;

		IProgramGroup(const OptixProgramGroup& _programGroup) : programGroup(_programGroup) {}
		~IProgramGroup()
		{
			if (programGroup)
				optixProgramGroupDestroy(programGroup);
		}

		OptixProgramGroup programGroup;
};


}
}
}

#endif
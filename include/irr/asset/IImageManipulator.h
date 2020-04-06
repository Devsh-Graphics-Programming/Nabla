// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_IMAGE_MANIPULATOR_H_INCLUDED__
#define __I_IMAGE_MANIPULATOR_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/filters/CCopyImageFilter.h"

namespace irr
{
namespace asset
{

// scaled copies with filters
class CBlitImageFilter;

// specialized case of CBlitImageFilter
class CMipMapGenerationImageFilter;

} // end namespace asset
} // end namespace irr

#endif

/*

CGraphicsPipelineLoaderMTL.cpp	964	         const size_t alignment = 1u<<core::findLSB(images[CMTLPipelineMetadata::EMP_REFL_POSX]->getRegions().begin()->bufferRowLength); 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CGraphicsPipelineLoaderMTL.cpp	975	                 assert(images[i]->getRegions().size()==1ull); 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CGraphicsPipelineLoaderMTL.cpp	977	                 regions_.push_back(images[i]->getRegions().begin()[0]); 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset
CGraphicsPipelineLoaderMTL.cpp	994	                 const void* src = reinterpret_cast<const uint8_t*>(images[i]->getBuffer()->getPointer()) + images[i]->getRegions().begin()[0].bufferOffset; 	C:\work\IrrlichtBaw\IrrlichtBAW parallel work\src\irr\asset

*/

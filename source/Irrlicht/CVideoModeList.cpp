// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CVideoModeList.h"
#include "irr/core/math/irrMath.h"

#include <algorithm>

namespace irr
{
namespace video
{

//! constructor
CVideoModeList::CVideoModeList()
{
	#ifdef _IRR_DEBUG
	setDebugName("CVideoModeList");
	#endif

	Desktop.depth = 0;
	Desktop.size = core::dimension2d<uint32_t>(0,0);
}


void CVideoModeList::setDesktop(int32_t desktopDepth, const core::dimension2d<uint32_t>& desktopSize)
{
	Desktop.depth = desktopDepth;
	Desktop.size = desktopSize;
}


//! Gets amount of video modes in the list.
int32_t CVideoModeList::getVideoModeCount() const
{
	return (int32_t)VideoModes.size();
}


//! Returns the screen size of a video mode in pixels.
core::dimension2d<uint32_t> CVideoModeList::getVideoModeResolution(int32_t modeNumber) const
{
	if (modeNumber < 0 || modeNumber > (int32_t)VideoModes.size())
		return core::dimension2d<uint32_t>(0,0);

	return VideoModes[modeNumber].size;
}


core::dimension2d<uint32_t> CVideoModeList::getVideoModeResolution(
		const core::dimension2d<uint32_t>& minSize,
		const core::dimension2d<uint32_t>& maxSize) const
{
	uint32_t best=VideoModes.size();
	// if only one or no mode
	if (best<2)
		return getVideoModeResolution(0);

	uint32_t i;
	for (i=0; i<VideoModes.size(); ++i)
	{
		if (VideoModes[i].size.Width>=minSize.Width &&
			VideoModes[i].size.Height>=minSize.Height &&
			VideoModes[i].size.Width<=maxSize.Width &&
			VideoModes[i].size.Height<=maxSize.Height)
			best=i;
	}
	// we take the last one found, the largest one fitting
	if (best<VideoModes.size())
		return VideoModes[best].size;
	const uint32_t minArea = minSize.getArea();
	const uint32_t maxArea = maxSize.getArea();
	uint32_t minDist = 0xffffffff;
	best=0;
	for (i=0; i<VideoModes.size(); ++i)
	{
		const uint32_t area = VideoModes[i].size.getArea();
		const uint32_t dist = core::min_(abs(int(minArea-area)), abs(int(maxArea-area)));
		if (dist<minDist)
		{
			minDist=dist;
			best=i;
		}
	}
	return VideoModes[best].size;
}


//! Returns the pixel depth of a video mode in bits.
int32_t CVideoModeList::getVideoModeDepth(int32_t modeNumber) const
{
	if (modeNumber < 0 || modeNumber > (int32_t)VideoModes.size())
		return 0;

	return VideoModes[modeNumber].depth;
}


//! Returns current desktop screen resolution.
const core::dimension2d<uint32_t>& CVideoModeList::getDesktopResolution() const
{
	return Desktop.size;
}


//! Returns the pixel depth of a video mode in bits.
int32_t CVideoModeList::getDesktopDepth() const
{
	return Desktop.depth;
}


//! adds a new mode to the list
void CVideoModeList::addMode(const core::dimension2d<uint32_t>& size, int32_t depth)
{
	SVideoMode m;
	m.depth = depth;
	m.size = size;

	for (uint32_t i=0; i<VideoModes.size(); ++i)
	{
		if (VideoModes[i] == m)
			return;
	}

	VideoModes.push_back(m);
	std::sort(VideoModes.begin(),VideoModes.end()); // TODO: could be replaced by inserting into right place
}


} // end namespace video
} // end namespace irr


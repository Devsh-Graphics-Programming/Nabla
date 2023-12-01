// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_UTILITIES_C_DUMB_PRESENTATION_ORACLE_H_INCLUDED_
#define _NBL_VIDEO_UTILITIES_C_DUMB_PRESENTATION_ORACLE_H_INCLUDED_

#include "nbl/video/utilities/IPresentationOracle.h"

namespace nbl::video
{

#if 0 // TODO: port
class CDumbPresentationOracle : public IPresentationOracle
{
	public:
		CDumbPresentationOracle()
		{
			reset();
		}

		~CDumbPresentationOracle() {
		}

		inline void reportBeginFrameRecord() override
		{
			lastTime = std::chrono::steady_clock::now();
		}

		inline void reportEndFrameRecord()
		{
			auto renderStart = std::chrono::steady_clock::now();
			dt = std::chrono::duration_cast<std::chrono::microseconds>(renderStart-lastTime).count();
			
			// Calculate Simple Moving Average for FrameTime
			{
				timeSum -= dtList[frameCount];
				timeSum += dt;
				dtList[frameCount] = dt;
				frameCount++;
				if(frameCount >= MaxFramesToAverage) {
					frameCount = 0;
					frameDataFilled = true;
				}
			}
			double averageFrameTime = (frameDataFilled) ? (timeSum / (double)MaxFramesToAverage) : (timeSum / frameCount);
			auto averageFrameTimeDuration = std::chrono::duration<double, std::micro>(averageFrameTime);
			auto nextPresentationTime = renderStart + averageFrameTimeDuration;
			nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());
		}

		inline std::chrono::microseconds getNextPresentationTimeStamp() const {return nextPresentationTimeStamp;}

		inline double getDeltaTimeInMicroSeconds() const {return dt;}

		inline std::chrono::microseconds acquireNextImage(ISwapchain* swapchain, IGPUSemaphore* acquireSemaphore, IGPUFence* fence, uint32_t* imageNumber) override
		{
			swapchain->acquireNextImage(acquireSemaphore,fence,imageNumber);
			reportEndFrameRecord();
			const auto retval = getNextPresentationTimeStamp();
			reportBeginFrameRecord();
			return retval;
		}

		inline void present(ILogicalDevice* device, ISwapchain* swapchain, IQueue* queue, IGPUSemaphore* renderFinishedSemaphore, const uint32_t imageNumber) override
		{
			// literally cant do anything here in this dumb algorithm
		}
	private:
	
		void reset()
		{
			frameCount = 0ull;
			frameDataFilled = false;
			timeSum = 0.0;
			for(size_t i = 0ull; i < MaxFramesToAverage; ++i) {
				dtList[i] = 0.0;
			}
			dt = 0.0;
		}

		static constexpr size_t MaxFramesToAverage = 100ull;

		bool frameDataFilled = false;
		size_t frameCount = 0ull;
		double timeSum = 0.0;
		double dtList[MaxFramesToAverage] = {};
		double dt = 0.0;
		std::chrono::steady_clock::time_point lastTime;
		std::chrono::microseconds nextPresentationTimeStamp;
};
#endif

}

#endif // _NBL_VIDEO_I_PRESENTATION_ORACLE__H_INCLUDED_
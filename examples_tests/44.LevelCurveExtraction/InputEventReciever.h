#pragma once
#include <iostream>
#include <cstdio>
#include <irrlicht.h>
class ChgSpacingEventReciever : public irr::IEventReceiver
{
public:
	ChgSpacingEventReciever() : spacing(10), running(true)
	{
	}

	bool OnEvent(const irr::SEvent& event)
	{
		if (event.EventType == irr::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
		{
			switch (event.KeyInput.Key)
			{
			case irr::KEY_PRIOR: 
				spacing =std::min(spacing+2,100);
				std::cout << spacing << std::endl;
				return true;
			case irr::KEY_NEXT:
				spacing = std::max(spacing - 2, 2);
				std::cout << spacing << std::endl;
				return true;
			case irr::KEY_KEY_Q:
				running = false;
				return true;
			case irr::KEY_KEY_S:
				saveBuffer = true;
				return true;
			default:
				break;
			}
		}

		return false;
	}
	inline bool keepOpen() const { return running; }
	inline int getSpacing() const { return spacing; }
	inline int doBufferSave() 
	{ 
		if (saveBuffer)
		{
			saveBuffer = false;
			return true;
		}
		return false;
	}

private:
	int spacing;
	bool running;
	bool saveBuffer;
};
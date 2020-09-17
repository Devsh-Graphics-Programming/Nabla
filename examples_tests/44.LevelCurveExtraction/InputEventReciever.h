#ifndef _INPUT_EVENT_RECEIVER_H_
#define _INPUT_EVENT_RECEIVER_H_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

class ChgSpacingEventReciever : public irr::IEventReceiver
{
	public:
		ChgSpacingEventReciever() : spacing(10), running(true), speed(1), resetCam(false) {
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
				case irr::KEY_KEY_R:
					resetCam = true;
					return true;
				case irr::KEY_MINUS:
					speed = std::clamp<float>(speed - 0.1f, 0.2f, 5.0f);
					std::cout << "Camera speed:  " << speed << std::endl;
					return true;
				case irr::KEY_PLUS:
					speed = std::clamp<float>(speed + 0.1f, 0.2f, 5.0f);
					std::cout << "Camera speed:  " << speed << std::endl;
					return true;
				default:
					break;
				}
			}
			else if (event.EventType == irr::EET_MOUSE_INPUT_EVENT && event.MouseInput.Wheel != 0.0f)
			{
				speed = std::clamp<float>(speed + event.MouseInput.Wheel / 10.0f, 0.2f, 5.0f);
				std::cout << "Camera speed:  " << speed << std::endl;

				return true;

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
		inline int resetCameraPosition()
		{
			if (resetCam)
			{
				resetCam = false;
				return true;
			}
			return false;
		}
		inline float getCameraSpeed() const { return speed; }
	private:
		float speed;
		int spacing;
		bool running;
		bool saveBuffer;
		bool resetCam;
};
#endif
#include "WaveSimApp.h"

int main()
{
	WaveSimParams params{
		512,            // witdh
		512,            // length
		{ 1000, 1000 }, // patch size
		{ 1.f, 1.f },   // wind direction
		31,             // wind speed 
		0.05,           // amplitude
		0.07,           // wind dependency
		1.1             // choppiness 
	};
	WaveSimApp app(params);
	app.Run();
}
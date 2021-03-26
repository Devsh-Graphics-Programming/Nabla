#include "WaveSimApp.h"

int main()
{
	WaveSimParams params{
		512, 
		512,
		{ 1000, 1000 },
		{ 1.f, 1.f },
		31,
		1, 
		0.07
	};
	WaveSimApp app(params);
	app.Run();
}
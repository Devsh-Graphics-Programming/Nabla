#include "WaveSimApp.h"

int main()
{
	WaveSimParams params{
		256, 
		256,
		{ 200.f, 200.f },
		{ 1.f, 1.f },
		7,
		8, 
		0
	};
	WaveSimApp app(params);
	app.Run();
}
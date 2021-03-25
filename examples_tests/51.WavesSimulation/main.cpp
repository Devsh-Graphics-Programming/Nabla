#include "WaveSimApp.h"

int main()
{
	WaveSimParams params{
		256, 
		256,
		{ 170, 170 },
		{ 1.f, 0.f },
		5,
		5, 
		0.
	};
	//Sleep(5000);
	WaveSimApp app(params);
	app.Run();
}
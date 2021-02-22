#include "WaveSimApp.h"

int main()
{
	WaveSimParams params{
		512, 
		512,
		{ 1.f, 1.f },
		{ 1.f, 1.f },
		5,
		1000
	};
	Sleep(5000);
	WaveSimApp app(params);
	app.Run();
}
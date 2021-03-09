#include "WaveSimApp.h"

int main()
{
	WaveSimParams params{
		128, 
		128,
		{ 1.f, 1.f },
		{ 1.f, 1.f },
		6,
		500
	};
	//Sleep(5000);
	WaveSimApp app(params);
	app.Run();
}
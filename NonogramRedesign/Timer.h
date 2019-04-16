#pragma once
#include <map>
#include <string>
#include <ctime>


struct timeData {
	int count = 0;
	double sum = 0, min = 0, max = 0;
	clock_t curr = 0;
	bool isRunning = false;
};

class Timer
{
public:
	static const timeData& start(const std::string& name);
	static const timeData& stop(const std::string& name);
	static void abort(const std::string& name);
	static void print(std::string name);
	static void printAll();
	static void print(std::pair<std::string, timeData> data);

private:
	// map[timerName] = {count, min time, avg time, max time}
	static std::map<std::string, timeData> times;
	Timer();
	~Timer();

};


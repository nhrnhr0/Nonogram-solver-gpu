#pragma once
#include <iostream>
#include <ostream>
#include <string>

enum typelog {
	DEBUG,
	INFO,
	WARN,
	ERR
};

class LOG {
public:
	LOG() : LOG(INFO) {}
	LOG(typelog type) : label(getLabel(type)) { stream << label; }

	template<class T>
	LOG& operator<<(const T& msg) {
		stream << msg;
		return *this;
	}


private:
	static std::ostream& stream;
	bool isOpen;
	std::string label;
	static inline std::string getLabel(typelog type) {
		switch (type) {
		case DEBUG: return "[D] ";
		case INFO:  return "[I] ";
		case WARN:  return "[W] ";
		case ERR:	return "[E] ";
		}
		return "[?] ";
	}
};
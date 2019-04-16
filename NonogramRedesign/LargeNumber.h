#pragma once
#define MAX_NUMBER_DIGITS 100

class LargeNumber
{
public:
	LargeNumber(int factorial);
	void Multiply(int n);
	void Multiply(const LargeNumber& other);
	int Divide(const LargeNumber& other)const;
	void Add(const LargeNumber& other);
	void MoveRight(int n);
	LargeNumber(const LargeNumber& other);
	bool isBigger(const LargeNumber& other);
	bool isSmaller(const LargeNumber& other);
	bool isEqual(const LargeNumber& other);
	static LargeNumber Factorial(int factorial);
	static LargeNumber Multiply(const LargeNumber& n1, const LargeNumber& n2, int index = 0);
	~LargeNumber();

private:
	int num[MAX_NUMBER_DIGITS];
	int numSize;
};


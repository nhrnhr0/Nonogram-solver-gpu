#include "LargeNumber.h"
#include <iostream>
#include "Log.h"
#include <assert.h>

LargeNumber::LargeNumber(int n) {
	memset(num, 0, sizeof(int)*MAX_NUMBER_DIGITS);
	int i = 0;
	numSize = 0;
	while (n) {
		num[i++] = n % 10;
		n /= 10;
		numSize++;
	}
}

LargeNumber LargeNumber::Factorial(int factorial) {
	LargeNumber ret(0);
	ret.num[0] = 1;
	ret.numSize = 1;
	for (int i = 2; i <= factorial; i++) {
		ret.Multiply(i);
	}
	return ret;
}

void LargeNumber::Multiply(const LargeNumber& other) {

	LargeNumber res = Multiply(*this, other);
	// copy the result to the object:
	memcpy(num, res.num, sizeof(int)*res.numSize);
	numSize = res.numSize;
}

// note: this function dose not change the number itself, only reutrns the answer. ignore carry
// TODO: very bad function, need to improve
// rounding up
int LargeNumber::Divide(const LargeNumber& other) const {
	LargeNumber t(other);
	int count = 0;
	while (t.isSmaller(*this)) {
		t.Add(other);
		count++;
	}
	return count+1;
}

LargeNumber LargeNumber::Multiply(const LargeNumber& n1, const LargeNumber& n2, int index) {
	LargeNumber t(n1);
	if (index < n2.numSize) {
		t.Multiply(n2.num[index]);
		t.MoveRight(index);
		t.Add(Multiply(n1, n2, index + 1));
		return t;
	}
	return LargeNumber(0);
}

void LargeNumber::MoveRight(int n) {
	assert(n + numSize < MAX_NUMBER_DIGITS);
	for (int i = numSize-1; i >= 0; i--) {
		num[i + n] = num[i];
	}
	for (int i = n-1; i >= 0; i--) {
		num[i] = 0;
	}
	numSize += n;
}

void LargeNumber::Add(const LargeNumber & other)
{
	int i = 0;
	int carry = 0;
	int newSize = 0;
	// as long as we didn't got to the end of the 2 numbers.
	// we need to add every this[i] + other[i] and save the carry
	while (this->numSize > i || other.numSize > i) {
		carry = this->num[i] + other.num[i] + carry;
		this->num[i] = carry % 10;
		carry /= 10;
		i++;
	}
	// if we left with a carry, we need to add it to the result
	while (carry) {
		this->num[i] = carry % 10;
		carry /= 10;
		i++;
	}
	numSize = i;
}

LargeNumber::LargeNumber(const LargeNumber & other)
{
	//LOG(DEBUG) << "LargeNumber copy constractor called\n";
	this->numSize = other.numSize;
	for (int i = 0; i < MAX_NUMBER_DIGITS; i++) {
		this->num[i] = other.num[i];
	}
}

bool LargeNumber::isSmaller(const LargeNumber & other)
{
	if (other.numSize > this->numSize)
		return true;
	else if (other.numSize < this->numSize)
		return false;
	// equal number size:
	for (int i = this->numSize-1; i >= 0 ; i--) {
		if (other.num[i] > this->num[i])
			return true;
		else if (other.num[i] < this->num[i])
			return false;
	}
	// they are equal
	return false;
}

bool LargeNumber::isBigger(const LargeNumber& other) {
	if (other.numSize < this->numSize)
		return true;
	else if (other.numSize > this->numSize)
		return false;
	// equal number size:
	for (int i = this->numSize - 1; i >= 0; i--) {
		if (other.num[i] < this->num[i])
			return true;
		else if (other.num[i] > this->num[i])
			return false;
	}
	// they are equal
	return false;
}

bool LargeNumber::isEqual(const LargeNumber& other) {
	if (other.numSize != this->numSize)
		return false;
	for (int i = 0; i < this->numSize; i++) {
		if (this->num[i] != other.num[i])
			return false;
	}
	return true;
}

void LargeNumber::Multiply(int n)
{
	int carry = 0;

	// One by one multiply n with individual digits of num[] 
	for (int i = 0; i < numSize; i++) {
		int prod = num[i] * n + carry;
		// Store last digit of 'prod' in num[]   
		num[i] = prod % 10;
		// Put rest in carry 
		carry = prod / 10;
	}

	// Put carry in num and increase result size 
	while (carry) {
		num[numSize] = carry % 10;
		carry = carry / 10;
		numSize++;
	}
}

LargeNumber::~LargeNumber()
{
}

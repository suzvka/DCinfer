#include "Type.h"
#include <limits>

namespace DC {
	Int64::Int64(int64_t input){
		_isScalar = true;
		_data.clear();
		_data.push_back(input);
	}
	Int64::Int64(uint64_t input){
		_isScalar = true;
		_data.clear();
		input > LLONG_MAX ? _data.push_back(LLONG_MAX) : _data.push_back(input);
	}
	Int64::Int64(const std::vector<int64_t>& input){
		_isScalar = false;
		_data = input;
	}
	Int64::Int64(const std::vector<uint64_t>& input) {
		_isScalar = false;
		_data.clear();
		_data.reserve(input.size());
		for (auto it : input) {
			it > LLONG_MAX ? _data.push_back(LLONG_MAX) : _data.push_back(it);
		}
	}
	Int64::operator Int32() const{
		if (_isScalar) {
			Int32 tempValue;
			_data[0] > INT_MAX ? tempValue = INT_MAX : tempValue = static_cast<int32_t>(_data[0]);
			return tempValue;
		}
		else {
			std::vector<int32_t> tempValue;
			for (auto it : _data) {
				it > INT_MAX ? tempValue.push_back(INT_MAX) : tempValue.push_back(it);
			}
			Int32 backValue = tempValue;
			return backValue;
		}
	}
	Int64::operator Float() const{
		if (_isScalar) {
			Float tempValue;
			_data[0] > std::numeric_limits<float>::max() ? tempValue = std::numeric_limits<float>::max() : tempValue = static_cast<float>(_data[0]);
			return tempValue;
		}
		else {
			std::vector<float> tempValue;
			for (auto it : _data) {
				it > std::numeric_limits<float>::max() ? tempValue.push_back(std::numeric_limits<float>::max()) : tempValue.push_back(it);
			}
			Float backValue = tempValue;
			return backValue;
		}
	}
	Int64::operator Bool() const {
		if (_isScalar) {
			Bool tempValue;
			_data[0] != 0 ? tempValue = static_cast <uint8_t>(1) : tempValue = static_cast <uint8_t>(0);
			return tempValue;
		}
		else {
			std::vector<uint8_t> tempValue;
			for (auto it : _data) {
				it != 0 ? tempValue.push_back(1) : tempValue.push_back(0);
			}
			Bool backValue = tempValue;
			return backValue;
		}
	}
	const bool Int64::isScalar() const{
		return _isScalar;
	}
	const std::vector<int64_t> Int64::data() const{
		return _data;
	}
}

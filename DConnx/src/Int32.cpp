#include "Type.h"
#include <limits>

namespace DC {
	Int32::Int32(int32_t input) {
		_isScalar = true;
		_data.clear();
		_data.push_back(input);
	}
	Int32::Int32(uint32_t input) {
		_isScalar = true;
		_data.clear();
		input > LONG_MAX ? _data.push_back(LONG_MAX) : _data.push_back(input);
	}
	Int32::Int32(const std::vector<int32_t>& input) {
		_isScalar = false;
		_data = input;
	}
	Int32::Int32(const std::vector<uint32_t>& input) {
		_isScalar = false;
		_data.clear();
		_data.reserve(input.size());
		for (auto it : input) {
			it > LONG_MAX ? _data.push_back(LONG_MAX) : _data.push_back(it);
		}
	}
	Int32::operator Int64() const {
		if (_isScalar) {
			Int64 tempValue = static_cast<int64_t>(_data[0]);
			return tempValue;
		}
		else {
			std::vector<int64_t> tempValue;
			for (auto it : _data) {
				it > std::numeric_limits<int64_t>::max() ? tempValue.push_back(std::numeric_limits<int64_t>::max()) : tempValue.push_back(it);
			}
			Int64 backValue = tempValue;
			return backValue;
		}
	}
	Int32::operator Float() const {
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
	Int32::operator Bool() const {
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
	const bool Int32::isScalar() const {
		return _isScalar;
	}
	const std::vector<int32_t> Int32::data() const {
		return _data;
	}
}
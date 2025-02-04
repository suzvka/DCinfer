#include "Type.h"
#include <limits>

namespace DC {
	Bool::Bool(uint8_t input) {
		_isScalar = true;
		_data.clear();
		_data.push_back(1);
	}
	Bool::Bool(bool input) {
		_isScalar = true;
		_data.clear();
		input != 0 ? _data.push_back(1) : _data.push_back(0);
	}
	Bool::Bool(const std::vector<uint8_t>& input) {
		_isScalar = false;
		_data.clear();
		_data.reserve(input.size());
		for (auto it : input) {
			it != 0 ? _data.push_back(1) : _data.push_back(0);
		}
	}
	Bool::Bool(const std::vector<bool>& input) {
		_isScalar = false;
		_data.clear();
		_data.reserve(input.size());
		for (auto it : input) {
			it? _data.push_back(1) : _data.push_back(0);
		}
	}
	Bool::operator Int64() const {
		if (_isScalar) {
			Int64 tempValue = static_cast<int64_t>(_data[0]);
			return tempValue;
		}
		else {
			std::vector<int64_t> tempValue;
			for (auto it : _data) {
				it != 0 ? tempValue.push_back(1) : tempValue.push_back(0);
			}
			Int64 backValue = tempValue;
			return backValue;
		}
	}
	Bool::operator Int32() const {
		if (_isScalar) {
			Int32 tempValue = static_cast<int32_t>(_data[0]);
			return tempValue;
		}
		else {
			std::vector<int32_t> tempValue;
			for (auto it : _data) {
				it != 0 ? tempValue.push_back(1) : tempValue.push_back(0);
			}
			Int32 backValue = tempValue;
			return backValue;
		}
	}
	Bool::operator Float() const {
		if (_isScalar) {
			Float tempValue = static_cast<float>(_data[0]);
			return tempValue;
		}
		else {
			std::vector<float> tempValue;
			for (auto it : _data) {
				it != 0 ? tempValue.push_back(1.0f) : tempValue.push_back(0.0f);
			}
			Float backValue = tempValue;
			return backValue;
		}
	}
	const bool Bool::isScalar() const {
		return _isScalar;
	}
	const std::vector<uint8_t> Bool::data() const {
		return _data;
	}
}
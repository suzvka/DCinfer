#include "Type.h"
#include <limits>

namespace DC {
	Float::Float(float input){
		_isScalar = true;
		_data.clear();
		_data.push_back(input);
	}
	Float::Float(const std::vector<float>& input) {
		_isScalar = false;
		_data.clear();
		_data.reserve(input.size());
		for (auto it : input) {
			it > std::numeric_limits<float>::max() ? _data.push_back(std::numeric_limits<float>::max()) : _data.push_back(it);
		}
	}
	Float::operator Int64() const {
		if (_isScalar) {
			Int64 tempValue;
			_data[0] != 0.0f ? tempValue = static_cast <int64_t>(_data[0]) : tempValue = static_cast <int64_t>(_data[0]);
			return tempValue;
		}
		else {
			std::vector<int64_t> tempValue;
			for (auto it : _data) {
				it > std::numeric_limits<float>::max() ? tempValue.push_back(std::numeric_limits<int64_t>::max()) : tempValue.push_back(it);
			}
			Int64 backValue = tempValue;
			return backValue;
		}
	}
	Float::operator Int32() const {
		if (_isScalar) {
			Int32 tempValue;
			_data[0] != 0.0f ? tempValue = static_cast <int32_t>(_data[0]) : tempValue = static_cast <int32_t>(_data[0]);
			return tempValue;
		}
		else {
			std::vector<int32_t> tempValue;
			for (auto it : _data) {
				it > std::numeric_limits<float>::max() ? tempValue.push_back(std::numeric_limits<int32_t>::max()) : tempValue.push_back(it);
			}
			Int32 backValue = tempValue;
			return backValue;
		}
	}
	Float::operator Bool() const {
		if (_isScalar) {
			Bool tempValue;
			_data[0] != 0.0f ? tempValue = static_cast <uint8_t>(1) : tempValue = static_cast <uint8_t>(0);
			return tempValue;
		}
		else {
			std::vector<uint8_t> tempValue;
			for (auto it : _data) {
				it != 0.0f ? tempValue.push_back(1) : tempValue.push_back(0);
			}
			Bool backValue = tempValue;
			return backValue;
		}
	}
	const bool Float::isScalar() const {
		return _isScalar;
	}
	const std::vector<float> Float::data() const {
		return _data;
	}
}
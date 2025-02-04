#pragma once
#include <vector>

namespace DC {
	class Int64;
	class Int32;
	class Float;
	class Bool;
	template<typename T>
	class TypeBase {
	public:
		// 获取实际的类型信息
		const type_info& info()const {
			return typeid(T);
		}

		// 获取张量中单个元素的尺寸
		const size_t elementSize()const {
			return sizeof(T);
		}

		// 获取张量实际尺寸
		//virtual const size_t size()const = 0;

		//virtual const bool isScalar()const = 0;

		//template<typename K>
		//const std::vector<K> toType()const {
		//	std::vector<T> tempValue;
		//	for () {

		//	}
		//}
	};

	class Int64 :public TypeBase<int64_t> {
	public:
		Int64() {};
		Int64(int64_t input);
		Int64(uint64_t input);
		Int64(const std::vector<int64_t>& input);
		Int64(const std::vector<uint64_t>& input);

		operator Int32() const;
		operator Float() const;
		operator Bool() const;

		const bool isScalar()const;
		const std::vector<int64_t> data()const;

	private:
		bool _isScalar = true;
		std::vector<int64_t> _data;
	};

	class Int32 :public TypeBase<int32_t> {
	public:
		Int32() {}
		Int32(int32_t input);
		Int32(uint32_t input);
		Int32(const std::vector<uint32_t>& input);
		Int32(const std::vector<int32_t>& input);

		operator Int64() const;
		operator Float() const;
		operator Bool() const;

		const bool isScalar()const;
		const std::vector<int32_t> data()const;

	private:
		bool _isScalar = true;
		std::vector<int32_t> _data;
	};

	class Float :public TypeBase<float> {
	public:
		Float() {}
		Float(float input);
		Float(const std::vector<float>& input);

		operator Int64() const;
		operator Int32() const;
		operator Bool() const;

		const bool isScalar()const;
		const std::vector<float> data()const;

	private:
		bool _isScalar = true;
		std::vector<float> _data;
	};

	class Bool :public TypeBase<uint8_t> {
	public:
		Bool() {}
		Bool(uint8_t input);
		Bool(bool input);
		Bool(const std::vector<uint8_t>& input);
		Bool(const std::vector<bool>& input);

		operator Int64() const;
		operator Int32() const;
		operator Float() const;
		
		const bool isScalar()const;
		const std::vector<uint8_t> data()const;

	private:
		bool _isScalar = true;
		std::vector<uint8_t> _data;
	};
}
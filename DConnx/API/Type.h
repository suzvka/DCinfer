#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <memory>
#include <stdexcept>

namespace DC {
	class TypeInfo;
	template<typename T> class Type;

	template<typename TEnum>
	class TypeManager {
	public:
		// 注册一个新类型及其到后端枚举类型的映射
		template<typename T>
		void registerType(TEnum enumValue, const std::string& name) {
			const std::type_index typeIndex(typeid(T));
			auto typeName = std::make_unique<Type<T>>(name);
			auto* typePtr = typeName.get();

			_typeFromIndex[typeIndex] = std::move(typeName);
			_typeFromName[name] = typePtr;
			_toEnumMap[typePtr] = enumValue;
			_fromEnumMap[enumValue] = typePtr;
		}

		// 按 C++ 类型获取内部类型
		template<typename T>
		const TypeInfo& get() const {
			return get(std::type_index(typeid(T)));
		}

		// 按 std::type_index 获取内部类型
		const TypeInfo& get(const std::type_index& typeIndex) const {
			auto it = _typeFromIndex.find(typeIndex);
			if (it == _typeFromIndex.end()) {
				throw std::runtime_error("Unregistered type: " + std::string(typeIndex.name()));
			}
			return *it->second;
		}

		// 从后端枚举类型转换为内部类型
		const TypeInfo& fromEnum(TEnum enumValue) const {
			auto it = _fromEnumMap.find(enumValue);
			if (it == _fromEnumMap.end()) {
				throw std::runtime_error("Unsupported enum type for conversion");
			}
			return *it->second;
		}

		// 从内部类型转换为后端枚举类型
		TEnum toEnum(const TypeInfo* typeName) const {
			auto it = _toEnumMap.find(typeName);
			if (it == _toEnumMap.end()) {
				throw std::runtime_error("Unsupported internal type for enum conversion");
			}
			return it->second;
		}

	private:
		std::unordered_map<std::type_index, std::unique_ptr<TypeInfo>> _typeFromIndex;
		std::unordered_map<std::string, TypeInfo*> _typeFromName;
		std::unordered_map<TEnum, TypeInfo*> _fromEnumMap;
		std::unordered_map<const TypeInfo*, TEnum> _toEnumMap;
	};

	// 类型基类
	class TypeInfo {
	public:
		virtual ~TypeInfo() = default;
		virtual const std::string& getTypeName() const = 0;
		virtual size_t getTypeSize() const = 0;
		bool isEqual(const TypeInfo& other) const {
			return typeid(*this) == typeid(other);
		}
	};

	// 泛型类型实现，简化新类型定义
	template<typename T>
	class Type : public TypeInfo {
	public:
		Type(std::string name) : _name(std::move(name)) {}
		const std::string& getTypeName() const override { return _name; }
		size_t getTypeSize() const override { return sizeof(T); }
	private:
		std::string _name;
	};
}
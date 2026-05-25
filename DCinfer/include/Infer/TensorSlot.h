#pragma once
#include <type_traits>
#include <stdexcept>
#include <optional>
#include <functional>
#include <memory>

#include "Tensor.hpp"
#include "Exception.h"
#include "DCtype.h"
#include "SlotType.h"
#include "NativeTensor.h"
#include <iostream>

namespace DC {
	class TensorSlot {
		using TensorType = TensorMeta::TensorType;
		using ErrorType  = TensorException::ErrorType;

	public:
		using Shape = Tensor::Shape;

		// ── 配置（精简后）──
		class Config {
		public:
			enum class Position {
				Input,
				Output,
				Auto
			};

			Config& setPosition(Position p);

			Position position = Position::Auto;
		};

		TensorSlot(const TensorSlot&) = delete;
		TensorSlot& operator=(const TensorSlot&) = delete;
		TensorSlot(TensorSlot&&) noexcept = default;
		TensorSlot& operator=(TensorSlot&&) noexcept = default;

		TensorSlot(
			const std::string& name,
			TensorMeta::TensorType type,
			size_t size,
			const Shape& shape,
			const Config& config = Config()
		);

		// ── 元数据（仅与 DC::Tensor 有关）──
		TensorSlot& setDefaultTensor(const Tensor& data);

		const std::string& name()      const;
		TensorType          type()      const;
		size_t              typeSize()  const;
		Shape               shape()     const;
		Shape               dataShape() const;

		bool isInput()  const;
		bool isOutput() const;

		bool hasDefaultData() const;
		const Tensor& defaultTensor() const;

		template<typename T>
		bool isType() const;

		// ── 运行时存储：类型擦除 ──
		// store：通过 DC::Type 推导 SlotDataType，经 ValidatorRegistry 校验后存储
		template<typename T>
		TensorSlot& store(T&& data);

		// take：移动取出，运行时检查类型标签是否匹配
		template<typename T>
		T take();

		// peek：只读指针，类型不匹配返回 nullptr
		template<typename T>
		const T* peek() const;

		// 便捷方法：以 const Tensor& 获取 DC::Tensor 数据
		// 仅当 storedType() == SlotDataType::DCTensor 时有效
		const Tensor& view() const;

		bool              hasData()     const;
		SlotDataType      storedType()  const;
		const void*       rawPtr()      const;

		void clear();
		void clearData();

		const Config& config() const;
		static Config CreateConfig();

	private:
		// ── 类型擦除存储 ──
		struct TypedBlob {
			void*                      ptr = nullptr;
			std::function<void(void*)> deleter;
			SlotDataType               type = SlotDataType::Unknown;
		};

		TensorMeta              _rule;
		std::unique_ptr<Tensor> _defaultData;  // 默认数据（永远是 DC::Tensor）
		std::optional<TypedBlob> _blob;         // 运行时数据
		Config                  _config;

		[[noreturn]] void abort(
			ErrorType errorType = ErrorType::Other,
			const std::string& message = ""
		) const;
	};

	// Template method definitions
	template<typename T>
	bool TensorSlot::isType() const {
		return type() == Type::getType<TensorMeta::TensorType, T>();
	}

	template<typename T>
	TensorSlot& TensorSlot::store(T&& data) {
		ValidatorRegistry::ensureDefaults();  // 保证默认注册已执行（std::call_once）
		auto typeEnum = DC::Type::getType<SlotDataType, std::decay_t<T>>();

		// 校验：始终按存储的实际类型进行校验
		auto status = ValidatorRegistry::instance().validate(
			std::addressof(data), typeEnum, _rule);

		// Diagnostic log
		try {
			std::cerr << "TensorSlot::store name='" << _rule.name << "' type="
					  << static_cast<int>(typeEnum)
					  << " status.ready=" << status.ready()
					  << " invalid=" << status.invalid
					  << " needConvert=" << status.needConvert
					  << " needAlign=" << status.needAlign
					  << std::endl;
		} catch(...) {}

		if (!status.ready()) {
			if (status.invalid) {
				abort(ErrorType::InvalidShape, "Input data is invalid");
			}
			if (status.needConvert) {
				abort(ErrorType::TypeMismatch, "Type mismatch and conversion not allowed");
			}
			if (status.needAlign) {
				abort(ErrorType::ShapeMismatch, "Shape mismatch and alignment not allowed");
			}
		}

		// 释放旧数据
		if (_blob.has_value() && _blob->deleter && _blob->ptr) {
			_blob->deleter(_blob->ptr);
		}

		// 类型擦除存储
		TypedBlob blob;
		blob.type = typeEnum;
		blob.ptr  = new std::decay_t<T>(std::forward<T>(data));
		blob.deleter = [](void* p) { delete static_cast<std::decay_t<T>*>(p); };
		_blob = std::move(blob);

		return *this;
	}

	template<typename T>
	T TensorSlot::take() {
		if (!_blob.has_value() || !_blob->ptr) {
			abort(ErrorType::NotData, "Slot is empty");
		}

		auto expectedType = DC::Type::getType<SlotDataType, T>();
		if (_blob->type != expectedType) {
			abort(ErrorType::TypeMismatch,
				"take<T>: type mismatch, stored=" +
				std::to_string(static_cast<uint32_t>(_blob->type)) +
				" expected=" +
				std::to_string(static_cast<uint32_t>(expectedType)));
		}

		auto* typed = static_cast<T*>(_blob->ptr);
		T result = std::move(*typed);

		// 释放存储（但不调用 deleter，因为已移动）
		typed->~T();
		operator delete(typed);
		_blob.reset();

		return result;
	}

	template<typename T>
	const T* TensorSlot::peek() const {
		if (!_blob.has_value() || !_blob->ptr) {
			return nullptr;
		}

		auto expectedType = DC::Type::getType<SlotDataType, T>();
		if (_blob->type != expectedType) {
			return nullptr;
		}

		return static_cast<const T*>(_blob->ptr);
	}
}
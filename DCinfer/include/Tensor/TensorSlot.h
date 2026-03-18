#pragma once
#include <type_traits>
#include <stdexcept>
#include <optional>
#include <functional>

#include "Tensor.hpp"
#include "Exception.h"

namespace DC {
	class TensorSlotBase {
		using TensorType = TensorMeta::TensorType;
		using ErrorType = TensorException::ErrorType;
		using DataBlock = Tensor::DataBlock;

	public:
		using Shape = Tensor::Shape;
		class Config {
		public:
			enum class Type {
				Value,
				Data,
				Auto
			};

			enum class Position {
				Input,
				Output,
				Auto
			};

			enum class CheckLevel {
				Strict,
				Lenient
			};

			bool allowTypeConversion() const;

			bool requiredcheckType() const;

			Config& setType(Type t);
			Config& setPosition(Position p);
			Config& setCheckLevel(CheckLevel level);

			Type type = Type::Auto;
			Position position = Position::Auto;
			CheckLevel checkLevel = CheckLevel::Strict;
		};

		struct DataStatus {
			bool needAlign = false; // 需要形状对齐
			bool needConvert = false; // 需要类型转换
			bool invalid = false; // 数据无效

			bool ready() const;
		};

		TensorSlotBase(const TensorSlotBase&) = delete;
		TensorSlotBase& operator=(const TensorSlotBase&) = delete;
		TensorSlotBase(TensorSlotBase&&) noexcept = default;
		TensorSlotBase& operator=(TensorSlotBase&&) noexcept = default;

		TensorSlotBase(
			const std::string& name,
			TensorMeta::TensorType type,
			size_t size,
			const Shape& shape,
			const Config& config = Config()
		);

		TensorSlotBase& setDefaultTensor(const Tensor& data);

		const std::string& name() const;

		TensorType type() const;

		size_t typeSize() const;

		Shape shape() const;

		Shape dataShape() const;

		bool isInput() const;
		bool isOutput() const;

		template<typename T>
		bool isType() const;

		TensorSlotBase& write(Tensor&& data);
		TensorSlotBase& operator<<(Tensor&& data);
		TensorSlotBase& operator<<(const Tensor& data);
		
		TensorSlotBase& read(Tensor& data);
		TensorSlotBase& operator>>(Tensor& data);

		template<typename InferTensor>
		InferTensor convert(const std::function<InferTensor(const Tensor&)>& toExternal) {
			return toExternal(view());
		}
		
		bool hasData() const;

		bool hasDefaultData() const;

		bool hasDynamicData() const;

		void clear();

		void clearData();

		const Tensor& view() const;

		DataStatus check() const;
		DataStatus check(const Tensor& data) const;

		const Config& config() const;

		static Config CreateConfig();

		TensorSlotBase& loadData(Tensor&& data);
	private:
		TensorMeta _rule;
		std::unique_ptr<Tensor> _defaultData; // 默认数据
		std::unique_ptr<Tensor> _data;
		TensorSlotBase::Config _config;

		Tensor takeTensor();

		// 异常中止
		void abort(
			ErrorType errorType = ErrorType::Other,
			const std::string& message = ""
		) const;

		// Todo：移到上层
		Tensor align(
			const Shape& target,
			std::byte fillData = {}
		);
	};

	template<typename InferTensor>
	class TensorSlot : public TensorSlotBase {
	public:
		TensorSlot(
			const std::string& name, 
			TensorMeta::TensorType type, 
			size_t size, const Shape& shape, 
			const std::function<Tensor(const InferTensor&)>& toInternal,
			const std::function<InferTensor(const Tensor&)>& toExternal,
			const Config& config = Config()
		): TensorSlotBase(name, type, size, shape, config) {
			_toInternal = toInternal;
			_toExternal = toExternal;
			if (!toInternal || !toExternal) {
				throw std::invalid_argument("Conversion functions cannot be null");
			}
		}

		TensorSlot& operator<<(const InferTensor& data) {
			_externalRef = &data;
			_externalOwned.reset();
			return *this;
		}

		TensorSlot& operator<<(InferTensor&& data) {
			_externalOwned = std::make_unique<InferTensor>(std::move(data));
			_externalRef = _externalOwned.get();
			return *this;
		}

		using TensorSlotBase::operator<<;

		TensorSlot& read(InferTensor& data) {
			if (_externalRef) {
				data = *_externalRef;
				return *this;
			}

			if (!hasData()) {
				throw std::runtime_error("Slot is empty");
			}

			_externalOwned = std::make_unique<InferTensor>(_toExternal(view()));
			_externalRef = _externalOwned.get();

			data = std::move(*_externalRef);
			
			return *this;
		}

		TensorSlot& operator>>(InferTensor& data) {
			return read(data);
		}

	private:
		std::function<Tensor(const InferTensor&)> _toInternal; // 外部张量转内部张量
		std::function<InferTensor(const Tensor&)> _toExternal; // 内部张量转外部张量
		std::unique_ptr<InferTensor> _externalOwned;
		const InferTensor* _externalRef = nullptr;
	};

	template<typename T>
	TensorSlotBase CreateSlot(
		const std::string& name,
		const std::vector<int64_t>& shape,
		const TensorSlotBase::Config& config
	) {
		return TensorSlotBase(
			name,
			Type::getType<TensorMeta::TensorType>(T()),
			Type::getSize<TensorMeta::TensorType>(T()),
			shape,
			config
		);
	}

	// Template method definitions for TensorSlotBase
	template<typename T>
	bool TensorSlotBase::isType() const {
		return type() == Type::getType<TensorMeta::TensorType, T>();
	}
}
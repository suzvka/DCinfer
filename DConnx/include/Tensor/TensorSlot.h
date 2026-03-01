#pragma once
#include <type_traits>
#include <stdexcept>
#include <optional>

#include "Tensor.hpp"
#include "Exception.h"


class InferBase;

namespace DC {
	class TensorSlot {
		using TensorType = TensorMeta::TensorType;
		using ErrorType = TensorException::ErrorType;
		using Shape = Tensor::Shape;
		using DataBlock = Tensor::DataBlock;

	public:
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

			bool allowShapeAlignment() const;

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

		TensorSlot& setDefaultTensor(const Tensor& data);

		const std::string& name() const;

		TensorType type() const;

		size_t typeSize() const;

		Shape shape() const;

		Shape dataShape() const;

		bool isInput() const;
		bool isOutput() const;

		template<typename T>
		bool isType() const;

		TensorSlot& write(Tensor&& data);
		TensorSlot& operator<<(Tensor&& data);
		TensorSlot& operator<<(const Tensor& data);
		
		TensorSlot& read(Tensor& data);
		TensorSlot& operator>>(Tensor& data);
		
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

		TensorSlot& loadData(Tensor&& data);
	private:
		TensorMeta _rule;
		std::unique_ptr<Tensor> _defaultData; // 默认数据
		std::unique_ptr<Tensor> _data;
		TensorSlot::Config _config;

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

	template<typename T>
	TensorSlot CreateSlot(
		const std::string& name,
		const std::vector<int64_t>& shape,
		const TensorSlot::Config& config
	) {
		return TensorSlot(
			name,
			Type::getType<TensorMeta::TensorType>(T()),
			Type::getSize<TensorMeta::TensorType>(T()),
			shape,
			config
		);
	}

	// Template method definitions for TensorSlot
	template<typename T>
	bool TensorSlot::isType() const {
		return type() == Type::getType<TensorMeta::TensorType, T>();
	}
}
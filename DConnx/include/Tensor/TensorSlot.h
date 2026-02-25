#pragma once
#include <type_traits>
#include <stdexcept>

#include "Tensor.hpp"
#include "Exception.h"

class InferBase;

namespace DC {
	class TensorSlot {
		using TensorType = TensorMeta::TensorType;
		using ErrorType = TensorException::ErrorType;

	public:
		TensorSlot() = default;

		// 使用 DType 构造张量槽
		// - 张量名称
		// - 张量类型
		// - 张量类型大小(字节)
		// - 张量形状(默认空)
		TensorSlot(
			const std::string& name,
			TensorMeta::TensorType type,
			size_t size,
			const std::vector<int64_t>& shape = {}
		);

		TensorSlot& setDefaultTensor(Tensor& data);

		std::string name() const { return _rule.name; }

		TensorType type() const { return _rule.type; }

		size_t typeSize() const { return _rule.typeSize; }

		TensorSlot& setName(const std::string& name);

		TensorSlot& setShapes(const std::vector<int64_t>& shape);


		bool operator==(const Tensor& data) const {
			if (!_rule.check(data.shape())) {
				return false;
			}
			if (_rule.type != TensorType::Void && _rule.type != data.type()) {
				return false;
			}
			if (_rule.typeSize != 0 && _rule.typeSize != data.typeSize()) {
				return false;
			}
			return true;
		}

		const TensorSlot& input(Tensor& data) const {
			if (*this != data) abort(ErrorType::TypeMismatch, "input tensor does not match slot requirements");
			_data = std::make_unique<Tensor>(std::move(data));
			return *this;
		}

		const bool hasData() const {
			return _data != nullptr || _defaultData != nullptr;
		}

		void clear() const {
			_data.reset();
		}

		Tensor& getTensor();

	private:
		InferBase* _infer = nullptr; // 归属的推理器
		TensorMeta _rule;
		std::unique_ptr<Tensor> _defaultData; // 默认数据
		mutable std::unique_ptr<Tensor> _data;

		// 异常中止
		void abort(
			ErrorType errorType = ErrorType::Other,
			const std::string& message = ""
		) const {
			std::string source = "TensorSlot";
			if (!_rule.name.empty()) {
				source += " (" + _rule.name + ")";
			}
			throw TensorException(errorType, source, message);
		}
	};

	template<typename T>
	TensorSlot CreateSlot(
		const std::string& name,
		const std::vector<int64_t>& shape = {}
	) {
		return TensorSlot(
			name,
			Type::getType<TensorMeta::TensorType>(T()),
			Type::getSize<TensorMeta::TensorType>(T()),
			shape
		);
	}
}
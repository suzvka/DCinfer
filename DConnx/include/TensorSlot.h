#pragma once
#include "Tensor.h"
#include "tool.h"
#include <type_traits>

class InferBase;

namespace DC {
	class TensorSlot {
		using TensorType = TensorMeta::TensorType;
		using State = TensorMeta::DataState;
		using MismatchPolicy = TensorMeta::MismatchPolicy;

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

		TensorSlot& setDefaultTensor(const Tensor& data);

		std::string name() const { return _rule.name; }

		TensorType type() const { return _rule.type; }

		size_t typeSize() const { return _rule.typeSize; }

		TensorSlot& setName(const std::string& name);

		TensorSlot& setShapes(const std::vector<int64_t>& shape);


		bool operator==(const Tensor& data) const {
			return _rule.check(data.getShape());
		}

		const TensorSlot& input(Tensor& data) const {
			if (*this != data) throw std::runtime_error("规则检查未通过");
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
		TensorType _type = TensorType::Void;
		size_t _typeSize = 0; // 模型真实元素大小（按字节解释）
		std::unique_ptr<Tensor> _defaultData; // 默认数据
		mutable std::unique_ptr<Tensor> _data;
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
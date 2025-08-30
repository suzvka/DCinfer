#pragma once
#include "Tensor.h"
#include "tool.h"

class Infer;

namespace DC {
	class TensorSlot {
	public:
		TensorSlot() = default;

		TensorSlot(
			const std::string& name,				// - 张量名称
			const TensorMeta::TensorType& type,		// - 张量类型
			const std::string& typeName,			// - 张量类型名
			const std::vector<int64_t>& shape = {}	// - 张量形状
		) {
			setName(name);
			setShaps(shape);
			switch (type) {
				case TensorMeta::TensorType::Float: {
					setType<float>(typeName);
					break;
				}
				case TensorMeta::TensorType::Int: {
					setType<int64_t>(typeName);
					break;
				}
				case TensorMeta::TensorType::Uint: {
					setType<uint64_t>(typeName);
					break;
				}
				case TensorMeta::TensorType::Bool: {
					setType<bool>(typeName);
					break;
				}
				default: {
					throw std::runtime_error("Unsupported tensor type");
					break;
				}
			}
		}

		template<typename T>
		static TensorSlot Create(
			const std::string& name,
			const std::vector<int64_t>& shape
		) {
			TensorSlot slot;
			slot.setName(name);
			slot.setType<T>();
			slot.setShaps(shape);
			return slot;
		}

		std::string name() const { return _rule.name; }
		std::string type() const { return _rule.typeName; }
		size_t typeSize() const { return _rule.typeSize; }
		std::vector<int64_t> shape() const { return _rule.shape; }

		TensorSlot& setName(const std::string& name) {
			_rule.name = name;
			return *this;
		}

		TensorSlot& setShaps(const std::vector<int64_t>& shape) {
			_rule.shape = shape;
			return *this;
		}

		template<typename T> TensorSlot& setType() {
			_rule.typeName = typeid(T).name();
			_rule.typeSize = sizeof(T);
			return *this;
		}

		template<typename T> TensorSlot& setType(
			const std::string& typeName
		) {
			_rule.typeName = typeName;
			_rule.typeSize = sizeof(T);
			return *this;
		}

		bool operator==(const Tensor& data) const {
			return _rule.check(data.getShape());
		}

		const TensorSlot& input(Tensor& data) const {
			if (*this != data) throw std::runtime_error("规则检查未通过");
			_data = std::make_unique<Tensor>(std::move(data));
			return *this;
		}

		const bool hasData() const {
			return _data != nullptr;
		}

		void clear() const {
			_data.reset();
		}

		Tensor& getTensor() {
			if (!hasData()) throw std::runtime_error("无数据");
			return *_data;
		}

	private:
		Infer* _infer = nullptr; // 归属的推理器
		TensorMeta _rule;
		mutable std::unique_ptr<Tensor> _data;
	};
}
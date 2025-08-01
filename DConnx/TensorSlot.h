#pragma once
#include "Tensor.h"

class Infer;

namespace DC {
	class InferFeedingPort {
	public:
		template<typename T> InferFeedingPort(
			const std::string& name,            // - 张量名称
			const std::vector<int64_t>& shape   // - 张量形状
		) {
			setName(name);
			setType<T>();
			setShaps(shape);
		}

		std::string name() const { return _rule.name; }
		std::string type() const { return _rule.type; }
		size_t typeSize() const { return _rule.typeSize; }
		std::vector<int64_t> shape() const { return _rule.shape; }

		InferFeedingPort& setName(const std::string& name) {
			_rule.name = name;
			return *this;
		}

		InferFeedingPort& setShaps(const std::vector<int64_t>& shape) {
			_rule.shape = shape;
			return *this;
		}

		template<typename T> InferFeedingPort& setType() {
			_rule.type = typeid(T).name();
			_rule.typeSize = sizeof(T);
			return *this;
		}

		bool operator==(const Tensor& data) const {
			return _rule.check(data.getShape());
		}

		const InferFeedingPort& input(Tensor& data) const {
			if (*this != data) throw std::runtime_error("规则检查未通过");
			_data = std::make_unique<Tensor>(std::move(data));
			return *this;
		}

	private:
		Infer* _infer = nullptr; // 归属的推理器
		TensorMeta _rule;
		mutable std::unique_ptr<Tensor> _data;
	};
}
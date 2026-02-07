#pragma once
#include <stdexcept>

#include "TensorMods.h"

namespace DC {
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// DC::Tensor
// 张量对象
// 栈占用：240 字节
// 
// 用于创建输入或输出张量
// 在启动时设置张量规则，之后可以对实际数据进行检查，以匹配规则
// 在运行时而非编译时进行类型检查，请做好异常处理
// 自行指定类型时会按位进行转换，可能出现精度丢失，请小心使用
//
// 形状术语约定：
// - RuleShape  : 规则形状（TensorMeta::shape），用于 check()
// - CurrentShape: 当前数据形状（TensorData::getCurrentShape）
// - View::path : 索引路径，不等于 shape
//
// 维度规则：RuleShape 某一维为 -1 表示动态维度，check() 时跳过该维度。
// 
// 	2026.2.3
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	class Tensor {
		using TensorType = TensorMeta::TensorType;
		using State = TensorMeta::DataState;
		using MismatchPolicy = TensorMeta::MismatchPolicy;

	public:
		class View {
		public:
			View(Tensor& tensor, std::vector<int64_t> path = {})
				: _tensor(&tensor), _path(std::move(path)) {
			}

			View operator[](uint64_t index) const {
				auto nextPath = _path;
				nextPath.push_back(static_cast<int64_t>(index));
				return View(*_tensor, std::move(nextPath));
			}

			template<typename T>
			View& operator=(const std::vector<T>& data) {
				_tensor->writeAt(_path, data);
				return *this;
			}

			template<typename T>
			std::vector<T> read() const {
				return _tensor->readAt<T>(_path);
			}

			const std::vector<int64_t>& path() const {
				return _path;
			}

		private:
			Tensor* _tensor = nullptr;
			std::vector<int64_t> _path;
		};

		virtual ~Tensor() = default;
		Tensor() = default;

		Tensor(
			const std::string& name,				// - 张量名称
			const TensorType& type,					// - 张量类型
			const std::vector<int64_t>& shape = {},	// - 规则形状（RuleShape）
			std::vector<char>&& data = {}			// - 张量数据
		) {
			_rule.name = name;
			_rule.type = type;
			_rule.typeSize = Type::getSize<TensorType>(type);
			_rule.shape = shape;
			_data = TensorData(shape, std::move(data));
		}

		template<typename T>
		static Tensor Create(
			const std::string& name,
			const std::vector<int64_t>& shape = {},
			std::vector<char>&& data = {}
		) {
			Tensor tensor(
				name,
				Type::getType<TensorType>(T()),
				shape,
				std::move(data)
			);

			return tensor;
		}

		Tensor& setShape(const std::vector<int64_t>& shape) {
			_rule.shape = shape;
			return *this;
		}

		View operator[](uint64_t index) {
			return View(*this, { static_cast<int64_t>(index) });
		}

		View at() {
			return View(*this, {});
		}

		// 重载赋值
		// 一个张量给另一个张量赋值时
		// 如果规则为空，说明是一个新张量，此时需要设置规则
		// 否则仅拷贝数据
		Tensor& operator=(const Tensor& other) {
			if (_rule.name.empty()) {
				_rule = other._rule;
			}
			_data = other._data;
			return *this;
		}

		// 移动构造
		// 迁移张量数据，保留本地规则
		Tensor(Tensor&& other) noexcept {
			move_from(std::move(other));
		}

		// 移动赋值运算符
		Tensor& operator=(Tensor&& other) noexcept {
			if (this != &other) {
				move_from(std::move(other));
			}
			return *this;
		}

		// 在 Tensor 类的 public 部分添加拷贝构造函数
		Tensor(const Tensor& other) {
			_rule = other._rule;
			_data = other._data;
		}

		// 重载赋值
		// 给当前设定的维度写入数据
		template<typename T>
		Tensor& operator=(const std::vector<T>& data) {
			_data.write({}, data);
			return *this;
		}

		// 带检查的拷贝
		Tensor& copy(const Tensor& other) {
			if (_rule.name.empty()) {
				_rule = other._rule;
			}
			if (other._rule.type != _rule.type) {
				throw;
			}
			_data = other._data;
			return *this;
		}

		// Todo: 工作开始
		// 仅做基本类型检查，例如是否为整数
		// 自动根据实际类型和规则类型调整数据应用方式
		template<typename T>
		Tensor& start() {
			return *this;
		}

		// 张量静态属性
		const std::string name() const { return _rule.name; }

		// 规则形状（RuleShape）
		const std::vector<int64_t> shape() const { return _rule.shape; }
		const std::vector<int64_t> ruleShape() const { return _rule.shape; }

		const TensorType type() const { return _rule.type; }

		// 获取一维化后的数据
		const std::vector<char> getData() const {
			return _data.getData<char>();
		}

		template<typename T>
		const std::vector<T> getData() const {
			return _data.getData<T>();
		}

		// 获取当前的动态形状（CurrentShape）
		std::vector<int64_t> getShape() const {
			return _data.getCurrentShape();
		}

		// 检查当前张量形状是否符合规则
		bool check() {
			return _rule.check(_data.getCurrentShape());
		}

		// 获取一维化后的原始字节数据（连续内存）。
		// 适合推理后端直接引用，避免每帧构造临时 vector<T>。
		const std::vector<char>& getBytes() const {
			return _data.getBytes();
		}

		// 直接设置为稠密（连续）数据。
		// bytes 为按 type 的字节序列（与 shape 对应）。
		Tensor& setDense(std::vector<char>&& bytes, const std::vector<int64_t>& shape) {
			_rule.shape = shape;
			_data.setDenseBytes(shape, _rule.typeSize, std::move(bytes));
			return *this;
		}

	private:
		friend class View;

		TensorMeta _rule;
		TensorData _data;

		template<typename T>
		void writeAt(const std::vector<int64_t>& path, const std::vector<T>& data) {
			_data.write(path, data);
		}

		template<typename T>
		std::vector<T> readAt(const std::vector<int64_t>& path) const {
			return _data.read<T>(path);
		}

		void move_from(Tensor&& other) noexcept {
			if (_rule.name.empty()) {
				_rule = other._rule;
			}
			_data = std::move(other._data);
		}
	};
}
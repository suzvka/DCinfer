#pragma once
#include <stdexcept>

#include "TensorMods.h"

namespace DC {
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// DC::Tensor
// 张量对象
// 栈占用：184 字节
// 
// 用于创建输入或输出张量
// 在启动时设置张量规则，之后可以对实际数据进行检查，以匹配规则
// 在运行时而非编译时进行类型检查，请做好异常处理
// 自行指定类型时会按位进行转换，可能出现精度丢失，请小心使用
// 
// 	2025.7.31
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	class Tensor : public TensorDim{
	public:
		Tensor() = default;
		// 构造函数
		// 设定张量规则
		// - 张量名称
		// - 张量形状
		template<typename T>
		Tensor(
			const std::string& name,				// - 张量名称
			const std::vector<int64_t>& shape = {}  // - 张量形状
		) {
			_rule.name = name;
			_rule.type = typeid(T).name();
			_rule.typeSize = sizeof(T);
			_rule.shape = shape;
		}

		// 设置规则形状
		Tensor& setShape() {
			_rule.shape = getPath();
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
		Tensor& operator=(Tensor&& other) noexcept {
			if (this != &other) {
				if (_rule.name.empty()) {
					_rule = other._rule;
				}
				_data = std::move(other._data);
			}
			return *this;
		}

		// 重载赋值
		// 给当前设定的维度写入数据
		template<typename T>
		Tensor& operator=(const std::vector<T>& data) {
			_data.write(getPath(), data);
			return *this;
		}

		// 带检查的拷贝
		Tensor& copy(const Tensor& other) {
			if (_rule.name.empty()) {
				_rule = other._rule;
			}
			if (other._rule.type != _rule.type) {
				throw std::runtime_error("输入的 " + other._rule.name + " 类型不匹配，期望 " + _rule.type + "，实际得到 " + other._rule.type);
			}
			_data = other._data;
			return *this;
		}

		// 工作开始
		// 检查张量编辑环境
		template<typename T>
		Tensor& start(){
			if (_rule.type != typeid(T).name()) {
				throw std::runtime_error("张量 " + _rule.name +" 期望得到 " + _rule.type + "，实际得到" + typeid(T).name());
			}
			return *this;
		}

		// 张量静态属性
		const std::string name() const { return _rule.name; }
		const std::string type() const { return _rule.type; }
		const std::vector<int64_t> shape() const {return _rule.shape;}

		// 获取一维化后的数据
		const std::vector<char> getData() const {
			return _data.getData<char>();
		}

		template<typename T>
		const std::vector<T> getData() const {
			return _data.getData<T>();
		}

		// 获取当前的动态形状
		std::vector<int64_t> getShape() const {
			return _data.getCurrentShape();
		}

		// 检查当前张量形状是否符合规则
		bool check() {
			return _rule.check(_data.getCurrentShape());
		}

	private:
		TensorMeta _rule;
		TensorData _data;
	};
}
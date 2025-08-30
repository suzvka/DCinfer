#pragma once
#include <stdexcept>

#include "TensorMods.h"

namespace DC {
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// DC::Tensor
// 张量对象
// 栈占用：232 字节
// 
// 用于创建输入或输出张量
// 在启动时设置张量规则，之后可以对实际数据进行检查，以匹配规则
// 在运行时而非编译时进行类型检查，请做好异常处理
// 自行指定类型时会按位进行转换，可能出现精度丢失，请小心使用
// 
// 	2025.8.29
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	class Tensor : public TensorDim{
		using TensorType = TensorMeta::TensorType;
	public:
		Tensor() = default;
		virtual  ~Tensor() = default;

		Tensor(
			const std::string& name,				// - 张量名称
			const TensorType& type,					// - 张量类型
			const std::vector<int64_t>& shape = {},	// - 张量形状
			std::vector<char>&& data = {}			// - 张量数据
		) {
			_rule.name = name;
			_rule.type = type;
			_rule.typeName = TensorMeta::_typeMap.fromEnum(type).getTypeName();
			_rule.typeSize = TensorMeta::_typeMap.fromEnum(type).getTypeSize();
			_rule.shape = shape;
			_data = TensorData(shape, std::move(data));
		}

		template<typename T>
		static Tensor Create(
			const std::string& name,
			const std::vector<int64_t>& shape = {},
			std::vector<char>&& data = {}
		) {
			Type<T> typeInfo = Type<T>(typeid(T).name());

			Tensor tensor(
				name, 
				TensorMeta::_typeMap.toEnum(&typeInfo), 
				shape,
				std::move(data)
			);

			return tensor;
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
			if (other._rule.typeName != _rule.typeName) {
				throw std::runtime_error(
					"输入的 " + other._rule.name + 
					" 类型不匹配，期望 " + _rule.typeName +
					"，实际得到 " + other._rule.typeName
				);
			}
			_data = other._data;
			return *this;
		}

		// Todo: 工作开始
		// 仅做基本类型检查，例如是否为整数
		// 自动根据实际类型和规则类型调整数据应用方式
		template<typename T>
		Tensor& start(){
			return *this;
		}

		// 张量静态属性
		const std::string name() const { return _rule.name; }
		const std::string typeName() const { return _rule.typeName; }
		const std::vector<int64_t> shape() const {return _rule.shape;}
		const TensorType type() const { return _rule.type; }

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

		template<typename T> Type<T> getType();

		void move_from(Tensor&& other) noexcept {
			if (_rule.name.empty()) {
				_rule = other._rule;
			}
			_data = std::move(other._data);
		}
	};

	template<typename T> Type<T> Tensor::getType(){
		if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
			return Type<float>(TensorMeta::_typeMap.fromEnum(TensorType::Float));
		}
		else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
			return Type<int64_t>(TensorMeta::_typeMap.fromEnum(TensorType::Int));
		}
		else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
			return Type<uint64_t>(TensorMeta::_typeMap.fromEnum(TensorType::Uint));
		}
		else if constexpr (std::is_same_v<T, bool>) {
			return Type<bool>(TensorMeta::_typeMap.fromEnum(TensorType::Bool));
		}
		else {
			throw std::runtime_error("不支持的张量类型");
		}
	}
}
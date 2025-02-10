#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <variant>
#include "Value.h"

#include "tool.h"
#include "vector.h"
#include <any>

namespace DC {
	static const std::unordered_map<std::string, std::string> findName = {
		{ "float"   , typeid(float).name()		},
//		{ "uint8"   , typeid(uint8_t).name()	},
//		{ "int8"    , typeid(int8_t).name()		},
//		{ "int16"	, typeid(uint16_t).name()	},
//		{ "uint16"  , typeid(int16_t).name()	},
		{ "int32"   , typeid(int32_t).name()	},
		{ "int64"   , typeid(int64_t).name()	},
//		{ "string"  , typeid(std::string).name()},
		{ "bool"    , typeid(bool).name()		}
//		{ "double"  , typeid(double).name()		}
	};
		static const std::unordered_map<std::string, std::string> getName = {
		{ typeid(float).name(),			"float"	},
//		{ typeid(uint8_t).name(),		"uint8"	},
//		{ typeid(int8_t).name(),		"int8"	},
//		{ typeid(uint16_t).name(),		"int16"	},
//		{ typeid(int16_t).name(),		"uint16"},
		{ typeid(int32_t).name(),		"int32"	},
		{ typeid(int64_t).name(),		"int64"	},
//		{ typeid(std::string).name(),	"string"},
		{ typeid(bool).name(),			"bool"	}
//		{ typeid(double).name(),		"double"}
	};
	class Value;
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// DC::Tensor
// 张量对象
// 栈占用：552 字节
// 
// 用于创建输入或输出张量
// 在启动时设置张量规则，之后可以对实际数据进行检查，以匹配规则
// 在运行时而非编译时进行类型检查，请做好异常处理
// 所有张量编辑操作都在内存中进行，显式调用 load() 后才会加载到显存
// 默认以规则中的类型加载显存，但也可以自行指定
// 自行指定类型时会按位进行转换，可能出现精度丢失，请小心使用
// 
// 	2025.1.20
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	class Tensor {
		class TensorEdit;
	public:
		Tensor();
		// 构造函数
		// 设定张量规则
		// - 张量名称
		// - 张量形状
		Tensor(
			const std::string& name,            // - 张量名称
			const std::string& type,            // - 张量类型
			const std::vector<int64_t>& shape   // - 张量形状
		);
		Tensor(
			const std::string& name,            // - 张量名称
			const Ort::Value& newValue          // - 张量参数
		);

		// 智能拷贝
		// 在创建新对象时完整拷贝所有数据
		// 在给旧对象赋值时不会覆盖张量规则
		Tensor& operator=(const Tensor& newObj) {
			if (_name.empty()) {
				_name = newObj._name;
			}
			if (_type.empty()) {
				_type = newObj._type;
			}
			if (_shape.empty()) {
				_shape = newObj._shape;
			}
			_tensorData = newObj._tensorData;
			return *this;
		}

		// 工作开始
		// 启用张量编辑环境
		template<typename T>
		tensorPro<T>& start(){
			if (_type != getName.at(typeid(T).name())) {
				throw std::runtime_error("张量 " + _name +" 期望得到 " + _type + "，实际得到" + typeid(T).name());
			}
			return _tensorData;
		}

		// 张量静态属性
		const std::string name() const { return _name; }
		const std::string type() const { return _type; }
		const std::vector<int64_t> shape() const {return _shape;}

		// 从另一个张量中加载数据
		Tensor& copy(const Tensor& input) {
			_tensorData = input._tensorData;
			return *this;
		}

		// 获取一维化后的数据
		const std::vector<char> getVector() const {
			return _tensorData.getVector();
		}
		template<typename T>
		const std::vector<T> getVector() const {
			return _tensorData.getVector<T>();
		}

		// 获取当前的动态形状
		std::vector<int64_t> getShape() {
			std::vector<int64_t> converted_vector;
			for (uint64_t value : _tensorData.getShape()) {
				converted_vector.push_back(static_cast<int64_t>(value));
			}
			return converted_vector;
		}

		// 检查当前形状或输入形状是否符合规则
		bool check(const std::vector<int64_t>& shape = {}) {
			std::vector<int64_t> tobeCheck;
			shape.empty() ? tobeCheck = getShape() : tobeCheck = shape;
			if (tobeCheck.size() != _shape.size()) {
				return false;
			}
			int64_t i = 0;
			for (auto dim: tobeCheck) {
				if (_shape[i] != -1 && dim != _shape[i]) {
					return false;
				}
				i++;
			}
			return true;
		}

		// 将数据加载到显存
		template<typename T>
		bool load() {
			value->load<T>(getVector<T>(), getShape());
			return true;
		}
		// 默认根据规则类型加载
		bool load() {
			if (_type == getName.at(typeid(float).name())) {
				load<float>();
			}
			else if (_type == getName.at(typeid(int32_t).name())) {
				load<int32_t>();
			}
			else if (_type == getName.at(typeid(int64_t).name())) {
				load<int64_t>();
			}
			else if (_type == getName.at(typeid(bool).name())) {
				load<bool>();
			}
			return true;
		}

		// 获取用当前数据加载的 Ort::Value
		Ort::Value getValue()const;

	private:
		// 自定义规则
		std::string _name;                // 张量名称
		std::string _type;                // 张量类型
		std::vector<int64_t> _shape;      // 张量形状

		std::shared_ptr<Value> value;	  // 张量显存管理器

		//============================================================
		// 张量编辑环境
		// 将模板 tensor 对象封装，去除其模板特性
		// 这样就能实现运行时类型匹配
		class TensorEdit : 
			public tensorPro<float>, 
			public tensorPro<int32_t>,
			public tensorPro<int64_t>,
			public tensorPro<bool>
		{
		public:
			TensorEdit(const std::string& type = "")
				: _type(type){}

			void set(const std::vector<int64_t>& shape, const std::vector<char>& data) {
				std::vector<uint64_t> myshape;
				for (auto& dim : shape) {
					myshape.push_back(dim);
				}

				if (_type == "float") {
					// 计算需要补零的字节数
					size_t remainder = data.size() % sizeof(float);
					size_t paddingSize = (remainder == 0) ? 0 : sizeof(float) - remainder;

					// 创建一个临时缓冲区，复制原始数据并补零
					std::vector<char> paddedData = data; // 复制原始数据
					paddedData.insert(paddedData.end(), paddingSize, 0); // 在尾部补零

					// 转换为 float 并处理
					std::vector<float> temp;
					temp.reserve(paddedData.size() / sizeof(float));
					for (size_t i = 0; i < paddedData.size(); i += sizeof(float)) {
						float value;
						std::memcpy(&value, &paddedData[i], sizeof(float));
						temp.push_back(value);
					}

					// 设置到 tensorPro
					tensorPro<float>::set(myshape, temp);
				}
				else if (_type == "int32") {
					// 计算需要补零的字节数
					size_t remainder = data.size() % sizeof(float);
					size_t paddingSize = (remainder == 0) ? 0 : sizeof(float) - remainder;
					// 创建一个临时缓冲区，复制原始数据并补零
					std::vector<char> paddedData = data; // 复制原始数据
					paddedData.insert(paddedData.end(), paddingSize, 0); // 在尾部补零
					std::vector<int32_t> temp;
					temp.reserve(paddedData.size() / sizeof(int32_t));
					for (size_t i = 0; i < paddedData.size(); i += sizeof(int32_t)) {
						float value;
						std::memcpy(&value, &paddedData[i], sizeof(float));
						temp.push_back(value);
					}
					tensorPro<int32_t>::set(myshape, temp);
				}
				else if (_type == "int64") {
					// 计算需要补零的字节数
					size_t remainder = data.size() % sizeof(float);
					size_t paddingSize = (remainder == 0) ? 0 : sizeof(float) - remainder;
					// 创建一个临时缓冲区，复制原始数据并补零
					std::vector<char> paddedData = data; // 复制原始数据
					paddedData.insert(paddedData.end(), paddingSize, 0); // 在尾部补零
					std::vector<int64_t> temp;
					temp.reserve(paddedData.size() / sizeof(int64_t));
					for (size_t i = 0; i < paddedData.size(); i += sizeof(int64_t)) {
						float value;
						std::memcpy(&value, &paddedData[i], sizeof(float));
						temp.push_back(value);
					}
					tensorPro<int64_t>::set(myshape, temp);
				}
				else if (_type == "bool") {
					std::vector<bool> temp;
					for (auto &it : data) {
						it != NULL ? temp.push_back(true) : temp.push_back(false);
					}
					tensorPro<bool>::set(myshape, temp);
				}
			}

			std::vector<char> getVector() const {
				if (_type == "float") {
					return tensorPro<float>::getData<char>();
				}
				else if (_type == "int32") {
					return tensorPro<int32_t>::getData<char>();
				}
				else if (_type == "int64") {
					return tensorPro<int64_t>::getData<char>();
				}
				else if (_type == "bool") {
					return tensorPro<bool>::getData<char>();
				}
			}

			std::vector<uint64_t> getShape() const {
				if (_type == "float") {
					return tensorPro<float>::getShape();
				}
				else if (_type == "int32") {
					return tensorPro<int32_t>::getShape();
				}
				else if (_type == "int64") {
					return tensorPro<int64_t>::getShape();
				}
				else if (_type == "bool") {
					return tensorPro<bool>::getShape();
				}
			}

			template<typename T>
			std::vector<T> getVector() const {
				if (_type == "float") {
					return tensorPro<float>::getData<T>();
				}
				else if (_type == "int32") {
					return tensorPro<int32_t>::getData<T>();
				}
				else if (_type == "int64") {
					return tensorPro<int64_t>::getData<T>();
				}
				else if (_type == "bool") {
					return tensorPro<bool>::getData<T>();
				}
			}

			TensorEdit& operator[](uint64_t index) {
				if (_type == "float") {
					tensorPro<float>::operator[](index);
				}
				else if (_type == "int32") {
					tensorPro<int32_t>::operator[](index);
				}
				else if (_type == "int64") {
					tensorPro<int64_t>::operator[](index);
				}
				else if (_type == "bool") {
					tensorPro<bool>::operator[](index);
				}
				return *this;
			}

		private:
			std::string _type;
		};

		// 张量数据库
		TensorEdit _tensorData;
	};
}
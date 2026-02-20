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
// ��������
// ջռ�ã�552 �ֽ�
// 
// ���ڴ���������������
// ������ʱ������������֮����Զ�ʵ�����ݽ��м�飬��ƥ�����
// ������ʱ���Ǳ���ʱ�������ͼ�飬�������쳣����
// ���������༭���������ڴ��н��У���ʽ���� load() ��Ż���ص��Դ�
// Ĭ���Թ����е����ͼ����Դ棬��Ҳ��������ָ��
// ����ָ������ʱ�ᰴλ����ת�������ܳ��־��ȶ�ʧ����С��ʹ��
// 
// 	2025.1.20
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	
	class Tensor {
		class TensorEdit;
	public:
		Tensor();
		// ���캯��
		// �趨��������
		// - ��������
		// - ������״
		Tensor(
			const std::string& name,            // - ��������
			const std::string& type,            // - ��������
			const std::vector<int64_t>& shape   // - ������״
		);
		Tensor(
			const std::string& name,            // - ��������
			const Ort::Value& newValue          // - ��������
		);

		// ���ܿ���
		// �ڴ����¶���ʱ����������������
		// �ڸ��ɶ���ֵʱ���Ḳ����������
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

		// ������ʼ
		// ���������༭����
		template<typename T>
		tensorPro<T>& start(){
			if (_type != getName.at(typeid(T).name())) {
				throw std::runtime_error("���� " + _name +" �����õ� " + _type + "��ʵ�ʵõ�" + typeid(T).name());
			}
			return _tensorData;
		}

		// ������̬����
		const std::string name() const { return _name; }
		const std::string type() const { return _type; }
		const std::vector<int64_t> shape() const {return _shape;}

		// ����һ�������м�������
		Tensor& copy(const Tensor& input) {
			_tensorData = input._tensorData;
			return *this;
		}

		// ��ȡһά���������
		const std::vector<char> getVector() const {
			return _tensorData.getVector();
		}
		template<typename T>
		const std::vector<T> getVector() const {
			return _tensorData.getVector<T>();
		}

		// ��ȡ��ǰ�Ķ�̬��״
		std::vector<int64_t> getShape() {
			std::vector<int64_t> converted_vector;
			for (uint64_t value : _tensorData.getShape()) {
				converted_vector.push_back(static_cast<int64_t>(value));
			}
			return converted_vector;
		}

		// ��鵱ǰ��״��������״�Ƿ���Ϲ���
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

		// �����ݼ��ص��Դ�
		template<typename T>
		bool load() {
			value->load<T>(getVector<T>(), getShape());
			return true;
		}
		// Ĭ�ϸ��ݹ������ͼ���
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

		// ��ȡ�õ�ǰ���ݼ��ص� Ort::Value
		Ort::Value getValue()const;

		//============================================================
		// DC::Tensor::View
		// Provides unified read/write access to a Tensor without a
		// separate ConstView class. Read-only access is enforced via
		// the C++ const system:
		//   - non-const View  -> read + write (mutable access)
		//   - const View      -> read-only (only const methods exposed)
		// A View constructed from a const Tensor also supports only
		// read-only access (non-const methods will throw at runtime).
		// Friend access is granted by nesting View inside Tensor.
		class View {
		public:
			// Construct a mutable View from a non-const Tensor
			explicit View(Tensor& tensor)
				: _tensor(&tensor), _constTensor(&tensor) {}

			// Construct a read-only View from a const Tensor
			explicit View(const Tensor& tensor)
				: _tensor(nullptr), _constTensor(&tensor) {}

			// Read-only accessors (const-qualified)
			// Available on both mutable and read-only (const) Views
			const std::string&        name()  const { return _constTensor->_name; }
			const std::string&        type()  const { return _constTensor->_type; }
			const std::vector<int64_t>& shape() const { return _constTensor->_shape; }
			std::vector<char> getVector() const {
				return _constTensor->_tensorData.getVector();
			}
			template<typename T>
			std::vector<T> getVector() const {
				return _constTensor->_tensorData.getVector<T>();
			}
			Ort::Value getValue() const { return _constTensor->getValue(); }

			// Mutable accessors (non-const-qualified)
			// Callable only on a non-const View; throws if the View was
			// constructed from a const Tensor (_tensor is nullptr).
			template<typename T>
			tensorPro<T>& start() {
				if (!_tensor) throw std::logic_error("Cannot modify a read-only View");
				return _tensor->start<T>();
			}
			View& copy(const Tensor& input) {
				if (!_tensor) throw std::logic_error("Cannot modify a read-only View");
				_tensor->copy(input);
				return *this;
			}
			bool load() {
				if (!_tensor) throw std::logic_error("Cannot modify a read-only View");
				return _tensor->load();
			}
			template<typename T>
			bool load() {
				if (!_tensor) throw std::logic_error("Cannot modify a read-only View");
				return _tensor->load<T>();
			}

		private:
			// _tensor is non-null only when View is constructed from a non-const Tensor.
			// It is null when constructed from a const Tensor, making mutable methods
			// throw at runtime. A const View always prevents calling non-const methods
			// at compile time regardless of how it was constructed.
			Tensor*       _tensor;
			const Tensor* _constTensor;  // always valid
		};

		// Factory methods – return a View bound to this Tensor.
		// Calling view() on a const Tensor yields a read-only View.
		View view()       { return View(*this); }
		View view() const { return View(*this); }

	private:
		// �Զ������
		std::string _name;                // ��������
		std::string _type;                // ��������
		std::vector<int64_t> _shape;      // ������״

		std::shared_ptr<Value> value;	  // �����Դ������

		//============================================================
		// �����༭����
		// ��ģ�� tensor �����װ��ȥ����ģ������
		// ��������ʵ������ʱ����ƥ��
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
					// ������Ҫ������ֽ���
					size_t remainder = data.size() % sizeof(float);
					size_t paddingSize = (remainder == 0) ? 0 : sizeof(float) - remainder;

					// ����һ����ʱ������������ԭʼ���ݲ�����
					std::vector<char> paddedData = data; // ����ԭʼ����
					paddedData.insert(paddedData.end(), paddingSize, 0); // ��β������

					// ת��Ϊ float ������
					std::vector<float> temp;
					temp.reserve(paddedData.size() / sizeof(float));
					for (size_t i = 0; i < paddedData.size(); i += sizeof(float)) {
						float value;
						std::memcpy(&value, &paddedData[i], sizeof(float));
						temp.push_back(value);
					}

					// ���õ� tensorPro
					tensorPro<float>::set(myshape, temp);
				}
				else if (_type == "int32") {
					// ������Ҫ������ֽ���
					size_t remainder = data.size() % sizeof(float);
					size_t paddingSize = (remainder == 0) ? 0 : sizeof(float) - remainder;
					// ����һ����ʱ������������ԭʼ���ݲ�����
					std::vector<char> paddedData = data; // ����ԭʼ����
					paddedData.insert(paddedData.end(), paddingSize, 0); // ��β������
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
					// ������Ҫ������ֽ���
					size_t remainder = data.size() % sizeof(float);
					size_t paddingSize = (remainder == 0) ? 0 : sizeof(float) - remainder;
					// ����һ����ʱ������������ԭʼ���ݲ�����
					std::vector<char> paddedData = data; // ����ԭʼ����
					paddedData.insert(paddedData.end(), paddingSize, 0); // ��β������
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

		// �������ݿ�
		TensorEdit _tensorData;
	};
}